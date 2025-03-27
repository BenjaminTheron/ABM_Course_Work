import numpy as np
import pandas as pd
import datetime
from orderbook import OrderBook, Order
import random

# TODO: What is the effect of traders being able to select multiple market places?
class MarketPlace:
  """
  Outlines the class for the marketplace to be used by the traders and auctioneer(s).
  """

  def __init__(self, auctioneer,parameters=None):
    self.order_book = OrderBook()
    self.auctioneer = auctioneer
    self.traders = {}
    self.fundamental_value = 100
    self.closing_price_history = [100]
    self.spread_history = [0]
    self.hft_sell_concentration_history = [0]
    self.lft_sell_concentration_history = [0]
    self.shock_events = []  # To track when shocks were applied
    self.parameters = parameters
    
    # Initialize shock steps
    self.shock_steps = []
    self._generate_random_shock_steps()

  def update_fundamental_value(self, delta=0.001, sigma_y=0.02,step=0):
    """Update the fundamental value according to geometric random walk"""

    shock = np.random.normal(0,sigma_y)
    self.fundamental_value *= (1+delta) * (1+shock)
    if step in self.shock_steps:
        shock_size = self.parameters.get("shock_size", 0.10)
        pre_shock_value = self.fundamental_value
        
        # Apply the shock (reduce value by shock_size)
        self.fundamental_value *= (1 - shock_size)
        
        # Mark the shock for visualization
        self.shock_events.append({
            "step": step,
            "pre_shock_value": pre_shock_value,
            "post_shock_value": self.fundamental_value
        })
        
        print(f"Fundamental value shock at step {step}: {pre_shock_value:.2f} → {self.fundamental_value:.2f} (-{shock_size*100:.0f}%)")
  
  def process_trades(self):
    """
    Process trades to update trader positions and budgets.
    Simplified to only handle Trade objects.
    """
    unprocessed_trades = self.auctioneer.trade_log.get_unprocessed_trades()

    if not unprocessed_trades:
        return

    for trade in unprocessed_trades:
        # Extract data from Trade object
        bid_trader_id = trade.bid_order.trader_id
        ask_trader_id = trade.ask_order.trader_id
        transaction_price = trade.price
        quantity = trade.quantity
        
        # Update buyer (bid trader)
        bid_trader = self.traders.get(bid_trader_id)
        if bid_trader:
            bid_trader.budget_size -= transaction_price * quantity
            bid_trader.stock += quantity
        
        # Update seller (ask trader)
        ask_trader = self.traders.get(ask_trader_id)
        if ask_trader:
          ask_trader.budget_size += transaction_price * quantity
          ask_trader.stock -= quantity
    
    self.auctioneer.trade_log.mark_trades_processed()
    
  def match_orders(self, step):
    """Match and execute orders in the book"""
    trades_executed = self.auctioneer.batch_clearing_policy(self.order_book, step)
    
    if trades_executed > 0:
        self.process_trades()
        
    highest_price_trade = self.auctioneer.trade_log.get_highest_price_trade(step)
    
    if highest_price_trade:
        # Use the highest price trade for this step
        new_price = highest_price_trade.price
        self.closing_price_history.append(new_price)
        #print(f"Step {step}: New closing price (highest trade): {new_price:.2f}")
    else:
        # If no trades, use previous closing price
        old_price = self.closing_price_history[-1] if self.closing_price_history else 100
        self.closing_price_history.append(old_price)
        #print(f"Step {step}: No trades - using previous price: {old_price:.2f}")
    
    # Print order book state
    best_bid = self.order_book.get_best_bid()
    best_ask = self.order_book.get_best_ask()
    
    if best_bid and best_ask:
        spread = best_ask.price - best_bid.price
        self.spread_history.append(spread)
        #print(f"Step {step}: Best bid: {best_bid.price:.2f}, Best ask: {best_ask.price:.2f}, Spread: {spread:.2f}")
    else:
      previous_spread = self.spread_history[-1] if self.spread_history else 0
      self.spread_history.append(previous_spread)

    return trades_executed
  
  def calculate_concentration_metrics(self, step):
    """
    Calculate sell concentration for both trader types.
    
    Args:
        step: Current simulation step
        
    Returns:
        tuple: (hft_sell_concentration, lft_sell_concentration)
    """
    # Count volumes by trader type and order type
    hft_sell_volume = 0
    hft_buy_volume = 0
    lft_sell_volume = 0
    lft_buy_volume = 0
    
    # Calculate sell volumes - all ask orders
    for price, orders in self.order_book.asks_by_price.items():
        for order in orders:
            if hasattr(order, 'agent_type'):
                if order.agent_type == "HF":
                    hft_sell_volume += order.quantity
                elif order.agent_type == "LF":
                    lft_sell_volume += order.quantity
    
    # Calculate buy volumes - all bid orders
    for price, orders in self.order_book.bids_by_price.items():
        for order in orders:
            if hasattr(order, 'agent_type'):
                if order.agent_type == "HF":
                    hft_buy_volume += order.quantity
                elif order.agent_type == "LF":
                    lft_buy_volume += order.quantity
    
    # Calculate total volumes by trader type
    hft_total_volume = hft_sell_volume + hft_buy_volume
    lft_total_volume = lft_sell_volume + lft_buy_volume
    
    # Calculate sell concentration ratios
    hft_sell_concentration = hft_sell_volume / hft_total_volume if hft_total_volume > 0 else 0
    lft_sell_concentration = lft_sell_volume / lft_total_volume if lft_total_volume > 0 else 0
    
    # Store the concentrations in history
    self.hft_sell_concentration_history.append(hft_sell_concentration)
    self.lft_sell_concentration_history.append(lft_sell_concentration)
    
    # Print high concentration values for debugging
    '''
    if hft_sell_concentration > 0.8:
        print(f"Step {step}: High HFT sell concentration: {hft_sell_concentration:.2f}")
    if lft_sell_concentration < 0.2:  # Equivalent to high buy concentration
        print(f"Step {step}: High LFT buy concentration: {(1-lft_sell_concentration):.2f}")
    '''

  def register_trader(self, trader):
    self.traders[trader.trader_id] = trader
  
  def add_order(self, order):
    return self.auctioneer.shout_accepting_policy(order, self.order_book)
  
  def _generate_random_shock_steps(self):
    """
    Generate random timestamps for fundamental value shocks.
    Called during initialization.
    """
    num_shocks = self.parameters.get("num_shocks", 0)
    if num_shocks <= 0:
        return
        
    min_shock_step = self.parameters.get("min_shock_step", 100)
    max_shock_step = self.parameters.get("max_shock_step", 800)
    
    # Create randomly spaced shock times within the range
    shock_range = max_shock_step - min_shock_step
    
    if num_shocks == 1:
        # For one shock, pick a random time in the range
        self.shock_steps = [random.randint(min_shock_step, max_shock_step)]
    else:
        # For multiple shocks, divide the range into segments and pick a random time in each segment
        segment_size = shock_range / num_shocks
        
        for i in range(num_shocks):
            segment_start = min_shock_step + int(i * segment_size)
            segment_end = min_shock_step + int((i + 1) * segment_size) - 1
            shock_step = random.randint(segment_start, segment_end)
            self.shock_steps.append(shock_step)
            
    # Sort the shock steps to ensure they're in chronological order
    self.shock_steps.sort()
    
    # Print only if there are actually shocks scheduled
    if self.shock_steps:
        print(f"Initialized with {num_shocks} random shock(s) at steps: {self.shock_steps}")

  def get_fundamental_value(self):
    """Return the current fundamental value"""
    return self.fundamental_value
    
  def get_trade_log_df(self):
    return self.auctioneer.trade_log.to_dataframe()
  
  def get_price_at(self, index):
    try:
        return self.closing_price_history[index]
    except IndexError:
        # Handle out-of-bounds 
        return None

  def get_price_history(self):
    return self.closing_price_history

  def get_history_length(self):
    return len(self.closing_price_history)
    
  def get_submitted_orders_df(self):
    return self.auctioneer.submitted_orders_log.to_dataframe()
  
  def get_last_price(self):
    """Get the last closing price"""
    if not self.closing_price_history:
        return 100  # Default starting price
    return self.closing_price_history[-1]
  
  def get_price_change_percent(self):
    """Get the percentage change in price"""
    if len(self.closing_price_history) < 2:
        return 0
        
    prev_price = self.closing_price_history[-2]
    curr_price = self.closing_price_history[-1]
    
    if prev_price == 0:
        return 0
        
    return (curr_price - prev_price) / prev_price

  def get_current_spread(self):
    if not self.spread_history:
      return 0
    else:
      return self.spread_history[-1]

  def get_spread_history(self, max=None):
    if max is None:
      return self.spread_history
    else:
      return self.spread_history[-max:]

  def get_hft_sell_concentration_history(self):
    return self.hft_sell_concentration_history
      
  def get_lft_sell_concentration_history(self):
    return self.lft_sell_concentration_history
  
  def get_volume(self):
    return self.auctioneer.trade_log.get_total_volume()
  

