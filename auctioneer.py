import datetime
import math
from bisect import bisect_left
import pandas as pd
from trade_log import Trade, TradeLog

class Auctioneer:
    """
    Outlines the class for marketplace auctioneer who matches the bid and ask
    orders on the limit order book. One per market.
    """
    def __init__(self, auctioneer_id, reg_fees, inf_fees, transac_fees,
                 shout_fees, profit_fees, min_trade_size):
        self.auctioneer_id = auctioneer_id
        self.reg_fees = reg_fees
        self.inf_fees = inf_fees
        self.transac_fees = transac_fees
        self.shout_fees = shout_fees
        self.profit_fees = profit_fees
        self.min_trade_size = min_trade_size
        
        # Columns for logging
        self.cols = ['traderID', 'orderID', 'orderType', 'price', 'quantity', 'time']
        
        # Initialize trade logs using efficient structure
        self.trade_log = TradeLog(self.cols)
        self.submitted_orders_log = TradeLog(self.cols)
    
    def shout_accepting_policy(self, order, order_book):
        """
        Determines whether an incoming trade is accepted by the auctioneer and placed
        on the limit order book.
        """
        # Log the submitted order
        new_order_record = [[
            order.trader_id, order.order_id, order.order_type,
            order.price, order.quantity, str(datetime.datetime.now())
        ]]
        self.submitted_orders_log.add_trade(new_order_record)
        
        # Check minimum trade size
        if order.quantity < self.min_trade_size:
            return False
        
        # Add the order to the book
        order_book.add_order(order)
        return True
    
    def pricing_policy(self, bid_price, ask_price):
        """
        Determines the final transaction price of matched orders.
        Uses the midpoint between bid and ask prices.
        
        Args:
            bid_price: The bid order price
            ask_price: The ask order price
            
        Returns:
            float: The transaction price
        """
        return round((bid_price + ask_price) / 2, 2)
    
    def batch_clearing_policy(self, order_book, step):
        """
        Implements batch matching strategy, matching orders in price-time priority.
        
        Args:
            order_book: The OrderBook instance
            
        Returns:
            int: Number of trades executed
        """
        trades_executed = 0
        
        # Continue matching while there are overlapping orders or orders to match
        while order_book.has_crossing_orders():
          best_bid = order_book.get_best_bid()
          best_ask = order_book.get_best_ask()

          if not best_bid or not best_ask:
            break
            
          # Determine transaction price and quantity
          transaction_price = self.pricing_policy(best_bid.price, best_ask.price)
          transaction_qty = min(best_bid.quantity, best_ask.quantity)
            
          # Create and log the trade
          trade = Trade(best_bid, best_ask, transaction_price, transaction_qty, step)
          self.trade_log.add_trade(trade)
          trades_executed += 1
            
          # Update order quantities
          best_bid.quantity -= transaction_qty
          best_ask.quantity -= transaction_qty
            
          # Remove filled orders
          if best_bid.quantity <= 0:
              order_book.remove_order(best_bid.order_id)
            
          if best_ask.quantity <= 0:
              order_book.remove_order(best_ask.order_id)
        
        return trades_executed

    def has_crossing_orders(self):
          ''' 
          Checks if there are any crossing orders in the orderbook
          '''
          
          best_bid = self.get_best_bid()
          best_ask = self.get_best_ask()

          if best_bid is None or best_ask is None:
            return False
          
          return best_bid.price >= best_ask.price