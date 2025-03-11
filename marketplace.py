import numpy as np
import pandas as pd
import datetime
from orderbook import OrderBook, Order

# TODO: What is the effect of traders being able to select multiple market places?
class MarketPlace:
  """
  Outlines the class for the marketplace to be used by the traders and auctioneer(s).
  """

  def __init__(self, auctioneer):
    self.order_book = OrderBook()
    self.auctioneer = auctioneer
    self.traders = {}

  def register_trader(self, trader):
    self.traders[trader.trader_id] = trader

  def add_order(self, order):
    return self.auctioneer.shout_accepting_policy(order, self.order_book)
    
  def match_orders(self, step):
    trades_executed = self.auctioneer.batch_clearing_policy(self.order_book, step)
    if trades_executed > 0:
      self.process_trades()
    return trades_executed
    
  def get_trade_log_df(self):
    return self.auctioneer.trade_log.to_dataframe()
    
  def get_submitted_orders_df(self):
    return self.auctioneer.submitted_orders_log.to_dataframe()
  
  def process_trades(self):
    unprocessed_trades = self.auctioneer.trade_log.get_unprocessed_trades()

    if not unprocessed_trades:
      return

    i = 0
    while i < len(unprocessed_trades):
      bid_record = unprocessed_trades[i]
      bid_trader_id = bid_record[0]
      bid_price = bid_record[3]
      quantity = bid_record[4]
      transaction_price = bid_record[3]

      ask_record = unprocessed_trades[i+1] if i+1 < len(unprocessed_trades) else None
      if not ask_record or ask_record[2] != 'ask':
            i += 1
            continue

      ask_trader_id = ask_record[0]
      ask_price = ask_record[3]

      bid_trader = self.traders.get(bid_trader_id)
      # trader had balance decreased at time of submission. refund the difference after transaction price found
      if bid_trader:
        refund = (bid_price - transaction_price) * quantity
        bid_trader.budget_size += refund
        bid_trader.stock += quantity
      
      ask_trader = self.traders.get(ask_trader_id)
      if ask_trader:
        ask_trader.budget_size += transaction_price * quantity
      i += 2
    
    self.auctioneer.trade_log.mark_trades_processed()

