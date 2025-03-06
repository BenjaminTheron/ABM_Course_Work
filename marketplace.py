import numpy as np
import pandas as pd
import datetime
from order import Order

# TODO: What is the effect of traders being able to select multiple market places?
class MarketPlace:
  """
  Outlines the class for the marketplace to be used by the traders and auctioneer(s).
  """

  def __init__(self):
    # init order to initialise the order book
    init_order = [[-1, -1, "bid", 0, 0, str(datetime.datetime.now())]]
    # The data to be stored with each order on the LOB
    self.cols = ['traderID', 'orderID', 'orderType', 'price', 'quantity', 'time']

    self.limit_order_book = pd.DataFrame(init_order, columns=self.cols)

  def Add_Order(self, order):
    """
    Given an order object that has been validated (can be added to the LOB),
    it appends the order to the LOB.
    """
    # Update the time of the order
    order.Set_Time(datetime.datetime.now())
    
    # Converts an order object into an DataFrame friendly array
    new_order = [[order.trader_id, order.order_id, order.order_type, order.price,\
                  order.quantity, str(order.time)]]

    self.limit_order_book = pd.concat([self.limit_order_book,\
                                      pd.DataFrame(new_order, columns=self.cols)],
                                      ignore_index=True)

  def Remove_Order(self, order):
    """
    Given an order object that has been matched, expired or deleted, it removes
    the order from the LOB.
    """
    # This function is polymorphic, can remove orders which are provided in array format or
    # as Order objects
    if type(order) is Order:
      # Find the index to remove (by traderID)
      index_to_remove = np.where(
          self.limit_order_book['traderID'] == order.trader_id)[0][0]

      # Remove the corresponding row from the dataframe
      self.limit_order_book = self.limit_order_book.drop(index_to_remove)
      # Reset the index in the dataframe
      self.limit_order_book = self.limit_order_book.reset_index(drop=True)
      
    elif type(order) is list:
      # Find the index to remove
      index_to_remove = np.where(
          self.limit_order_book['traderID'] == order[0][0])[0][0]

      # Remove the corresponding row from the dataframe
      self.limit_order_book = self.limit_order_book.drop(index_to_remove)
      # Reset the index in the dataframe
      self.limit_order_book = self.limit_order_book.reset_index(drop=True)

  def Match_Orders(self, order_bid, order_ask, auctioneer):
    """
    Given two orders, this function 'matches' them, generating the appropriate
    trade and updating all the relevant positions
    """
    print()

  def Store_Orders(self):
    """
    Adds any matched trades to the dataframe the auctioneer uses to store the list
    of all successfully matched trades.
    """
    print()
