import datetime
import math
from bisect import bisect_left

# TODO: What is the impact of different matching strategies?
class Auctioneer:
  """
  Outlines the class for marketplace auctioneer who matches the bid and ask
  orders on the limit order book. One per market.
  """
  def __init__(self, auctioneerID, reg_fees, inf_fees, transac_fees,\
               shout_fees, profit_fees, min_trade_size):
    self.auctioneerID = auctioneerID
    self.reg_fees = reg_fees
    self.inf_fees = inf_fees
    self.transac_fees = transac_fees
    self.shout_fees = shout_fees
    self.profit_fees = profit_fees
    self.min_trade_size = min_trade_size
    
    # An init trade used to initialise the successful trades log
    init_trade = [[-1, -1, "bid", 0, 0, str(datetime.datetime.now())]]
    # The data to be stored with each order on the LOB
    self.cols = ['traderID', 'orderID', 'orderType', 'price', 'quantity', 'time']

    self.trade_log = pd.DataFrame(init_trade, columns=self.cols) # Dataframe storing all successful trades
    self.submitted_orders_log = pd.DataFrame(init_trade, columns=self.cols) # Dataframe storing every submitted order

  def Shout_Accepting_Policy(self, order, marketplace):
    """
    Determines whether an incoming trade is accepted by the auctioneer and placed
    on the limit order book in the marketplace.
    """
    # Add the order to the list of all submitted trades
    new_order = [[order.trader_id, order.order_id, order.order_type, order.price,\
                  order.quantity, str(datetime.datetime.now())]]

    self.submitted_orders_log = pd.concat([self.submitted_orders_log,\
                          pd.DataFrame(new_order, columns=self.cols)],
                          ignore_index=True)
    
    # Currently this amounts to checking if the trader already has an active order
    # and that the order meets a minimum size
    active_traders = list(marketplace.limit_order_book['traderID'])

    if order.trader_id in active_traders or order.quantity < self.min_trade_size:
      return False
    else:
      # Add the order to the LOB
      marketplace.Add_Order(order)
      return True
      
  def Clearing_Policy(self, match, marketplace):
    """
    This is the auctioneer's clearing/ matching strategy, this determines how the bid
    and ask orders are matched.

    TODO: Implement Pro-Rata scheduling, implement orders with a constant proportion of all
    orders placed at the best bid price, independently of the arrival times of the orders.
    """
    if match == "batch":
      # This method matches orders as-you-go, so whenever this function is called
      # Splits the LOB into Bids and asks
      bids = marketplace.limit_order_book.loc[marketplace.limit_order_book['orderType'] == 'bid']
      asks = marketplace.limit_order_book.loc[marketplace.limit_order_book['orderType'] == 'ask']

      # Sort the bids in descending order
      bids = bids.sort_values('price', ascending=False)
      # Sort the asks in ascending order
      asks = asks.sort_values('price')

      # Matches the orders on a first come first serve basis (those at the top of the LOB are
      # matched first) -> THIS IS JUST TO CREATE BASE FUNCTIONALITY
      # To prevent some buyers from paying much more than they're asking and some sellers from
      # selling for much less than they are asking, only the top half of the LOB is cleared

      index = 0
      # Go though the asks and find the closest match
      while index <= int(len(asks)/2) and len(bids) > 0:
        # Stores the current ask order
        current_ask = asks.iloc(index)
        # Check if there is a direct match
        while len(bids.loc[bids['price'] == current_ask.price]) > 0:
          compatible_bid = bids.loc[bids['price'] == current_ask.price]
          # Select the first compatible bid
          compatible_bid = compatible_bid.iloc[0]

          if compatible_bid.quantity - current_ask.quantity < 0:
            # This is the case for a partial match
            # Match all the bid you can and move on to the next bid

            # Log the bid trade and remove it from the LOB
            executed_bid = [[compatible_bid.trader_id, compatible_bid.order_id,\
               compatible_bid.order_type, compatible_bid.price,\
               compatible_bid.quantity, str(datetime.datetime.now())]]
            self.Add_Trade(executed_bid)
            marketplace.Remove_Order(compatible_bid)

            # Log the ask trade and update it (partially matched so not removed)
            executed_ask = [[current_ask.trader_id, current_ask.order_id,\
                             current_ask.order_type, current_ask.price,\
                             compatible_bid.quantity, str(datetime.datetime.now())]]

            self.Add_Trade(executed_ask)
            # Update the ask trade
            current_ask.Set_Quantity(current_ask.quantity - compatible_bid.quantity)

            # TODO: Payout/ Debit each trader
            
          elif compatible_bid.quantity - current_ask.quantity > 0:
            # This is the case for a complete match
            # Log the bid and ask on the trade log
            executed_bid = [[compatible_bid.trader_id, compatible_bid.order_id,\
               compatible_bid.order_type, compatible_bid.price,\
               current_ask.quantity, str(datetime.datetime.now())]]
            executed_ask = [[current_ask.trader_id, current_ask.order_id,\
               current_ask.order_type, current_ask.price,\
               current_ask.quantity, str(datetime.datetime.now())]]

            self.Add_Trade(executed_ask)
            self.Add_Trade(executed_bid)

            # Update the bid order
            compatible_bid.Set_Quantity(compatible_bid.quantity - current_ask.quantity)

            # Remove the ask order from the LOB
            marketplace.Remove_Order(current_ask)

            # TODO: Payout / Debit each trader
          
          else:
            # If there is an exact match
            # Log both trades
            executed_bid = [[compatible_bid.trader_id, compatible_bid.order_id,\
               compatible_bid.order_type, compatible_bid.price,\
               current_ask.quantity, str(datetime.datetime.now())]]
            executed_ask = [[current_ask.trader_id, current_ask.order_id,\
               current_ask.order_type, current_ask.price,\
               current_ask.quantity, str(datetime.datetime.now())]]
          
            self.Add_Trade(executed_ask)
            self.Add_Trade(executed_bid)
            # Remove both trades from the LOB
            marketplace.Remove_Order(current_ask)
            marketplace.Remove_Order(current_ask)

            # TODO: Payout / Debit each trader
            
        # Find the closest match -> until the entire order is matched (or there are no more bids)
        
        

      index += 1
    
    elif match == "pro-rata":
      print()

    # Any orders that haven't been matched in over a minute are removed
    for order in bids:

  def Pricing_Policy(self, bid_price, ask_price):
    """
    Determines the final transaction price of matched bid and ask orders, if there is
    not a complete or direct price match. For simplicity sake, this follows industry
    convention of choosing the price halfway between the bid and ask
    """
    # The result is rounded to 2 decimal places so it is in-line with normal pricing
    return round((bid_price + ask_price)/2, 2)

  def Add_Trade(self, trade):
    """
    Given a trade/ order that has been executed/ confirmed,
    it appends the order to the list of all executed orders.

    The trade parameter is a DataFrame friendly array
    """
    # Converts the trade/ order object into a DataFrame friendly array
    # new_trade = [[trade.trader_id, trade.order_id, trade.order_type, trade.price,\
    #               trade.quantity, str(trade.time)]]

    self.trade_log = pd.concat([self.trade_log,\
                               pd.DataFrame(trade, columns=self.cols)],
                               ignore_index=True)

  def Find_Closest_Value(self, bids, ask_price):
    """
    Finds and returns the order_id of the order whose price is the closest
    to the current bid
    """
    smallest_difference = math.inf
    compatible_bid = 0

    # Sequentially moves down the list until the difference starts increasing
    for x in range(0, len(bids)):
      current_bid = bids.iloc(x)
      if abs(current_bid.price - ask_price) < smallest_difference:
        smallest_difference = abs(current_bid.price - ask_price)
        compatible_bid = current_bid
      else:
        break

    return compatible_bid