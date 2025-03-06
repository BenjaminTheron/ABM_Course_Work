# sends messages to the marketplace

# TODO: What is the effect of different trading strategies?

# Basic assumption that

# TODO -> Objective function for the trader, what are they trying to maximise or minimise (budget, p&l, etc.)

# TODO: What is the effect of traders having different memories
# (i.e. simulate a retail and institutional investor)
class Trader:
  """
  Outlines the class for a given trader in a marketplace, who, using a defined
  trading strategy, generates a bid/ ask order and submits it to the marketplace.
  Different traders may have access to different trading histories.
  """

  def __init__(self, trader_id, trader_type, memory_type, budget_size):
    self.trader_id = trader_id
    self.trader_type = trader_type # what does this do?
    self.memory_type = memory_type
    self.budget_size = budget_size
    self.profit_loss = 0
    self.stock = 10 # Each trader starts with 10 stock?

    # Start each trader off with x amount of stock?

  def Generate_BidAsk(self, marketplace):
    """
    Uses the trading strategy outlined in the Trading_Strategy() function to
    instantiate a valid bid/ask order and submit it to the marketplace.
    """
    # Use a basic momentum trading strategy to generate orders

    # Only generate trades that are within the budget of the trader

  def Submit_Shout(self, order, auctioneer, marketplace):
    """
    Submits a provided order to the marketplace.
    """
    accepted = auctioneer.Shout_Accepting_Policy(order, marketplace)

    if accepted:
      # Order has been submitted to the LOB
      return True
    else:
      # Order has been rejected and has not been submitted to the LOB
      return False
      # Generate another order? -> change something here?

  def Trading_Strategy(self):
    """
    This is the trader's trading strategy. Given the budget, trading history, and
    other market parameters, this function generates a trade the trader wishes to
    execute.
    """
    print()

  def Set_Trader_Memory(self, trade_log):
    """
    Updates the dataframe containing all the past trades (successful, total or both)
    that the trader can view
    """

  def Delete_Trade(self, auctioneer, marketplace):
    """
    Enables a trader to cancel a trade should they choose (and should it actually be on
    the marketplace).
    """
