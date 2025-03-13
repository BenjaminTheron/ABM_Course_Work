import random
import datetime
import math
from typing import Optional

# Import the Order class
from orderbook import Order


class Trader:
    """
    Outlines the class for a given trader in a marketplace, who, using a defined
    trading strategy, generates a bid/ask order and submits it to the marketplace.
    Different traders may have access to different trading histories.
    """

    def __init__(self, trader_id, trader_type, memory_type, budget_size, starting_price):
        self.trader_id = trader_id
        self.trader_type = trader_type
        self.memory_type = memory_type
        self.budget_size = budget_size
        self.profit_loss = 0
        self.stock = 10  # Each trader starts with 10 stock
        self.frequency = random.uniform(1, 5)  # How often the trader acts

        self.starting_price = starting_price # required for fundamentalist traders
        # price history tracking for momentum traders
        self.price_history = []
        self.history_max_length = 25
        self.last_update_step = math.inf

    def update_price_history(self, marketplace, step):
        """
        Updates the price history for momentum trading, should be called every step.
        
        Args:
            marketplace: The marketplace instance
            step: Current simulation step
        """
        # Only update once per step
        if step <= self.last_update_step:
            return
        
        # Add the current market price to the price history
        self.price_history.append(self.find_market_price(self, marketplace))
        
        # Keep history at the max allowed length
        if len(self.price_history) > self.history_max_length:
            self.price_history.pop(0)
            
        # Update last update step
        self.last_update_step = step
        
    def generate_bid_ask(self, marketplace, step):
        """
        Uses the trading strategy to instantiate a valid bid/ask order.
        
        Args:
            marketplace: The marketplace instance
            
        Returns:
            Order: An order object or None if no order is generated
        """        
        # Generate an order using the trading strategy
        order = self.trading_strategy(self.find_market_price(marketplace), step)
        
        return order
    
    def submit_shout(self, order, marketplace):
        """
        Submits a provided order to the marketplace.
        
        Args:
            order: The order to submit
            marketplace: The marketplace instance
            
        Returns:
            bool: Whether the order was accepted
        """
        accepted = False
        # Checks whether the trader has enough stock/ funds for the order
        if (order.order_type == "bid" and self.budget_size >= order.price * order.quantity)\
            or (order.order_type == "ask" and self.stock >= order.quantity):
            # Submits the order to the market place
            accepted = marketplace.add_order(order)

            if accepted and order.order_type == "bid":
                # Subtract funds from trader
                self.budget_size -= order.price * order.quantity
            elif accepted and order.order_type == "ask":
                # Subtract stock from the trader
                self.stock -= order.quantity
    
        return accepted
    
    def trading_strategy(self, current_market_price=100, step=0) -> Optional[Order]:
        """
        This is the trader's trading strategy. Given the budget, trading history, and
        other market parameters, this function generates a trade the trader wishes to
        execute.
        
        Args:
            current_market_price: The current market price estimate
            
        Returns:
            Order: An order object or None if no order should be placed
        """
        # Different strategies based on trader type
        if self.trader_type == "aggressive":
            # Aggressive traders place orders close to the top bid-ask to have their orders instantly filled
            rand_val = random.random()
            price = round(current_market_price * (1 + random.uniform(0.01, 0.05)), 2) if rand_val < 0.55 \
                    else round(current_market_price * (1 - random.uniform(0.01, 0.05)), 2)
            
            return self.create_order(current_market_price, step, price, 10, 0.55, rand_val)
        
        elif self.trader_type == "passive":
            # Passive traders place orders with better prices but may not get filled
            rand_val = random.random()
            price = round(current_market_price * (1 - random.uniform(0.01, 0.03)), 2) if rand_val < 0.5 \
                    else round(current_market_price * (1 + random.uniform(0.01, 0.03)), 2)
            
            return self.create_order(current_market_price, step, price, 5, 0.5, rand_val)
        
        elif self.trader_type == "momentum":
            # This is a price action momentum trader -> Go long when the close is
            # at a 25 day high and go short when the close is at a 25 day low
            # The size of the bids/ asks is weighted by the recent trend strength
            if len(self.price_history) >= 5:
                if current_market_price >= max(self.price_history):
                    # Use the reciprocal of the recent (5 day) trend strength to weight the size of the bid
                    trend_strength = 1 + self.price_history[-5] / self.price_history[-1]
                    # Price the bid lower/ at current price if there's an upward trend
                    price = round(current_market_price * (1 + random.uniform(-0.03, 0)), 2)

                    if self.budget_size // price > 0:
                        qty = int(min(self.budget_size // price,\
                                  round(trend_strength *\
                                        random.randint(1,\
                                                       max(1, round((self.budget_size / price)\
                                                                     / trend_strength))))))
                        # Submit the order
                        return Order(trader_id=self.trader_id,
                                    order_type="bid",
                                    price=price,
                                    quantity=qty,
                                    time=step)

                elif current_market_price <= min(self.price_history):
                    # Use the reciprocal of the recent (5 day) trend strength to weight the size of the bid
                    trend_strength = 1 + self.price_history[-5] / self.price_history[-1]
                    # Price the ask higher/ at current price if there's an upward trend
                    price = round(current_market_price * (1 + random.uniform(0, 0.03)), 2)

                    if self.stock > 0:
                        qty = int(min(self.stock,\
                                  round(trend_strength *\
                                        random.randint(1, max(1, round(self.stock / trend_strength))))))
                        # Submit the order
                        return Order(trader_id=self.trader_id,
                                 order_type="ask",
                                 price=current_market_price,
                                 quantity=qty,
                                 time=step)
                # If not approaching a high or low, hold
            else:
                # If there isn't enough price history, act randomly
                price = round(current_market_price * (1 + random.uniform(-0.02, 0.02)), 2)
                return self.create_order(current_market_price, step, price, 5, 0.5, random.random())

        elif "fundamental" in self.trader_type:
            # Fundamentalists believe the stock price is a certain price based on company fundamentals (in this case nothing)
            value = 20 # The percentage by which the stock is over/under-valued
            fundamental_price = round((1 + (value/100)) * self.starting_price, 2) if self.trader_type == "fundamental_up" \
                                else round((1 - (value/100)) * self.starting_price, 2)
            
            # Submit bids if the stock is trading below the fundamental price, and submit asks if it's trading above
            if fundamental_price > current_market_price:
                if self.budget_size // current_market_price > 0:
                    trade_ratio = fundamental_price / current_market_price
                    qty = int(min(self.budget_size//current_market_price,\
                              round(trade_ratio * random.randint(1, max(1,\
                                                                        round((self.budget_size\
                                                                               / current_market_price)\
                                                                               / trade_ratio))))))
                    # As fundamentalists already believe in over/under-valuation, trade at current market price
                    return Order(trader_id=self.trader_id,
                                 order_type="bid",
                                 price=current_market_price,
                                 quantity=qty,
                                 time=step)

            elif fundamental_price < current_market_price:
                # Submit ask if the trader has sufficient stock
                if self.stock > 0:
                    trade_ratio = current_market_price / fundamental_price
                    qty = int(min(self.stock,\
                              round(trade_ratio *\
                                    random.randint(1, max(1, round(self.stock / trade_ratio))))))
                    # For aforementioned reason, trade at market price
                    return Order(trader_id=self.trader_id,
                                 order_type="ask",
                                 price=current_market_price,
                                 quantity=qty,
                                 time=step)
            # Hold if stock is trading at the believed price
            # The size of the bid/asks are determined by how over/under-valued the stocks are
        elif self.trader_type == "random":
            # Traders with completely random behaviour
            price = round(current_market_price * (1 + random.uniform(-0.05, 0.05)), 2)
            return self.create_order(current_market_price, step, price, 10, 0.5, random.random())

        # If no order was generated, return None
        return None
    
    def delete_trade(self, marketplace, order_id):
        """
        Enables a trader to cancel a trade should they choose.
        
        Args:
            marketplace: The marketplace instance
            order_id: The ID of the order to cancel
            
        Returns:
            bool: Whether the order was successfully canceled
        """
        # Find and remove the order from the order book
        order = marketplace.order_book.get_order(order_id)
        
        if order and order.trader_id == self.trader_id:
            marketplace.order_book.remove_order(order_id)
            
            # If it was a bid, return the reserved funds
            if order.order_type == "bid":
                self.budget_size += order.price * order.quantity
                
            return True
        
        return False

    def create_order(self, current_market_price, step, order_price, min_qty, order_chance, rand_val):
        """
        Creates and returns an order object, given budget/ stock restrictions and provided limits.
        """
        if rand_val < order_chance:
            if self.budget_size // current_market_price > 0:
                qty = random.randint(1, min(min_qty, int(self.budget_size // current_market_price)))

                # If there are sufficient funds in the budget
                if qty * order_price <= self.budget_size:
                    return Order(trader_id=self.trader_id,
                                 order_type="bid",
                                 price=order_price,
                                 quantity=qty,
                                 time=step)
        else:
            # While the trader has stock
            if self.stock > 0:
                return Order(trader_id=self.trader_id,
                             order_type="ask",
                             price=order_price,
                             quantity=random.randint(1, min(min_qty, self.stock)),
                             time=step)
        
        return None
    
    def find_market_price(self, marketplace):
        """
        Finds the current market price of the stock

        Returns:
            Returns the current price of the stock
        """
        # Calculate current market price based on the order book
        current_market_price = 100  # Default value
        
        # Access the order book through the marketplace
        order_book = marketplace.order_book
        
        # If there are orders in the book, calculate a more accurate price
        if not order_book.is_empty():
            best_bid = order_book.get_best_bid()
            best_ask = order_book.get_best_ask()
            
            if best_bid and best_ask:
                # Calculate mid price between best bid and best ask
                current_market_price = (best_bid.price + best_ask.price) / 2
            elif best_bid:
                # Only bids available
                current_market_price = best_bid.price
            elif best_ask:
                # Only asks available
                current_market_price = best_ask.price

        return current_market_price