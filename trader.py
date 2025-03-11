import random
import datetime
from typing import Optional

# Import the Order class
from orderbook import Order


class Trader:
    """
    Outlines the class for a given trader in a marketplace, who, using a defined
    trading strategy, generates a bid/ask order and submits it to the marketplace.
    Different traders may have access to different trading histories.
    """

    def __init__(self, trader_id, trader_type, memory_type, budget_size):
        self.trader_id = trader_id
        self.trader_type = trader_type
        self.memory_type = memory_type
        self.budget_size = budget_size
        self.profit_loss = 0
        self.stock = 10  # Each trader starts with 10 stock
        self.frequency = random.uniform(1, 5)  # How often the trader acts

        # price history tracking for momentum traders
        self.price_history = []
        self.history_max_length = 10
        self.last_update_step = -1

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
        
        # Add to price history
        self.price_history.append(current_market_price)
        
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
        
        # Generate an order using the trading strategy
        order = self.trading_strategy(current_market_price, step)
        
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
        # check if bid trader has sufficient funds
        if order.order_type == "bid":
            if order.price * order.quantity > self.budget_size:
                return False
            self.budget_size -= order.price * order.quantity

        # check if trader has enough stock
        elif order.order_type == "ask":
            if order.quantity > self.stock:
                return False  

            self.stock -= order.quantity
    
        # Submit the order to the marketplace
        accepted = marketplace.add_order(order)
    
        # If order was rejected, return the reserved resources
        if not accepted:
            if order.order_type == "bid":
                self.budget_size += order.price * order.quantity
            elif order.order_type == "ask":
                self.stock += order.quantity
    
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
            # Aggressive traders place orders with prices further from the current market price
            if random.random() < 0.55:  # 55% chance to place a bid (buy)
                # Only place order if we have budget
                if self.budget_size // current_market_price > 0:
                    qty = random.randint(1, min(10, int(self.budget_size // current_market_price)))
                    
                    # Aggressive buyer is willing to pay up to 5% more
                    price = current_market_price * (1 + random.uniform(0.01, 0.05))
                    price = round(price, 2)
                    
                    # Check if we can afford this
                    if qty * price <= self.budget_size:
                        return Order(
                            trader_id=self.trader_id,
                            order_type="bid",
                            price=price,
                            quantity=qty,
                            time = step
                        )
            else:
                # Place ask (sell) order if we have stock to sell
                if self.stock > 0:
                    qty = random.randint(1, self.stock)
                    
                    # Aggressive seller is willing to accept up to 5% less
                    price = current_market_price * (1 - random.uniform(0.01, 0.05))
                    price = round(price, 2)
                    
                    return Order(
                        trader_id=self.trader_id,
                        order_type="ask",
                        price=price,
                        quantity=qty,
                        time = step
                    )
        
        elif self.trader_type == "passive":
            # Passive traders place orders with better prices but may not get filled
            if random.random() < 0.5:  # 50% chance to place a bid
                if self.budget_size // current_market_price > 0:
                    qty = random.randint(1, min(5, int(self.budget_size // current_market_price)))
                    
                    # Passive buyer wants to pay less than market price
                    price = current_market_price * (1 - random.uniform(0.01, 0.03))
                    price = round(price, 2)
                    
                    if qty * price <= self.budget_size:
                        return Order(
                            trader_id=self.trader_id,
                            order_type="bid",
                            price=price,
                            quantity=qty,
                            time = step
                        )
            else:
                if self.stock > 0:
                    qty = random.randint(1, min(5, self.stock))
                    
                    # Passive seller wants more than market price
                    price = current_market_price * (1 + random.uniform(0.01, 0.03))
                    price = round(price, 2)
                    
                    return Order(
                        trader_id=self.trader_id,
                        order_type="ask",
                        price=price,
                        quantity=qty,
                        time = step
                    )
        
        elif self.trader_type == "momentum":
            # Momentum traders follow the market trend
            # For simplicity, we'll simulate this with random trend
            if len(self.price_history) >= 2:
                short_window = min(5, len(self.price_history))
                short_avg = sum(self.price_history[-short_window:]) / short_window
                market_trend = 1 if short_avg > long_avg else -1
                
                # Calculate trend strength (how much short-term deviates from long-term)
                trend_strength = abs(short_avg - long_avg) / long_avg
                
                # In uptrend, more likely to buy (follow the momentum)
                if market_trend > 0:
                    # Probability increases with trend strength (stronger trend = more confident)
                    buy_probability = 0.6 + (trend_strength * 10) if trend_strength < 0.04 else 0.9
                    
                    if random.random() < buy_probability and self.budget_size // current_market_price > 0:
                        # Buy more aggressively in stronger uptrends
                        qty = random.randint(1, min(8, int(self.budget_size // current_market_price)))
                        
                        # Price premium based on trend strength
                        premium = random.uniform(0.01, 0.02 + (trend_strength * 10))
                        price = current_market_price * (1 + premium)
                        price = round(price, 2)
                        
                        if qty * price <= self.budget_size:
                            return Order(
                                trader_id=self.trader_id,
                                order_type="bid",
                                price=price,
                                quantity=qty,
                                time=step
                            )
                else:
                    # In downtrend, more likely to sell
                    sell_probability = 0.6 + (trend_strength * 10) if trend_strength < 0.04 else 0.9
                    
                    if random.random() < sell_probability and self.stock > 0:
                        # Sell more aggressively in stronger downtrends
                        qty = random.randint(1, self.stock)
                        
                        # Price discount based on trend strength
                        discount = random.uniform(0.01, 0.02 + (trend_strength * 5))
                        price = current_market_price * (1 - discount)
                        price = round(price, 2)
                        
                        return Order(
                            trader_id=self.trader_id,
                            order_type="ask",
                            price=price,
                            quantity=qty,
                            time=step
                        )
            else:
                # Not enough price history, use random behavior initially
                if random.random() < 0.5:  # 50% chance to place a bid
                    if self.budget_size // current_market_price > 0:
                        qty = random.randint(1, min(5, int(self.budget_size // current_market_price)))
                        price = current_market_price * (1 + random.uniform(-0.02, 0.02))
                        price = round(price, 2)
                        
                        if qty * price <= self.budget_size:
                            return Order(
                                trader_id=self.trader_id,
                                order_type="bid",
                                price=price,
                                quantity=qty,
                                time=step
                            )
                else:
                    if self.stock > 0:
                        qty = random.randint(1, min(3, self.stock))
                        price = current_market_price * (1 + random.uniform(-0.02, 0.02))
                        price = round(price, 2)
                        
                        return Order(
                            trader_id=self.trader_id,
                            order_type="ask",
                            price=price,
                            quantity=qty,
                            time=step
                        )
        
        elif self.trader_type == "random":
            # Completely random behavior
            if random.random() < 0.5:  # 50% chance to place a bid
                if self.budget_size // current_market_price > 0:
                    qty = random.randint(1, min(10, int(self.budget_size // current_market_price)))
                    price = current_market_price * (1 + random.uniform(-0.05, 0.05))
                    price = round(price, 2)
                    
                    if qty * price <= self.budget_size:
                        return Order(
                            trader_id=self.trader_id,
                            order_type="bid",
                            price=price,
                            quantity=qty,
                            time = step
                        )
            else:
                if self.stock > 0:
                    qty = random.randint(1, self.stock)
                    price = current_market_price * (1 + random.uniform(-0.05, 0.05))
                    price = round(price, 2)
                    
                    return Order(
                        trader_id=self.trader_id,
                        order_type="ask",
                        price=price,
                        quantity=qty,
                        time = step
                    )
        
        # If no order was generated, return None
        return None
    
    def set_trader_memory(self, trade_log):
        """
        Updates the dataframe containing all the past trades (successful, total or both)
        that the trader can view.
        """
        pass
    
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
    
    def unreserve_order(self, order):
        if order.order_type == "bid":
            # Return the reserved funds
            self.budget_size += order.price * order.quantity
        else:  # "ask"
            # Return the reserved stock
            self.stock += order.quantity