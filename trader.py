import random
import datetime
import math
from typing import Optional
import numpy as np

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


class Simple_Trader(Trader):
    """ Outlines the trader implementation for the six simple strategies
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

    
class LFTrader(Trader):
    """
    Low-Frequency Trader implementing fundamentalist and chartist strategies
    as described in the paper.
    """
    def __init__(self, trader_id, memory_type, budget_size, starting_price, parameters=None):
        super().__init__(trader_id, "LF", memory_type, budget_size, starting_price)
        # Store parameters or use defaults
        self.parameters = parameters or {}
        
        # Initialize with a random strategy (fundamentalist or chartist)
        self.strategy = "fundamentalist" if random.random() < 0.5 else "chartist"
        # Trading frequency based on chronological time - drawn from truncated exponential
        self.frequency = self._draw_trading_frequency()
        self.next_trading_time = 0
        self.theoretical_price = 100
        
    
    def _draw_trading_frequency(self):
        """
        Draw trading frequency from truncated exponential distribution
        as specified in the paper.
        """
        # Get parameters from instance or use defaults
        theta = self.parameters.get("theta", 20)      # LF traders' trading frequency mean
        theta_min = self.parameters.get("theta_min", 10)  # Min trading frequency
        theta_max = self.parameters.get("theta_max", 40)  # Max trading frequency
        
        while True:
            freq = np.random.exponential(theta)
            if theta_min <= freq <= theta_max:
                return freq
    
    def generate_order(self, marketplace, step, fundamental_value):
        """
        Generate orders based on current strategy (fundamentalist or chartist).
        
        Args:
            marketplace: The marketplace instance
            step: Current simulation step
            fundamental_value: The current fundamental value
            
        Returns:
            Order: Generated order or None
        """

        last_price = marketplace.get_last_price()
        sigma_z = self.parameters.get("sigma_z", 0.01)  # LF traders' price tick standard deviation
        delta = self.parameters.get("sigma_z", 0.0001)
        self.theoretical_price = last_price * (1 + delta) * (1 + np.random.normal(0, sigma_z))
        
        if step < self.next_trading_time:
            return None
        
        self.next_trading_time = step + self.frequency

        if marketplace.get_history_length() < 2 and self.strategy == "chartist":
            return
        
        quantity = self.calculate_quantity(self.strategy, marketplace.get_price_at(-1), marketplace.get_price_at(-2), fundamental_value)
        
        if quantity > 0: 
                order_type = "bid"
        elif quantity < 0:  
                order_type = "ask"
        else:
            return
        
        quantity = abs(quantity)
            
        order = Order(
            trader_id=self.trader_id,
            order_type=order_type,
            price=self.theoretical_price,
            quantity=quantity,
            time=step
        )
        
        # Print order details
        #print(f"LFT {self.trader_id} creating {order_type} order: {quantity} @ {self.theoretical_price}")
        
        return order
    
    def update_strategy(self, last, second_last, fundamental_value):
        """
        Update strategy based on profitability as described in the paper.
        """
        # Get parameters from instance or use defaults
        zeta = self.parameters.get("zeta", 1)  # Intensity of switching

        c_profit = (last - self.theoretical_price) * self.calculate_quantity("chartist", last, second_last, fundamental_value)
        f_profit = (last - self.theoretical_price) * self.calculate_quantity("fundamentalist", last, second_last, fundamental_value) 
        
        # Calculate switching probability using the logit model from the paper
        prob_chartist = np.exp(zeta * c_profit) / (np.exp(zeta * c_profit) + np.exp(zeta * f_profit))
            
        # Switch strategy based on probability
        self.strategy = "chartist" if random.random() < prob_chartist else "fundamentalist"
                       
    
    def calculate_quantity(self, strategy, last, second_last, fundamental_value):
        # Get parameters from instance or use defaults
        alpha_f = self.parameters.get("alpha_f", 0.04)  # Fundamentalists' order size parameter
        sigma_f = self.parameters.get("sigma_f", 0.01)  # Fundamentalists' shock standard deviation
        alpha_c = self.parameters.get("alpha_c", 0.04)  # Chartists' order size parameter
        sigma_c = self.parameters.get("sigma_c", 0.05)  # Chartists' shock standard deviation
        quantity = 0

        if self.strategy == "fundamentalist":
            # Fundamentalist strategy
            
            # Generate order based on fundamental value
            price_diff = fundamental_value - last
            quantity = alpha_f * price_diff + np.random.normal(0, sigma_f)
            
        else:  # Chartist strategy
            # Calculate trend
            price_trend = last - second_last
            quantity = alpha_c * price_trend + np.random.normal(0, sigma_c)
        
        return quantity

class HFTrader(Trader):
    """
    High-Frequency Trader implementing directional strategies
    as described in the paper.
    """
    def __init__(self, trader_id, memory_type, budget_size, starting_price, parameters=None):
        super().__init__(trader_id, "HF", memory_type, budget_size, starting_price)
        
        # Store parameters or use defaults
        self.parameters = parameters or {}
        
        # Get parameters from instance or use defaults
        eta_min = self.parameters.get("eta_min", 0)    # HF traders' activation threshold min
        eta_max = self.parameters.get("eta_max", 0.2)  # HF traders' activation threshold max
            
        # Price threshold for activation (event-based trading)
        self.price_threshold = random.uniform(eta_min, eta_max)
        self.position_limit = 3000  # Position limit as mentioned in the paper
        self.net_position = 0  # Current net position
        
    def should_activate(self, prev_price, curr_price):
        """
        Determine if HF trader should be activated based on price movement.
        Event-based trading as described in the paper.
        
        The paper specifies that HFTs activate when price changes exceed 
        their individual thresholds (drawn from uniform distribution).
        
        Args:
            prev_price: Previous closing price
            curr_price: Current closing price
            
        Returns:
            bool: Whether the HF trader should be activated
        """
        if prev_price <= 0:
            return False
            
        # Calculate absolute percentage price change
        price_change = abs((curr_price - prev_price) / prev_price)    
        effective_threshold = self.price_threshold # 10x more sensitive
        is_active = price_change > effective_threshold

        return is_active
    
    def generate_order(self, marketplace, step):
        """
        Generate HF orders based on directional strategies that exploit
        order book information from LF traders.
        """
        # Get parameters from instance or use defaults
        lambda_param = self.parameters.get("lambda", 0.625)  # Market volumes weight
        kappa_min = self.parameters.get("kappa_min", 0)      # Order price distribution support min
        kappa_max = self.parameters.get("kappa_max", 0.01)   # Order price distribution support max
        
        order_book = marketplace.order_book
        
        # Equal probability of buy or sell orders
        is_buy = random.random() < 0.5
        
        # HF traders place orders near the best bid/ask
        best_bid = order_book.get_best_bid()
        best_ask = order_book.get_best_ask()

        sell_volume = sum(order.quantity for orders in order_book.asks_by_price.values() for order in orders)
        buy_volume = sum(order.quantity for orders in order_book.bids_by_price.values() for order in orders)

         # Determine whether to buy or sell, ensuring it respects position limits
        if self.net_position < self.position_limit and sell_volume > 0:
            order_type = "bid"
            reference_price = best_ask.price if best_ask else self.find_market_price(marketplace)
            quantity_cap = min(self.position_limit - self.net_position, sell_volume // 4)
        elif self.net_position > -self.position_limit and buy_volume > 0:
            order_type = "ask"
            reference_price = best_bid.price if best_bid else self.find_market_price(marketplace)
            quantity_cap = min(self.position_limit + self.net_position, buy_volume // 4)
        else:
            return None
            
         # Apply small price improvement
        kappa = random.uniform(kappa_min, kappa_max)
        price = reference_price * (1 + kappa) if order_type == "bid" else reference_price * (1 - kappa)
        
        # Set a valid order quantity
        quantity = max(1, min(int(np.random.poisson(lambda_param * 100)), quantity_cap))
        if quantity <= 0:
            return None
        
        # Adjust net position accordingly
        if order_type == "bid":
            self.net_position += quantity
        else:
            self.net_position -= quantity
        
        # Create the order
        order = Order(
            trader_id=self.trader_id,
            order_type=order_type,
            price=price, #max(0.01, round(price, 2)),
            quantity=quantity,
            time=step
        )
        order.agent_type = "HF"
        
        return order