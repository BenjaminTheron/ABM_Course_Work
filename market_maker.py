import random
import numpy as np
from orderbook import Order
from trader import Trader

class MarketMaker(Trader):
    """
    Market Maker that provides liquidity by continuously posting bid and ask orders.
    Inherits from the base Trader class.
    """
    def __init__(self, trader_id, initial_budget=100000, parameters=None):
        """
        Initialize a Market Maker with genetic parameters
        
        Args:
            trader_id: Unique ID for the market maker
            initial_budget: Starting capital
            parameters: Dictionary of genetic and simulation parameters
        """
        # Call parent constructor
        super().__init__(
            trader_id=trader_id,
            trader_type="MM",  # Market Maker type
            memory_type="full",
            budget_size=initial_budget,
            starting_price=100
        )
        
        self.parameters = parameters or {}
        self.active_orders = []  # Track active orders
        
        # Genetic parameters (with defaults)
        self.bid_spread_factor = parameters.get("bid_spread_factor", 0.001)  # Bid price adjustment
        self.ask_spread_factor = parameters.get("ask_spread_factor", 0.001)  # Ask price adjustment
        self.max_inventory_limit = parameters.get("max_inventory_limit", 500)  # Max inventory
        self.hedge_ratio = parameters.get("hedge_ratio", 0.3)  # Fraction to hedge
        self.order_size_multiplier = parameters.get("order_size_multiplier", 0.1)  # Order size
        self.skew_factor = parameters.get("skew_factor", 0.0)  # Inventory-based price skew
        
        # Performance tracking
        self.initial_budget = initial_budget
        self.initial_stock = self.stock
        self.trades_executed = 0
        self.total_volume = 0
        self.fitness = 0
        
        # Risk management
        self.inventory = self.stock
        self.max_position_seen = 0
        self.inventory_history = []
        self.pnl_history = []
        self.current_mid_price = 100  # Starting mid price
    
    def calculate_current_value(self, mid_price):
        """Calculate current total value of the market maker"""
        return self.budget_size + (self.stock * mid_price)
    
    def generate_order(self, marketplace, step, fundamental_value=None):
        """
        Generate market making orders based on current market conditions.
        
        Args:
            marketplace: The marketplace instance
            step: Current simulation step
            fundamental_value: Unused but included for compatibility with Trader interface
            
        Returns:
            List[Order]: List of generated orders
        """
        # Cancel previous orders first
        self.cancel_active_orders(marketplace)
        
        # Get market data
        order_book = marketplace.order_book
        best_bid = order_book.get_best_bid()
        best_ask = order_book.get_best_ask()
        
        # Calculate mid price
        if best_bid and best_ask:
            mid_price = (best_bid.price + best_ask.price) / 2
        else:
            # Fallback to last price if no bids or asks
            mid_price = marketplace.get_last_price()
        
        self.current_mid_price = mid_price
        
        # Track current inventory and position
        self.inventory = self.stock
        self.max_position_seen = max(self.max_position_seen, abs(self.inventory))
        self.inventory_history.append(self.inventory)
        
        # Calculate current PnL for tracking
        current_value = self.calculate_current_value(mid_price)
        initial_value = self.initial_budget + (self.initial_stock * mid_price)
        current_pnl = current_value - initial_value
        self.pnl_history.append(current_pnl)
        
        # Apply inventory skew to spreads based on current position
        inventory_skew = (self.inventory / self.max_inventory_limit) * self.skew_factor
        
        # Calculate bid and ask prices with inventory skew
        bid_spread = self.bid_spread_factor + max(0, inventory_skew)  # Higher inventory -> wider bid spread
        ask_spread = self.ask_spread_factor + max(0, -inventory_skew)  # Lower inventory -> wider ask spread
        
        bid_price = mid_price * (1 - bid_spread)
        ask_price = mid_price * (1 + ask_spread)
        
        # Determine order size based on genetic parameter
        base_order_size = max(1, int(self.max_inventory_limit * self.order_size_multiplier))
        
        # Create orders
        orders = []
        
        # Only submit bid if we're not over inventory limit
        if self.inventory < self.max_inventory_limit:
            bid_order = Order(
                trader_id=self.trader_id,
                order_type="bid",
                price=bid_price,
                quantity=base_order_size,
                time=step
            )
            bid_order.agent_type = "MM"
            orders.append(bid_order)
        
        # Only submit ask if we have inventory to sell
        if self.inventory > -self.max_inventory_limit:
            ask_order = Order(
                trader_id=self.trader_id,
                order_type="ask",
                price=ask_price,
                quantity=base_order_size,
                time=step
            )
            ask_order.agent_type = "MM"
            orders.append(ask_order)
        
        # Perform hedging if inventory exceeds limit
        if abs(self.inventory) > self.max_inventory_limit:
            hedge_size = int(self.hedge_ratio * abs(self.inventory))
            
            if hedge_size > 0:
                hedge_order = Order(
                    trader_id=self.trader_id,
                    order_type="ask" if self.inventory > 0 else "bid",
                    price=mid_price,
                    quantity=hedge_size,
                    time=step
                )
                hedge_order.agent_type = "MM"
                orders.append(hedge_order)
        
        # Track orders for later cancellation
        self.active_orders = [order.order_id for order in orders]
        
        return orders
    
    def submit_orders(self, orders, marketplace):
        """Submit multiple orders to the marketplace"""
        submitted_orders = []
        
        for order in orders:
            # Submit the order using the parent class method
            accepted = self.submit_shout(order, marketplace, solvency=True)
            
            if accepted:
                submitted_orders.append(order)
        
        return submitted_orders
    
    def cancel_active_orders(self, marketplace):
        """Cancel all active orders"""
        for order_id in self.active_orders:
            # Don't use delete_trade method since it uses get_order
            # Instead access orders_by_id directly
            order = marketplace.order_book.orders_by_id.get(order_id)
            
            if order and order.trader_id == self.trader_id:
                # Remove the order from the book
                removed_order = marketplace.order_book.remove_order(order_id)
                
                if removed_order:
                    # Return reserved funds or stock
                    if removed_order.order_type == "bid":
                        self.budget_size += removed_order.price * removed_order.quantity
                    else:  # ask
                        self.stock += removed_order.quantity
        
        # Clear active orders list
        self.active_orders = []
    
    def calculate_fitness(self, final_mid_price):
        """
        Calculate the fitness of this market maker.
        Higher is better.
        """
        # Calculate final PnL
        final_value = self.budget_size + (self.stock * final_mid_price)
        initial_value = self.initial_budget + (self.initial_stock * final_mid_price)
        pnl = final_value - initial_value
        
        # Calculate fitness components
        pnl_component = pnl
        risk_component = -self.max_position_seen * 0.01  # Penalize high positions
        volume_component = self.total_volume * 0.01  # Reward high volume
        
        # Calculate final fitness score
        self.fitness = pnl_component + risk_component + volume_component
        
        return self.fitness
    
    def get_genome(self):
        """Return the current genetic parameters"""
        return {
            "bid_spread_factor": self.bid_spread_factor,
            "ask_spread_factor": self.ask_spread_factor,
            "max_inventory_limit": self.max_inventory_limit,
            "hedge_ratio": self.hedge_ratio,
            "order_size_multiplier": self.order_size_multiplier,
            "skew_factor": self.skew_factor
        }
    
    def set_genome(self, genome):
        """Set genetic parameters from a genome dictionary"""
        self.bid_spread_factor = genome.get("bid_spread_factor", self.bid_spread_factor)
        self.ask_spread_factor = genome.get("ask_spread_factor", self.ask_spread_factor)
        self.max_inventory_limit = genome.get("max_inventory_limit", self.max_inventory_limit)
        self.hedge_ratio = genome.get("hedge_ratio", self.hedge_ratio)
        self.order_size_multiplier = genome.get("order_size_multiplier", self.order_size_multiplier)
        self.skew_factor = genome.get("skew_factor", self.skew_factor)