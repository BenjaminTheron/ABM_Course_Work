import datetime
import heapq
from collections import defaultdict, deque
from enum import Enum
from typing import List, Dict, Tuple, Optional, Union, Any
import random



class Order:
  """
  Outlines the class for a given order. 
  """

  def __init__(self, trader_id, order_type, price, quantity, time):
    self.trader_id = trader_id  
    self.order_id = random.randint(0, 10_000_000_000)
    self.order_type = order_type
    self.price = price
    self.quantity = quantity
    self.time = time
    self.original_quantity = quantity  

  def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary format for logging"""
        return {
            'traderID': self.trader_id,
            'orderID': self.order_id,
            'orderType': self.order_type,
            'price': self.price,
            'quantity': self.quantity,
            'time': str(self.time)
        }
  def __lt__(self, other):
        """
        Comparison for heapq - for different ordering of bids and asks
        """
        if self.order_type == "bid":
            # For bids: higher price first, then earlier time
            return (-self.price, self.time) < (-other.price, other.time)
        else:
            # For asks: lower price first, then earlier time
            return (self.price, self.time) < (other.price, other.time)

  def set_time(self, time):
    """
    Updates the time attribute for a given order.
    """
    self.time = time

  def set_quantity(self, quantity):
    """
    Updates the quantity attribute for a given order.
    Used for partially matched orders.
    """
    self.quantity = quantity

class OrderBook:
    """
    Efficient implementation of a limit order book using specialised data structures.
    """
    def __init__(self):
        # Bid and ask orders as heaps (priority queues)
        self.bid_orders = []  # max heap (negative prices)
        self.ask_orders = []  # min heap
        
        # Maps to quickly find orders by ID
        self.orders_by_id = {}  # order_id -> Order
        
        # Maps to quickly find orders at a price level
        self.bids_by_price = defaultdict(list)  # price -> [orders]
        self.asks_by_price = defaultdict(list)  # price -> [orders]
        
        # Keep track of best bid and ask prices
        self.best_bid_price = None
        self.best_ask_price = None
    
    def add_order(self, order: Order) -> None:
        """Add an order to the book"""
        self.orders_by_id[order.order_id] = order
        
        if order.order_type == "bid":
            heapq.heappush(self.bid_orders, order)
            self.bids_by_price[order.price].append(order)
            
            # Update best bid price
            if self.best_bid_price is None or order.price > self.best_bid_price:
                self.best_bid_price = order.price
        else:
            heapq.heappush(self.ask_orders, order)
            self.asks_by_price[order.price].append(order)
            
            # Update best ask price
            if self.best_ask_price is None or order.price < self.best_ask_price:
                self.best_ask_price = order.price
    
    def remove_order(self, order_id: int) -> None:
        """Remove an order from the book"""
        if order_id not in self.orders_by_id:
            return None
        
        order = self.orders_by_id[order_id]
        del self.orders_by_id[order_id]
        
        # Remove from price maps
        if order.order_type == "bid":
            self.bids_by_price[order.price].remove(order)
            if not self.bids_by_price[order.price]:
                del self.bids_by_price[order.price]
                # Update best bid price if needed
                if self.best_bid_price == order.price:
                    self.best_bid_price = max(self.bids_by_price.keys()) if self.bids_by_price else None
        else:
            self.asks_by_price[order.price].remove(order)
            if not self.asks_by_price[order.price]:
                del self.asks_by_price[order.price]
                # Update best ask price if needed
                if self.best_ask_price == order.price:
                    self.best_ask_price = min(self.asks_by_price.keys()) if self.asks_by_price else None
        
        # We don't remove from heaps as that would be O(n)
        # Instead, filter out removed orders during get_best operations
        return order
    
    def get_best_bid(self) -> Optional[Order]:
        """Get the best (highest) bid order"""
        while self.bid_orders:
            order = self.bid_orders[0]  # Peek at top
            if order.order_id not in self.orders_by_id:
                # Order was removed, clean up
                heapq.heappop(self.bid_orders)
                continue
            return order
        return None
    
    def get_best_ask(self) -> Optional[Order]:
        """Get the best (lowest) ask order"""
        while self.ask_orders:
            order = self.ask_orders[0]  # Peek at top
            if order.order_id not in self.orders_by_id:
                # Order was removed, clean up
                heapq.heappop(self.ask_orders)
                continue
            return order
        return None
    
    def get_orders_at_price(self, price: float, order_type: str) -> List[Order]:
        """Get all orders at a specific price level"""
        if order_type == "bid":
            return self.bids_by_price.get(price, []).copy()
        else:
            return self.asks_by_price.get(price, []).copy()
    
    def is_empty(self) -> bool:
        """Check if the order book is empty"""
        return len(self.orders_by_id) == 0
    
    def clean_expired_orders(self, current_time, step, expiry_seconds: int = 60) -> List[Order]:
        """Remove and return all expired orders"""
        current_time = step
        expired_orders = []
        for order_id, order in list(self.orders_by_id.items()):
            if current_time - order.time > expiry_seconds:
                self.remove_order(order_id)
                expired_orders.append(order)
        
        return expired_orders
    def has_crossing_orders(self):
          ''' 
          Checks if there are any crossing orders in the orderbook
          '''
          
          best_bid = self.get_best_bid()
          best_ask = self.get_best_ask()

          if best_bid is None or best_ask is None:
            return False
          
          return best_bid.price >= best_ask.price
    
    def get_bid_ask_spread(self):
        """
        Calculate the current bid-ask spread.
    
        Returns:
            float or None: The spread between best ask and best bid, or None if no valid spread
        """
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
    
        if best_bid and best_ask:
            spread = best_ask.price - best_bid.price
            # Only return positive spreads
            return max(0, spread)
    
        return None

    def get_agent_order_concentration(self, agent_type: str, order_type: str) -> float:
        """
        Calculate the concentration of orders by agent type and order type.
        Returns the proportion of volume for the specified agent and order type.
        """
        agent_volume = 0
        total_agent_volume = 0
        
        # Calculate volumes for the specified agent type
        for price, orders in self.bids_by_price.items():
            for order in orders:
                if order.agent_type == agent_type:
                    total_agent_volume += order.quantity
                    if order.order_type == "bid":
                        agent_volume += order.quantity
        
        for price, orders in self.asks_by_price.items():
            for order in orders:
                if order.agent_type == agent_type:
                    total_agent_volume += order.quantity
                    if order.order_type == "ask":
                        agent_volume += order.quantity
        
        # Calculate concentration
        if total_agent_volume == 0:
            return 0
            
        return agent_volume / total_agent_volume if order_type == "ask" else 1 - (agent_volume / total_agent_volume)


