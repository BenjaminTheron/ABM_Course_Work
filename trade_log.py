from collections import deque, defaultdict
from typing import Dict, List, Any, Deque, Optional
import heapq

# First, update the Trade class to include a processed flag
class Trade:
    """Represents a completed trade between two orders"""
    def __init__(self, bid_order, ask_order, price: float, quantity: float, step):
        self.bid_order = bid_order
        self.ask_order = ask_order
        self.price = price
        self.quantity = quantity
        self.time = step
        self.processed = False  # Flag to track if this trade has been processed
    
    def to_records(self):
        """Convert trade to two records (bid and ask) for logging"""
        bid_record = [
            [
                self.bid_order.trader_id, 
                self.bid_order.order_id,
                self.bid_order.order_type,
                self.price,
                self.quantity,
                self.time
            ]
        ]
        
        ask_record = [
            [
                self.ask_order.trader_id,
                self.ask_order.order_id,
                self.ask_order.order_type,
                self.price,
                self.quantity,
                self.time
            ]
        ]
        
        return bid_record + ask_record
    def __lt__(self, other):
        """
        Comparison method needed for heapq operations.
        This is required when two trades have the same price,
        so heapq needs a way to compare them.
        """
        # Compare by time if same price, otherwise by price
        if hasattr(self, 'price') and hasattr(other, 'price'):
            return self.price < other.price
        # Fallback for safety
        return id(self) < id(other)


class TradeLog:
    """
    Simplified trade logging system that organizes trades by step and tracks highest prices.
    """
    def __init__(self, columns):
        self.columns = columns
        # Main data structure: trades organized by step with price-based max heap
        self.trades_by_step = defaultdict(list)
        # A single queue for trades waiting to be processed
        self.unprocessed_trades = deque()
    
    def add_trade(self, trade):
        """
        Add a trade to the log.
        
        Args:
            trade: Either a Trade object or a list of record data
        """
        if isinstance(trade, Trade):
            # Store Trade object directly in trades_by_step
            step = trade.time
            # Use negative price for max heap (since heapq is a min heap)
            heapq.heappush(self.trades_by_step[step], (-trade.price, trade))
            # Also add to unprocessed_trades for processing
            self.unprocessed_trades.append(trade)
        else:
            # Handle raw records
            for record in trade:
                self.unprocessed_trades.append(record)
    
    def get_unprocessed_trades(self):
        """Get all unprocessed trades."""
        return list(self.unprocessed_trades)
    
    def mark_trades_processed(self):
        """Process all unprocessed trades."""
        self.unprocessed_trades.clear()  # Simply clear the queue since trades are already in trades_by_step
    
    def get_highest_price_trade(self, step):
        """
        Get the trade with the highest price for a specific step.
        
        Args:
            step: The step to query
            
        Returns:
            The Trade object with the highest price, or None if no trades exist for that step
        """
        if step in self.trades_by_step and self.trades_by_step[step]:
            # Return just the Trade object (second element in the tuple)
            return self.trades_by_step[step][0][1]
        return None
    
    def to_dataframe(self):
        """
        Convert all trades to DataFrame.
        """
        import pandas as pd
        
        all_records = []
        # Process all stored trades into records
        for step, trades_heap in self.trades_by_step.items():
            for neg_price, trade in trades_heap:
                if isinstance(trade, Trade):
                    all_records.extend(trade.to_records())
        
        # Add any unprocessed trades too
        for item in self.unprocessed_trades:
            if isinstance(item, Trade):
                all_records.extend(item.to_records())
            else:
                # It's already a record
                all_records.append(item)
                
        if not all_records:
            return pd.DataFrame(columns=self.columns)
            
        # Flatten the list of lists if needed
        flattened_records = []
        for record_group in all_records:
            if isinstance(record_group, list) and record_group and isinstance(record_group[0], list):
                flattened_records.extend(record_group)
            else:
                flattened_records.append(record_group)
            
        return pd.DataFrame(flattened_records, columns=self.columns)