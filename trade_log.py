from collections import deque
from typing import Dict, List, Any, Deque

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
    
    def to_records(self) -> List[Dict]:
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


# Update the TradeLog class to manage unprocessed trades
class TradeLog:
    """
    Efficient trade logging system with support for unprocessed trades.
    """
    def __init__(self, columns):
        self.columns = columns
        self.processed_trades = deque()  # Already processed trades
        self.unprocessed_trades = deque()  # Trades waiting to be processed
    
    def add_trade(self, trade_records: List[List], processed: bool = False):
        """
        Add trade records to the log.
        
        Args:
            trade_records: The trade data
            processed: Whether this trade has already been processed
        """
        for record in trade_records:
            if processed:
                self.processed_trades.append(record)
            else:
                self.unprocessed_trades.append(record)
    
    def get_unprocessed_trades(self) -> List[List]:
        """
        Get all unprocessed trades.
        
        Returns:
            List of unprocessed trade records
        """
        return list(self.unprocessed_trades)
    
    def mark_trades_processed(self):
        """Move all trades from unprocessed to processed queue"""
        while self.unprocessed_trades:
            trade = self.unprocessed_trades.popleft()
            self.processed_trades.append(trade)
    
    def to_dataframe(self):
        """Convert all trades (processed and unprocessed) to DataFrame"""
        import pandas as pd
        all_trades = list(self.processed_trades) + list(self.unprocessed_trades)
        if not all_trades:
            return pd.DataFrame(columns=self.columns)
        return pd.DataFrame(all_trades, columns=self.columns)