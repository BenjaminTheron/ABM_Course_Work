import time
import pandas as pd
import random
import pandas as pd
from typing import Dict, List, Any
from orderbook import Order
from auctioneer import Auctioneer
from trader import Trader
from marketplace import MarketPlace


class Simulator:
    """
    Simulator for running marketplace trading scenarios using an iteration-based approach.
    Each simulation consists of a fixed number of discrete time steps.
    """
    def __init__(self, iterations: int, num_traders: int):
        self.iterations = iterations  # number of times to simulate the marketplace
        self.num_traders = num_traders  # number of traders in each simulation
        self.performance_log = {}  # stores results of each simulation
        
    def run_simulations(self):
        """
        Run multiple iterations of the marketplace simulation.
        
        Returns:
            Dict: Performance metrics across all simulations
        """
        for x in range(self.iterations):
            self.performance_log[x] = self.simulate_marketplace()
            
        return self.generate_performance_log()
        
    def simulate_marketplace(self):
        """
        Runs a single simulation of the marketplace with traders and auctioneer.
        Uses a fixed number of discrete time steps.
        
        Returns:
            Dict: Performance metrics for this simulation
        """
        # Initialize the auctioneer
        auctioneer = Auctioneer(
            auctioneer_id=0,
            reg_fees=0, 
            inf_fees=0, 
            transac_fees=0, 
            shout_fees=0, 
            profit_fees=0, 
            min_trade_size=1
        )
        
        # Initialize the marketplace with our auctioneer
        marketplace = MarketPlace(auctioneer)
        
        # Likelihood a trader acts a given way
        trader_types_weights = {
            "aggressive": 20,
            "passive": 20,
            "momentum": 20,
            "random": 20,
            "fundamental_up": 20,
            "fundamental_down": 20,
        }

        traders = []
        trader_id = 0  # Initialize trader ID

        # Create traders based on the specified weights
        for trader_type, count in trader_types_weights.items():
            for _ in range(count):
                trader = Trader(trader_id, trader_type, "full", 10000, 100)
                traders.append(trader)
                marketplace.register_trader(trader)
                trader_id += 1  # Increment ID for each new trader

        # Track initial values to calculate performance
        initial_budgets = {trader.trader_id: trader.budget_size for trader in traders}
        initial_stocks = {trader.trader_id: trader.stock for trader in traders}
        
        # Performance tracking
        trades_per_trader = {trader.trader_id: 0 for trader in traders}
        orders_per_trader = {trader.trader_id: 0 for trader in traders}
        last_action_time = {trader.trader_id: 0 for trader in traders}
        
        # Fixed number of simulation steps (one trading day)
        simulation_steps = 240
        
        # Main simulation loop - iterate through discrete time steps
        for step in range(simulation_steps):
            # Give each trader a chance to act in this time step
            for trader in traders:
                # Check if it's time for this trader to act based on frequency
                time_since_last_action = step - last_action_time.get(trader.trader_id, 0)
                
                if time_since_last_action >= trader.frequency:
                    # Generate an order
                    order = trader.generate_bid_ask(marketplace, step)
                    
                    if order:
                        # Track order generation
                        orders_per_trader[trader.trader_id] += 1
                        
                        # Submit the order
                        accepted = trader.submit_shout(order, marketplace)
                        
                        if accepted:
                            # Update trader's budget if a bid was placed
                            #if order.order_type == "bid":
                            #    trader.budget_size -= order.price * order.quantity
                            
                            # Update last action time
                            last_action_time[trader.trader_id] = step
            
            # Execute market clearing after all traders have had a chance to act
            trades_executed = marketplace.match_orders(step)
            
            # Don't need to process traders positions as this is done in the auctioneer object
        
        # End of simulation - calculate performance metrics
        performance_metrics = self.calculate_performance_metrics(
            marketplace,
            traders,
            initial_budgets,
            initial_stocks,
            trades_per_trader,
            orders_per_trader
        )
        
        return performance_metrics
    
    def calculate_performance_metrics(self, marketplace, traders, initial_budgets, 
                                     initial_stocks, trades_per_trader, orders_per_trader):
        """
        Calculate performance metrics at the end of a simulation.
        
        Returns:
            Dict: Performance metrics for traders and the market
        """
        performance_metrics = {}
        
        # Get trade log for final calculations
        trade_log_df = marketplace.get_trade_log_df()
        
        # Determine last price for valuation
        '''
        last_trade_price = 100  # Default
        if len(trade_log_df) > 1:
            last_trade = trade_log_df.iloc[-1]
            if last_trade['traderID'] != -1:
                last_trade_price = last_trade['price']
        '''

        if not trade_log_df.empty:
            last_trade = trade_log_df.iloc[-1]
            last_trade_price = last_trade['price']
        
        # Calculate P&L for each trader
        for trader in traders:
            trader_id = trader.trader_id
            
            # Calculate total value
            initial_value = initial_budgets[trader_id] + initial_stocks[trader_id] * last_trade_price
            final_value = trader.budget_size + trader.stock * last_trade_price
            
            # Calculate return
            pnl = final_value - initial_value
            pnl_percent = (pnl / initial_value) * 100 if initial_value > 0 else 0
            
            # is this valid?
            trader_trades = trade_log_df[trade_log_df['traderID'] == trader_id]
            trade_count = len(trader_trades)
            
            # Store metrics
            performance_metrics[f"trader_{trader_id}"] = {
                "type": trader.trader_type,
                "initial_budget": initial_budgets[trader_id],
                "final_budget": trader.budget_size,
                "initial_stock": initial_stocks[trader_id],
                "final_stock": trader.stock,
                "pnl": pnl,
                "pnl_percent": pnl_percent,
                "orders_submitted": orders_per_trader[trader_id],
                "trades_executed": trades_per_trader[trader_id]
            }
        
        # Calculate market-wide metrics
        total_trades = len(trade_log_df) 
        actual_trades = total_trades // 2 if total_trades > 0 else 0
        unique_prices = trade_log_df['price'].nunique() if not trade_log_df.empty else 0
        avg_trade_size = trade_log_df['quantity'].mean() if not trade_log_df.empty else 0

        performance_metrics["market"] = {
            "total_trades": total_trades,
            "unique_prices": unique_prices,
            "avg_trade_size": avg_trade_size
        }
        
        return performance_metrics
    
    def generate_performance_log(self):
        """
        Aggregates performance metrics across simulations.
        
        Returns:
            Dict: Aggregated performance metrics
        """
        # Simple aggregation across simulations
        aggregated_metrics = {
            "num_simulations": self.iterations,
            "simulations": self.performance_log
        }
        
        # Calculate average metrics across all simulations
        if self.iterations > 0:
            avg_total_trades = sum(sim["market"]["total_trades"] 
                                  for sim in self.performance_log.values()) / self.iterations
            
            avg_unique_prices = sum(sim["market"]["unique_prices"] 
                                   for sim in self.performance_log.values()) / self.iterations
                                   
            avg_trade_size = sum(sim["market"]["avg_trade_size"] 
                                   for sim in self.performance_log.values()) / self.iterations

            aggregated_metrics["avg_total_trades"] = avg_total_trades
            aggregated_metrics["avg_unique_prices"] = avg_unique_prices
            aggregated_metrics["avg_trade_size"] = avg_trade_size
        
        return aggregated_metrics


if __name__ == "__main__":
    # Create and run a simulation with 5 iterations and 10 traders
    simulator = Simulator(iterations=5, num_traders=10)
    results = simulator.run_simulations()
    
    # Print results
    print("Simulation completed:")
    print(f"Total simulations: {results['num_simulations']}")
    print(f"Average trades per simulation: {results['avg_total_trades']}")
    print(f"Average trade size: {results["avg_trade_size"]}")