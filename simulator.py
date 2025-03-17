import time
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from orderbook import Order
from auctioneer import Auctioneer
from trader import Trader
from marketplace import MarketPlace
# import matplotlib as plt


class Simulator:
    """
    Simulator for running marketplace trading scenarios using an iteration-based approach.
    Each simulation consists of a fixed number of discrete time steps.
    """
    def __init__(self, iterations: int, num_traders: int):
        self.iterations = iterations  # number of times to simulate the marketplace
        self.num_traders = num_traders  # number of traders in each simulation
        self.performance_log = {}  # stores results of each simulation
        
    def run_simulations(self, trader_weights, starting_price):
        """
        Run multiple iterations of the marketplace simulation.
        
        Returns:
            Dict: Performance metrics across all simulations
        """
        for x in range(self.iterations):
            self.performance_log[x] = self.simulate_marketplace(trader_weights, starting_price)
            
        return self.generate_performance_log()
        
    def simulate_marketplace(self, trader_weights, starting_price):
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

        traders = []
        trader_id = 0  # Initialize trader ID

        # From the distribution of trader types, assign and store the trader types for this simulation
        trader_types = np.array(np.random.choice(list(trader_weights.keys()),
                                                 self.num_traders,
                                                 p=list(trader_weights.values())))

        # Create the traders from the list of assigned trader types
        for i in range(0, len(trader_types)):
            trader = Trader(trader_id, trader_types[i], "full", 10_000, starting_price)
            traders.append(trader)
            marketplace.register_trader(trader)
            trader_id += 1 # Increment for each new traders

        # Track initial values to calculate performance
        initial_budgets = {trader.trader_id: trader.budget_size for trader in traders}
        initial_stocks = {trader.trader_id: trader.stock for trader in traders}
        
        # Performance tracking
        trades_per_trader = {trader.trader_id: 0 for trader in traders}
        orders_per_trader = {trader.trader_id: 0 for trader in traders}
        last_action_time = {trader.trader_id: 0 for trader in traders}
        # Trader type performance tracking
        trader_type_metrics = {
            "aggressive" : {},
            "passive" : {},
            "momentum" : {},
            "fundamental_up" : {},
            "fundamental_down" : {},
            "random" : {},
        }
        # Initialise the metric dictionary for each trader type
        for type in trader_type_metrics.keys():
            trader_type_metrics[type] = {
                "volume":0,
                "bid":0,
                "ask":0
            }
        
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
                            # Update the traders by per trader
                            trades_per_trader[trader.trader_id] += 1
                            # Update the total volume of trades submitted by this trader's type
                            trader_type_metrics[trader.trader_type]["volume"] += order.quantity
                            # Update the type of the submitted trade for this trader's type
                            trader_type_metrics[trader.trader_type][order.order_type] += 1
                            # Update last action time
                            last_action_time[trader.trader_id] = step
            
            # Execute market clearing after all traders have had a chance to act
            trades_executed = marketplace.match_orders(step)
            # Update each trader's price history once all orders for this step have been submitted
            for trader in traders:
                trader.update_price_history(marketplace, step)
            
            # Don't need to process traders positions as this is done in the auctioneer object
        
        # End of simulation - calculate performance metrics
        performance_metrics = self.calculate_performance_metrics(
            marketplace,
            traders,
            initial_budgets,
            initial_stocks,
            trades_per_trader,
            orders_per_trader,
            starting_price,
            trader_type_metrics
        )
        
        return performance_metrics
    
    def calculate_performance_metrics(self, marketplace, traders, initial_budgets, 
                                     initial_stocks, trades_per_trader, orders_per_trader,
                                     starting_price, trader_type_metrics):
        """
        Calculate performance metrics at the end of a simulation.
        
        Returns:
            Dict: Performance metrics for traders and the market
        """
        performance_metrics = {}
        
        # Get trade log for final calculations
        trade_log_df = marketplace.get_trade_log_df()
        
        # Provides a default value for last price
        last_trade_price = 100

        if not trade_log_df.empty:
            last_trade = trade_log_df.iloc[-1]
            last_trade_price = last_trade['price']
  
        # A dictionary used to store metrics specific to each trader type, PnL, bid/ask ratio
        # Average trade size and number of trades
        trader_type_performance = {
            "aggressive" : {},
            "passive" : {},
            "momentum" : {},
            "fundamental_up" : {},
            "fundamental_down" : {},
            "random" : {}
        }

        # Initialise a dictionary to store the perfomance metrics for each trader type
        for type in trader_type_performance.keys():
            # Validation to ensure that the number of bids/ asks submitted is > 0
            if trader_type_metrics[type]["ask"] == 0:
                trader_type_metrics[type]["ask"] = 1

            trader_type_performance[type] = {"num_traders" : 0,
                                             "PnL" : 0,
                                             "bid/ask_ratio": trader_type_metrics[type]["bid"]\
                                              / trader_type_metrics[type]["ask"],
                                             # Stores the volume of all trades submitted by this trader type
                                             "avg_trade_volume": trader_type_metrics[type]["volume"],
                                             "avg_trades": 0}
        
        # Calculate P&L for each trader
        for trader in traders:
            trader_id = trader.trader_id
            
            # Calculate total value
            initial_value = initial_budgets[trader_id] + initial_stocks[trader_id] * starting_price
            final_value = trader.budget_size + trader.stock * last_trade_price
            
            # Calculate return
            pnl = round(final_value - initial_value, 2) # Rounds the PnL to 2 decimal places
            pnl_percent = (pnl / initial_value) * 100 if initial_value > 0 else 0
            
            # is this valid???
            trader_trades = trade_log_df[trade_log_df['traderID'] == trader_id]
            trade_count = len(trader_trades)
            
            # Increment the number of traders for the given type
            trader_type_performance[trader.trader_type]["num_traders"] += 1
            # Store the PnL for the given trader type
            trader_type_performance[trader.trader_type]["PnL"] += pnl
            # Increment the number of trades submitted by this type
            trader_type_performance[trader.trader_type]["avg_trades"] += trades_per_trader[trader_id]

        # Calculate market-wide metrics
        total_trades = len(trade_log_df)
        actual_trades = total_trades // 2 if total_trades > 0 else 0
        unique_prices = trade_log_df['price'].nunique() if not trade_log_df.empty else 0
        avg_trade_size = trade_log_df['quantity'].mean() if not trade_log_df.empty else 0

        # Track the difference between the final and starting price
        price_change = last_trade_price - starting_price

        # The trade log is not initialised with the starting price
        # If no trades are made, the highest and lowest price are the starting price
        highest_price = max(trade_log_df['price']) if total_trades > 0\
                        else starting_price
        lowest_price = min(trade_log_df['price']) if total_trades > 0\
                        else starting_price
        
        # Compute the averages for certain metrics for each trader type
        for type in trader_type_performance.keys():
            # Validation in the case that there are zero traders
            if trader_type_performance[type]["num_traders"] == 0:
                # To prevent division by zero and ensure the correct figures are stored/ displayed
                trader_type_performance[type]["PnL"] = 0
                trader_type_performance[type]["avg_trade_volume"] = 0
                trader_type_performance[type]["avg_trades"] = 0
            else:
                # Computes the average PnL
                trader_type_performance[type]["PnL"] = round(trader_type_performance[type]["PnL"]\
                                                    / trader_type_performance[type]["num_traders"], 2)
                # Computes the average trade size
                trader_type_performance[type]["avg_trade_volume"] = trader_type_performance[type]["avg_trade_volume"]\
                                                                    / trader_type_performance[type]["avg_trades"]
                # At this point avg_trades stores the total number of trades made by this type
                # Computes the average number of trades
                trader_type_performance[type]["avg_trades"] = trader_type_performance[type]["avg_trades"]\
                                                            / trader_type_performance[type]["num_traders"]

        performance_metrics["market"] = {
            "total_trades": total_trades,
            "unique_prices": unique_prices,
            "avg_trade_size": avg_trade_size,
            "highest_price": highest_price,
            "lowest_price": lowest_price,
            "price_change": price_change
        }

        # Add the trader type metrics to the performance metrics log
        performance_metrics["type"] = trader_type_performance

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
            "simulations": self.performance_log,
        }
        
        # The trader types as column names to be used to store trader type specific metrics
        type_headers = ["aggressive", "passive", "momentum", "fundamental_up", "fundamental_down", "random"]

        # Skips all the calculation in the event that there are zero traders
        # Calculate average metrics across all simulations
        if self.iterations > 0:
            aggregated_metrics["avg_total_trades"] = round(sum(sim["market"]["total_trades"]\
                                                         for sim in self.performance_log.values()) / self.iterations, 4)
            
            aggregated_metrics["avg_unique_prices"] = sum(sim["market"]["unique_prices"]\
                                                          for sim in self.performance_log.values()) / self.iterations
                                   
            aggregated_metrics["avg_trade_size"]  = round(sum(sim["market"]["avg_trade_size"]\
                                                        for sim in self.performance_log.values()) / self.iterations, 4)
            
            # Calculate the average highest price
            aggregated_metrics["avg_highest_price"] = round(sum(sim["market"]["highest_price"]\
                                                          for sim in self.performance_log.values()) / self.iterations, 2)
            # Calculate the average lowest price
            aggregated_metrics["avg_lowest_price"] = round(sum(sim["market"]["lowest_price"]\
                                                         for sim in self.performance_log.values()) / self.iterations, 2)
            # Calculate the average price difference between closing and starting prices
            aggregated_metrics["avg_price_change"] = round(sum(sim["market"]["price_change"]\
                                                         for sim in self.performance_log.values()) / self.iterations, 2)
            
            # Calculate average metrics across specific trader types
            for type in type_headers:
                aggregated_metrics[type] = {
                    "avg_pnl": round(sum(sim["type"][type]["PnL"]\
                                   for sim in self.performance_log.values()) / self.iterations, 2),
                                   # The average PnL for this type across all the run simulations
                    "avg_trade_size": round(sum(sim["type"][type]["avg_trade_volume"]\
                                          for sim in self.performance_log.values()) / self.iterations, 4),
                                          # The average trade size for this type across all the run simulations
                    "avg_num_trades": round(sum(sim["type"][type]["avg_trades"]\
                                          for sim in self.performance_log.values()) / self.iterations, 4),
                                          # The average number of trades for this type across all the run simulations
                    "avg_bid/ask_ratio": round(sum(sim["type"][type]["bid/ask_ratio"]\
                                             for sim in self.performance_log.values()) / self.iterations, 4),
                                             # The average ratio of bid to ask trades ...
                    "avg_traders": sum(sim["type"][type]["num_traders"]\
                                      for sim in self.performance_log.values()) / self.iterations,
                                      # The average number of traders ...
                }
        
        return aggregated_metrics


if __name__ == "__main__":
    num_of_traders = 100
    # For this simulation an equal proportion of traders is picked
    # The weights/ likelihood for a trader to act a given way
    trader_weights = {
        "aggressive": 1,
        "passive": 0,
        "momentum": 0,
        "random": 0,
        "fundamental_up": 0,
        "fundamental_down": 0,
    }
    
    # Create and run a simulation with 10 iterations and 20 traders
    simulator = Simulator(iterations=10, num_traders=num_of_traders)
    results = simulator.run_simulations(trader_weights, 100)

    # Output the results of the simulation
    print("Simulation completed:")
    print(f"Total simulations ran: {results["num_simulations"]}")
    # Outputs the overall market results
    print(f"Average trades executed per simulation: {results["avg_total_trades"]}")
    print(f"Average trade size per simulation: {results["avg_trade_size"]}")
    print(f"Average market high: ${results["avg_highest_price"]}")
    print(f"Average market low: ${results["avg_lowest_price"]}")
    print(f"Average price change over the simulation: ${results["avg_price_change"]}\n")

    # Outputs the trader specific results
    for trade_type in trader_weights.keys():
        print("For the", trade_type, "strategy:")
        print(f"The average PnL per simulation: {results[trade_type]["avg_pnl"]}")
        print(f"The average trade size per simulation: {results[trade_type]["avg_trade_size"]}")
        print(f"The average number of trades per trader per simulation: {results[trade_type]["avg_num_trades"]}")
        print(f"The average ratio of bid to ask trades: {results[trade_type]["avg_bid/ask_ratio"]}")
        print(f"The average number of traders using the strategy: {results[trade_type]["avg_traders"]}\n")