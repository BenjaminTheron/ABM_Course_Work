# simulator.py

import time
import pandas as pd
import random
import numpy as np
from typing import Dict, List, Any
from orderbook import Order
from auctioneer import Auctioneer
from trader import LFTrader, HFTrader
from marketplace import MarketPlace

# Import market maker if available
try:
    from market_maker import MarketMaker
except ImportError:
    MarketMaker = None

class Simulator:
    """
    Simulator for running marketplace trading scenarios using an iteration-based approach.
    Each simulation consists of a fixed number of discrete time steps.
    """
    def __init__(self, iterations: int, num_lf_traders: int, num_hf_traders: int, parameters=None,
                 use_market_maker=False, market_maker_params=None):
        """
        Initialize the simulator
        
        Args:
            iterations: Number of simulation iterations
            num_lf_traders: Number of low-frequency traders
            num_hf_traders: Number of high-frequency traders
            parameters: Simulation parameters
            use_market_maker: Whether to include a market maker
            market_maker_params: Parameters for the market maker
        """
        self.iterations = iterations  # number of times to simulate the marketplace
        self.num_lf_traders = num_lf_traders
        self.num_hf_traders = num_hf_traders
        self.use_market_maker = use_market_maker
        self.market_maker_params = market_maker_params
        self.performance_log = {}  # stores results of each simulation
        self.market_maker = None  # Reference to market maker if used
        
        # Store parameters
        if parameters is None:
            # Import default parameters if none provided
            from parameters import PARAMETERS
            self.parameters = PARAMETERS.copy()
        else:
            self.parameters = parameters
        
    def run_simulations(self):
        """
        Run multiple iterations of the marketplace simulation.
        
        Returns:
            Dict: Performance metrics across all simulations
        """
        for x in range(self.iterations):
            print(f"Running simulation {x+1}/{self.iterations}")
            self.performance_log[x] = self.simulate_marketplace()

            num_flash_crashes = len(self.performance_log[x].get("flash_crashes", []))
            print(f"Number of flash crashes: {num_flash_crashes}")
            
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
            min_trade_size=0
        )
        
        # Initialize the marketplace with our auctioneer
        marketplace = MarketPlace(auctioneer)
        
        # Create LF traders
        lf_traders = []
        for i in range(self.num_lf_traders):
            trader = LFTrader(
                trader_id=i,
                memory_type="full",
                budget_size=10000,
                starting_price=100,
                parameters=self.parameters
            )
            lf_traders.append(trader)
            marketplace.register_trader(trader)
            
        # Create HF traders
        hf_traders = []
        for i in range(self.num_hf_traders):
            trader = HFTrader(
                trader_id=self.num_lf_traders + i,
                memory_type="full",
                budget_size=10000,
                starting_price=100,
                parameters=self.parameters
            )
            hf_traders.append(trader)
            marketplace.register_trader(trader)

        # Create market maker if requested
        if self.use_market_maker and MarketMaker is not None:
            # Create with trader ID after all other traders
            mm_id = self.num_lf_traders + self.num_hf_traders
            self.market_maker = MarketMaker(
                trader_id=mm_id,
                initial_budget=100000,
                parameters=self.market_maker_params
            )
            marketplace.register_trader(self.market_maker)
            print("Market Maker activated for this simulation")

        # Store initial state for performance tracking
        all_traders = lf_traders + hf_traders
        if self.market_maker:
            all_traders.append(self.market_maker)
            
        initial_budgets = {t.trader_id: t.budget_size for t in all_traders}
        initial_stocks = {t.trader_id: t.stock for t in all_traders}

        # Metrics tracking
        trade_metrics = {
            "price_series": [],
            "bid_ask_spreads": [],
            "flash_crashes": [],
            "recovery_times": [],
            "hft_sell_concentration": [],
            "lft_buy_concentration": [],
            "mm_inventory": [] if self.market_maker else None,
            "mm_orders": [] if self.market_maker else None,
        }
        
        # Get simulation steps from parameters
        simulation_steps = self.parameters.get("T", 1200)
        
        # Initialize activation counters
        hft_activation_count = 0
        
        # Main simulation loop - iterate through discrete time steps
        for step in range(simulation_steps):
            # Update fundamental value
            marketplace.update_fundamental_value(
                delta=self.parameters.get("delta", 0.0001),
                sigma_y=self.parameters.get("sigma_y", 0.01)
            )
            
            fundamental_value = marketplace.get_fundamental_value()
            
            # Market maker acts first to provide liquidity
            if self.market_maker:
                mm_orders = self.market_maker.generate_order(marketplace, step)
                if mm_orders:
                    # Market maker may generate multiple orders
                    for order in mm_orders:
                        self.market_maker.submit_shout(order, marketplace, solvency=True)
                    
                    # Track market maker activity
                    trade_metrics["mm_orders"].append(len(mm_orders))
                    trade_metrics["mm_inventory"].append(self.market_maker.inventory)
            
            # LF traders act
            for trader in lf_traders:
                order = trader.generate_order(marketplace, step, fundamental_value)
                if order:
                    # Set agent type for order concentration analysis
                    order.agent_type = "LF"
                    trader.submit_shout(order, marketplace)
            
            # Count active HF traders for this step
            active_hft_count = 0
            
            # HF traders act
            for trader in hf_traders:
                # Check if HF trader should be activated based on price changes
                if len(marketplace.closing_price_history) >= 2:
                    prev_price = marketplace.closing_price_history[-2]
                    curr_price = marketplace.closing_price_history[-1]

                    if trader.should_activate(prev_price, curr_price):
                        active_hft_count += 1
                        order = trader.generate_order(marketplace, step)
                        if order:
                            order.agent_type = "HF"
                            trader.submit_shout(order, marketplace)
            
            if active_hft_count > 0:
                hft_activation_count += 1
                #print(f"Step {step}: {active_hft_count} HF traders activated")

            marketplace.calculate_concentration_metrics(step)

            # Execute market clearing
            trades_executed = marketplace.match_orders(step)
            
            # Update LF trader strategies
            for trader in lf_traders:
                trader.update_strategy(marketplace.get_last_price(), marketplace.get_price_at(-2), fundamental_value)

            # Clean expired orders - critical for HFT order cancellation behavior
            lf_expired = self.clean_expired_orders(marketplace.order_book, step, "LF", self.parameters.get("gamma_L", 20))
            hf_expired = self.clean_expired_orders(marketplace.order_book, step, "HF", self.parameters.get("gamma_H", 1))
            
            trade_metrics["bid_ask_spreads"] = marketplace.get_spread_history()
            trade_metrics["price_series"] = marketplace.get_price_history()
            
            # Detect flash crashes
            self.detect_flash_crashes(trade_metrics, step, marketplace)
        
        trade_metrics["hft_sell_concentration"] = marketplace.get_hft_sell_concentration_history()
        trade_metrics["lft_buy_concentration"] = [1 - c for c in marketplace.get_lft_sell_concentration_history()]
        trade_metrics["bid_ask_spreads"] = marketplace.get_spread_history()
        
        # Print summary statistics
        print(f"HFT activation frequency: {hft_activation_count/simulation_steps:.2%} of periods")
        
        # Calculate market maker fitness if present
        if self.market_maker:
            final_price = trade_metrics["price_series"][-1] if trade_metrics["price_series"] else 100
            self.market_maker.calculate_fitness(final_price)
            print(f"Market Maker fitness: {self.market_maker.fitness:.2f}")
        
        # Calculate final performance metrics
        performance_metrics = self.calculate_performance_metrics(
            marketplace,
            all_traders,
            initial_budgets,
            initial_stocks,
            trade_metrics
        )
        
        return performance_metrics
        
    def clean_expired_orders(self, order_book, current_step, trader_type, expiry_periods):
        """
        Remove expired orders from the book based on trader type
    
        Args:
            order_book: The order book to clean
            current_step: The current simulation step
            trader_type: Type of trader ("HF" or "LF")
            expiry_periods: Number of periods until expiration
        """
        expired_orders = []
        expired_count = 0
    
        # Process all orders in the book
        for order_id, order in list(order_book.orders_by_id.items()):
            # Check if the order has the agent_type attribute and matches the requested type
            if hasattr(order, 'agent_type') and order.agent_type == trader_type:
                # Check if the order has expired
                if current_step - order.time >= expiry_periods:
                    # Remove order from the book
                    removed_order = order_book.remove_order(order_id)
                    if removed_order:
                        expired_orders.append(removed_order)
                        expired_count += 1
    
        # Print debug info for HFT order cancellations
        #if trader_type == "HF" and expired_count > 0:
        #    print(f"Step {current_step}: Cancelled {expired_count} HFT orders after {expiry_periods} periods")
        
        return expired_orders
                
    def detect_flash_crashes(self, metrics, step, marketplace):
        """
        Detect flash crashes based on the paper's definition:
        - Drop of at least 5% followed by a recovery within 30 periods
        """
        price_series = metrics["price_series"]

        # Need at least 2 prices to detect a drop
        if len(price_series) < 2:
            return
        
        current_price = price_series[-1]

        # Use a 5-period lookback for detecting recent highs
        lookback = min(5, len(price_series) - 1)
        recent_high = max(price_series[-lookback-1:-1]) if lookback > 0 else price_series[-1]

        # Calculate percentage drop
        if recent_high > 0:  # Avoid division by zero
            percentage_drop = (recent_high - current_price) / recent_high
        
            # Debug print for significant drops
            #if percentage_drop > 0.03:
            #    print(f"Price drop detected at step {step}: {percentage_drop:.2%} (from {recent_high:.2f} to {current_price:.2f})")
        
            # Check for flash crash conditions
            if percentage_drop >= 0.05:
                # This could be a flash crash - make sure it's new
                if not metrics["flash_crashes"] or step - metrics["flash_crashes"][-1]["start"] > 30:
                    print(f"FLASH CRASH DETECTED at step {step}: {percentage_drop:.2%} drop")
                    
                    # Get highest price trade for this step for more accurate analysis
                    highest_price_trade = marketplace.auctioneer.trade_log.get_highest_price_trade(step)
                    highest_price = highest_price_trade.price if highest_price_trade else current_price
                    
                    # New crash (not within recovery period of previous crash)
                    metrics["flash_crashes"].append({
                        "start": step,
                        "low_price": current_price,
                        "pre_crash_price": recent_high,
                        "highest_trade_price": highest_price,
                        "recovery_step": None
                    })
    
        # Check for recovery of ongoing flash crashes
        for crash in metrics["flash_crashes"]:
            if crash["recovery_step"] is None:
                # Recovery defined as price returning to within 1% of pre-crash level
                recovery_threshold = 0.99 * crash["pre_crash_price"]
                if current_price >= recovery_threshold:
                    crash["recovery_step"] = step
                    crash["duration"] = step - crash["start"]
                    metrics["recovery_times"].append(crash["duration"])
                    print(f"FLASH CRASH RECOVERY at step {step}: Duration {crash['duration']} periods")
    
    def calculate_performance_metrics(self, marketplace, traders, initial_budgets, 
                                      initial_stocks, trade_metrics):
        """Calculate final performance metrics"""
        # Get trade log
        trade_log_df = marketplace.get_trade_log_df()
        
        # Last price for valuation
        last_price = trade_metrics["price_series"][-1] if trade_metrics["price_series"] else 100
        
        # Calculate volatility
        price_series = np.array(trade_metrics["price_series"])
        if len(price_series) > 1:
            returns = np.diff(price_series) / price_series[:-1]
            volatility = np.std(returns)
        else:
            volatility = 0
        
        # Count and analyze flash crashes
        num_flash_crashes = len(trade_metrics["flash_crashes"])
        avg_crash_duration = np.mean(trade_metrics["recovery_times"]) if trade_metrics["recovery_times"] else 0
        
        # Calculate average bid-ask spread
        avg_spread = np.mean(trade_metrics["bid_ask_spreads"])
        
        # Calculate P&L for each trader
        trader_metrics = {}
        for trader in traders:
            trader_id = trader.trader_id
            
            # Calculate total value
            initial_value = initial_budgets[trader_id] + initial_stocks[trader_id] * last_price
            final_value = trader.budget_size + trader.stock * last_price
            
            # Calculate return
            pnl = final_value - initial_value
            pnl_percent = (pnl / initial_value) * 100 if initial_value > 0 else 0
            
            # Store metrics
            trader_metrics[f"trader_{trader_id}"] = {
                "type": trader.trader_type,
                "initial_budget": initial_budgets[trader_id],
                "final_budget": trader.budget_size,
                "initial_stock": initial_stocks[trader_id],
                "final_stock": trader.stock,
                "pnl": pnl,
                "pnl_percent": pnl_percent
            }
        
        # Aggregate metrics
        performance_metrics = {
            "volatility": volatility,
            "num_flash_crashes": num_flash_crashes,
            "avg_crash_duration": avg_crash_duration,
            "avg_bid_ask_spread": avg_spread,
            "traders": trader_metrics,
            "price_series": trade_metrics["price_series"],
            "flash_crashes": trade_metrics["flash_crashes"],
            "bid_ask_spreads": trade_metrics["bid_ask_spreads"],
            "hft_sell_concentration": trade_metrics["hft_sell_concentration"],
            "lft_buy_concentration": trade_metrics["lft_buy_concentration"]
        }
        
        return performance_metrics
    
    def generate_performance_log(self):
        """Aggregate performance metrics across simulations"""
        aggregated_metrics = {
            "num_simulations": self.iterations,
            "simulations": self.performance_log
        }
        
        # Calculate average metrics
        if self.iterations > 0:
            total_volatility = sum(sim["volatility"] for sim in self.performance_log.values())
            total_flash_crashes = sum(sim["num_flash_crashes"] for sim in self.performance_log.values())
            
            # Calculate average flash crash duration across simulations with crashes
            crash_durations = []
            for sim in self.performance_log.values():
                if sim["num_flash_crashes"] > 0 and sim["avg_crash_duration"] > 0:
                    crash_durations.append(sim["avg_crash_duration"])
            
            avg_crash_duration = np.mean(crash_durations) if crash_durations else 0
            
            aggregated_metrics["avg_volatility"] = total_volatility / self.iterations
            aggregated_metrics["avg_flash_crashes"] = total_flash_crashes / self.iterations
            aggregated_metrics["avg_crash_duration"] = avg_crash_duration
        
        return aggregated_metrics