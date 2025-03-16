# analysis_tools.py - Tools for analyzing simulation results

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any

def analyze_flash_crashes(results):
    """
    Analyze flash crash characteristics from simulation results.
    
    Args:
        results: Aggregated simulation results
        
    Returns:
        DataFrame: Flash crash statistics
    """
    crash_data = []
    
    for sim_idx, sim_data in results["simulations"].items():
        for crash in sim_data.get("flash_crashes", []):
            crash_data.append({
                "simulation": sim_idx,
                "start_time": crash["start"],
                "low_price": crash["low_price"],
                "pre_crash_price": crash["pre_crash_price"],
                "percent_drop": (1 - crash["low_price"] / crash["pre_crash_price"]) * 100,
                "duration": crash.get("duration", np.nan)
            })
    
    if not crash_data:
        return pd.DataFrame()
        
    df = pd.DataFrame(crash_data)
    return df

def compare_scenarios(baseline_results, lf_only_results, hf_cancel_results):
    """
    Compare different simulation scenarios.
    
    Args:
        baseline_results: Results from baseline scenario
        lf_only_results: Results from LF-only scenario
        hf_cancel_results: Dictionary of results with different cancellation rates
        
    Returns:
        DataFrame: Comparison of key metrics
    """
    comparison_data = {
        "Scenario": ["Baseline", "LF Only"],
        "Volatility": [
            baseline_results.get("avg_volatility", 0),
            lf_only_results.get("avg_volatility", 0)
        ],
        "Flash Crashes": [
            baseline_results.get("avg_flash_crashes", 0),
            lf_only_results.get("avg_flash_crashes", 0)
        ],
        "Avg Crash Duration": [
            baseline_results.get("avg_crash_duration", 0),
            lf_only_results.get("avg_crash_duration", 0)
        ]
    }
    
    # Add HF cancellation scenarios
    for rate, results in hf_cancel_results.items():
        comparison_data["Scenario"].append(f"HF Cancel = {rate}")
        comparison_data["Volatility"].append(results.get("avg_volatility", 0))
        comparison_data["Flash Crashes"].append(results.get("avg_flash_crashes", 0))
        comparison_data["Avg Crash Duration"].append(results.get("avg_crash_duration", 0))
    
    return pd.DataFrame(comparison_data)

def plot_price_series(results, simulation_idx=0, highlight_crashes=True):
    """
    Plot price series from a simulation, highlighting flash crashes.
    
    Args:
        results: Simulation results
        simulation_idx: Index of the simulation to plot
        highlight_crashes: Whether to highlight flash crash periods
    """
    sim_data = results["simulations"][simulation_idx]
    price_series = np.array(sim_data.get("price_series", []))
    
    if len(price_series) == 0:
        print("No price data available for plotting")
        return
    
    plt.figure(figsize=(12, 6))
    plt.plot(price_series, label='Asset Price')
    
    if highlight_crashes:
        for crash in sim_data.get("flash_crashes", []):
            start = crash["start"]
            end = crash.get("recovery_step", start + 30)
            plt.axvspan(start, end, color='red', alpha=0.3)
    
    plt.title('Asset Price Dynamics')
    plt.xlabel('Trading Session')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_order_concentration(results, simulation_idx=0):
    """
    Plot order concentration metrics for HF and LF traders.
    
    Args:
        results: Simulation results
        simulation_idx: Index of the simulation to plot
    """
    sim_data = results["simulations"][simulation_idx]
    
    # Check if we have the required data
    hft_sell = sim_data.get("hft_sell_concentration", [])
    lft_buy = sim_data.get("lft_buy_concentration", [])
    
    if not hft_sell or not lft_buy:
        print("Order concentration data not available")
        return
    
    plt.figure(figsize=(12, 6))
    plt.plot(hft_sell, label='HFT Sell Concentration', color='red')
    plt.plot(lft_buy, label='LFT Buy Concentration', color='blue')
    
    # Highlight flash crashes
    for crash in sim_data.get("flash_crashes", []):
        start = crash["start"]
        end = crash.get("recovery_step", start + 30)
        plt.axvspan(start, end, color='gray', alpha=0.3)
    
    plt.title('Order Concentration During Normal and Flash Crash Periods')
    plt.xlabel('Trading Session')
    plt.ylabel('Concentration Ratio')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_bid_ask_spread(results, simulation_idx=0):
    """
    Plot bid-ask spread during the simulation.
    
    Args:
        results: Simulation results
        simulation_idx: Index of the simulation to plot
    """
    sim_data = results["simulations"][simulation_idx]
    spreads = sim_data.get("bid_ask_spreads", [])
    
    if not spreads:
        print("Bid-ask spread data not available")
        return
    
    plt.figure(figsize=(12, 6))
    plt.plot(spreads, label='Bid-Ask Spread')
    
    # Highlight flash crashes
    for crash in sim_data.get("flash_crashes", []):
        start = crash["start"]
        end = crash.get("recovery_step", start + 30)
        plt.axvspan(start, end, color='red', alpha=0.3)
    
    plt.title('Bid-Ask Spread During Simulation')
    plt.xlabel('Trading Session')
    plt.ylabel('Spread')
    plt.legend()
    plt.grid(True)
    plt.show()