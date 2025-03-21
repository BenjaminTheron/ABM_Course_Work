# analysis_tools.py - Tools for analyzing simulation results

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import os


def create_performance_bar_plots(simulation_results, result_names, output_dir='./figures'):
    """
    Create and save individual bar plots for each performance metric across different simulation runs.
    Also plots additional detailed metrics including bid-ask spread time series and trader type metrics.
    
    Args:
        simulation_results: List of results from run_simulations() with different parameters
        result_names: List of names for each simulation result set
        output_dir: Directory to save the figures (will be created if it doesn't exist)
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.ticker import PercentFormatter
    import matplotlib.dates as mdates
    from matplotlib.gridspec import GridSpec
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Metrics to extract and compare
    metrics = ['avg_volatility', 'avg_flash_crashes', 'avg_crash_duration', 
               'avg_bid_ask_spread', 'avg_volume']
    
    # Create individual bar plot for each metric
    for metric in metrics:
        # Check if metric exists in all results
        if all(metric in result for result in simulation_results):
            # Extract data for this metric
            values = [result[metric] for result in simulation_results]
            
            # Create a new figure for this metric
            plt.figure(figsize=(10, 6))
            x = np.arange(len(values))
            
            # Create the bar plot
            bars = plt.bar(x, values, width=0.6)
            
            # Customize the plot
            plt.title(f'Comparison of {metric}')
            plt.xticks(x, result_names, rotation=45, ha='right')
            plt.ylabel(metric.replace('_', ' ').title())
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Save this metric's figure
            fig_path = os.path.join(output_dir, f'{metric}_comparison.png')
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"Saved {metric} comparison to {fig_path}")
            plt.close()
            
    # Extract flash crash counts from each simulation
    flash_crash_data = []
    for result in simulation_results:
        sim_crashes = []
        for sim_id, sim in result['simulations'].items():
            sim_crashes.append(sim['num_flash_crashes'])
        flash_crash_data.append(sim_crashes)
    
    # Create box plot of flash crash distribution
    plt.figure(figsize=(10, 6))
    plt.boxplot(flash_crash_data, labels=result_names)
    plt.title('Distribution of Flash Crashes Across Simulations')
    plt.ylabel('Number of Flash Crashes')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save flash crashes figure
    fig_path = os.path.join(output_dir, 'flash_crashes_distribution.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved flash crashes distribution to {fig_path}")
    plt.close()
    
    # NEW SECTION: Visualize bid-ask spread time series
    plt.figure(figsize=(12, 6))
    
    # Plot bid-ask spread time series for one simulation from each result
    for i, result in enumerate(simulation_results):
        # Get the first simulation's data
        sim_data = list(result['simulations'].values())[0]
        if 'bid_ask_spreads' in sim_data:
            spreads = sim_data['bid_ask_spreads']
            # Create x-axis representing time steps
            x_values = list(range(len(spreads)))
            # Plot the bid-ask spread time series with a label
            plt.plot(x_values, spreads, label=result_names[i], alpha=0.7)
    
    plt.title('Bid-Ask Spread Time Series')
    plt.xlabel('Simulation Step')
    plt.ylabel('Bid-Ask Spread')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save bid-ask spread time series figure
    fig_path = os.path.join(output_dir, 'bid_ask_spread_time_series.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved bid-ask spread time series to {fig_path}")
    plt.close()
    
    # NEW SECTION: Create trader type metric comparison plots
    # Check if all results have trader type metrics
    if all('avg_trader_types' in result for result in simulation_results):
        # Get all unique trader types across all simulations
        trader_types = set()
        for result in simulation_results:
            trader_types.update(result['avg_trader_types'].keys())
        
        # Create plots for each trader type
        for trader_type in trader_types:
            # Get metrics to visualize for this trader type
            trader_metrics = ['avg_pnl', 'avg_pnl_percent', 'avg_final_budget', 'avg_final_stock']
            
            # Create a figure with 2x2 subplots for trader type metrics
            fig, axs = plt.subplots(2, 2, figsize=(15, 10))
            axs = axs.flatten()
            
            for i, metric in enumerate(trader_metrics):
                # Get values for this metric across all simulations that have this trader type
                values = []
                labels = []
                
                for j, result in enumerate(simulation_results):
                    if trader_type in result.get('avg_trader_types', {}):
                        values.append(result['avg_trader_types'][trader_type].get(metric, 0))
                        labels.append(result_names[j])
                
                if values:  # Only create the plot if we have values
                    # Create bar plot
                    x = np.arange(len(values))
                    bars = axs[i].bar(x, values, width=0.6)
                    
                    # Add value labels on top of bars
                    for bar in bars:
                        height = bar.get_height()
                        axs[i].annotate(f'{height:.2f}',
                                    xy=(bar.get_x() + bar.get_width() / 2, height),
                                    xytext=(0, 3),  # 3 points vertical offset
                                    textcoords="offset points",
                                    ha='center', va='bottom')
                    
                    # Format y-axis as percentage for pnl_percent
                    if metric == 'avg_pnl_percent':
                        axs[i].yaxis.set_major_formatter(PercentFormatter(100))
                    
                    # Add labels
                    axs[i].set_title(f'{trader_type}: {metric.replace("_", " ").title()}')
                    axs[i].set_xticks(x)
                    axs[i].set_xticklabels(labels, rotation=45, ha='right')
            
            plt.tight_layout()
            
            # Save trader type metrics figure
            fig_path = os.path.join(output_dir, f'{trader_type}_metrics_comparison.png')
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"Saved {trader_type} metrics comparison to {fig_path}")
            plt.close()
        
        # Create a comprehensive trader performance comparison
        # Compare average PnL percentage across trader types and simulations
        plt.figure(figsize=(12, 8))
        
        # Prepare data for grouped bar chart
        simulation_count = len(simulation_results)
        bar_width = 0.8 / simulation_count
        
        for i, trader_type in enumerate(sorted(trader_types)):
            # Get PnL percentage for this trader type across all simulations
            for j, result in enumerate(simulation_results):
                if trader_type in result.get('avg_trader_types', {}):
                    pnl_percent = result['avg_trader_types'][trader_type].get('avg_pnl_percent', 0)
                    x_pos = i + (j - simulation_count/2 + 0.5) * bar_width
                    plt.bar(x_pos, pnl_percent, width=bar_width, label=f"{result_names[j]}" if i == 0 else "")
        
        plt.title('PnL Percentage Comparison Across Trader Types')
        plt.xlabel('Trader Type')
        plt.ylabel('Average PnL (%)')
        plt.xticks(range(len(sorted(trader_types))), sorted(trader_types))
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save comprehensive comparison figure
        fig_path = os.path.join(output_dir, 'trader_pnl_comparison.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Saved trader PnL comparison to {fig_path}")
        plt.close()

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