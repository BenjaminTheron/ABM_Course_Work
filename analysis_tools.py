# analysis_tools.py - Tools for analyzing simulation results

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import os
from scipy.stats import norm
from statsmodels.nonparametric.kde import KDEUnivariate
from statsmodels.tsa.stattools import acf
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.animation import FuncAnimation
from IPython.display import display, clear_output



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

def stylized_facts_plots(results, output_dir='./figures/stylized_facts'):
    """
    Create plots showing stylized facts of financial time series from simulation results.
    
    Args:
        results (dict): Simulation results from run_simulations()
        output_dir (str): Directory to save the figures
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract price series from all simulations
    price_series_all = []
    print("Extracting price series from simulations...")
    
    for sim_idx, sim_data in results["simulations"].items():
        if "price_series" in sim_data:
            price_series = np.array(sim_data["price_series"])
            if len(price_series) > 0:
                price_series_all.append(price_series)
                print(f"Simulation {sim_idx}: {len(price_series)} prices")
    
    if not price_series_all:
        print("No price series data found in results")
        return
    
    # Calculate returns for all simulations
    returns_all = []
    for prices in price_series_all:
        # Ensure there are no zeros or negative values
        prices = np.maximum(prices, 0.0001)
        # Calculate log returns
        returns = np.diff(np.log(prices))
        # Remove any NaN or infinite values
        returns = returns[np.isfinite(returns)]
        if len(returns) > 0:
            returns_all.append(returns)
    
    if not returns_all:
        print("No valid returns data available")
        return
    
    # Pool all returns
    pooled_returns = np.concatenate(returns_all)
    print(f"Pooled {len(pooled_returns)} returns from {len(returns_all)} simulations")
    
    # --------------------------------
    # 1. Box-Whisker plots of price returns autocorrelations
    # --------------------------------
    max_lag = min(20, len(returns_all[0]) - 1)  # Ensure max_lag is not too large
    
    # Initialize data structure for boxplot
    acf_by_lag = [[] for _ in range(max_lag)]
    
    # Calculate autocorrelations for each simulation
    for returns in returns_all:
        if len(returns) > max_lag + 1:  # Need enough data
            try:
                acf_vals = acf(returns, nlags=max_lag, fft=True)
                # Skip lag 0
                for i, val in enumerate(acf_vals[1:], 0):
                    acf_by_lag[i].append(val)
            except Exception as e:
                print(f"Error in ACF calculation: {e}")
    
    # Ensure all lags have data
    valid_lags = []
    valid_acf_data = []
    
    for i, lag_data in enumerate(acf_by_lag):
        if lag_data:  # If there's data for this lag
            valid_lags.append(i + 1)  # +1 because we skipped lag 0
            valid_acf_data.append(lag_data)
    
    # Create boxplot
    if valid_acf_data:
        plt.figure(figsize=(12, 6))
        plt.boxplot(valid_acf_data, positions=valid_lags, widths=0.7)
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.title('Box-Whisker Plots of Price Returns Autocorrelations')
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/returns_autocorr_boxplot.png", dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print("Not enough data for boxplot")
    
    # --------------------------------
    # 2. ACF of absolute and squared returns
    # --------------------------------
    plt.figure(figsize=(12, 6))
    
    # Calculate average ACF of absolute returns and squared returns
    max_lag = min(20, len(returns_all[0]) - 1)  # Ensure max_lag is not too large
    lags = list(range(max_lag + 1))
    
    # Lists to store average ACFs
    avg_abs_acf = np.zeros(max_lag + 1)
    avg_squared_acf = np.zeros(max_lag + 1)
    count = 0
    
    for returns in returns_all:
        if len(returns) > max_lag + 1:
            try:
                # Absolute returns
                abs_returns = np.abs(returns)
                abs_acf_vals = acf(abs_returns, nlags=max_lag, fft=True)
                
                # Squared returns
                squared_returns = returns**2
                squared_acf_vals = acf(squared_returns, nlags=max_lag, fft=True)
                
                # Add to cumulative sum
                avg_abs_acf += abs_acf_vals
                avg_squared_acf += squared_acf_vals
                count += 1
            except Exception as e:
                print(f"Error in ACF calculation: {e}")
    
    if count > 0:
        # Calculate averages
        avg_abs_acf /= count
        avg_squared_acf /= count
        
        # Plot
        plt.plot(lags, avg_abs_acf, 'b-', linewidth=2, label='Absolute Returns')
        plt.plot(lags, avg_squared_acf, 'r--', linewidth=2, label='Squared Returns')
        
        plt.title('Autocorrelation Functions of Absolute and Squared Returns')
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/abs_squared_returns_acf.png", dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print("Not enough data for ACF plots")
    
    # --------------------------------
    # 3. Density of pooled returns vs Normal fit
    # --------------------------------
    plt.figure(figsize=(12, 6))
    
    # Filter out extreme values (more than 5 standard deviations)
    std_dev = np.std(pooled_returns)
    mean_return = np.mean(pooled_returns)
    filtered_returns = pooled_returns[np.abs(pooled_returns - mean_return) < 5 * std_dev]
    
    try:
        # Use numpy's histogram for density estimation instead of KDEUnivariate
        hist, bin_edges = np.histogram(filtered_returns, bins=50, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Normal fit
        x = np.linspace(np.min(filtered_returns), np.max(filtered_returns), 1000)
        normal_pdf = norm.pdf(x, loc=mean_return, scale=std_dev)
        
        # Plot
        plt.semilogy(x, normal_pdf, 'r-', linewidth=2, label='Normal Fit')
        plt.semilogy(bin_centers, hist, 'b*', markersize=3, label='Empirical Density')
        
        plt.title('Density of Pooled Price Returns vs Normal Fit')
        plt.xlabel('Return')
        plt.ylabel('Density (log scale)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/returns_density.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error creating density plot: {e}")
    
    print(f"Stylized facts plots saved to {output_dir}")

def compare_genomes(genome_paths, output_path=None, title="Comparison of Market Maker Genomes"):
    """
    Read multiple genome JSON files and compare them on a single radar plot.
    
    Args:
        genome_paths (list): List of paths to genome JSON files
        output_path (str, optional): Path to save the output plot
        title (str, optional): Title for the plot
        
    Returns:
        plotly.graph_objects.Figure: The radar plot figure
    """
    import json
    import os
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    genomes = []
    names = []
    
    # Define parameter bounds for normalization
    param_bounds = {
        "bid_spread_factor": (0.0001, 0.01),
        "ask_spread_factor": (0.0001, 0.01),
        "max_inventory_limit": (50, 1000),
        "hedge_ratio": (0.0, 1.0),
        "order_size_multiplier": (0.01, 0.3),
        "skew_factor": (0.0, 0.1)
    }
    
    # Read all genome files
    for path in genome_paths:
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                
            # Extract genome and name
            if "best_genome" in data:
                genome = data["best_genome"]
                genome_name = data.get("genome_name", os.path.basename(path))
                genomes.append(genome)
                names.append(genome_name)
            else:
                print(f"Warning: No 'best_genome' found in {path}")
                
        except Exception as e:
            print(f"Error reading genome file {path}: {e}")
    
    if not genomes:
        print("No valid genomes found")
        return None
    
    # Normalize all genomes
    normalized_genomes = []
    for genome in genomes:
        norm_genome = {}
        for param, (min_val, max_val) in param_bounds.items():
            if param in genome:
                # Normalize to [0, 1]
                norm_genome[param] = (genome[param] - min_val) / (max_val - min_val)
            else:
                print(f"Warning: Parameter '{param}' not found in genome")
                norm_genome[param] = 0.0
        normalized_genomes.append(norm_genome)
    
    # Create a subplot with 2 rows: radar plot and table
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        specs=[[{"type": "polar"}], [{"type": "table"}]],
        vertical_spacing=0.1
    )
    
    # Get all parameters
    all_params = list(param_bounds.keys())
    
    # Create color scale
    colors = ['rgba(31, 119, 180, 0.8)', 'rgba(255, 127, 14, 0.8)', 
              'rgba(44, 160, 44, 0.8)', 'rgba(214, 39, 40, 0.8)',
              'rgba(148, 103, 189, 0.8)', 'rgba(140, 86, 75, 0.8)',
              'rgba(227, 119, 194, 0.8)', 'rgba(127, 127, 127, 0.8)']
    
    # Prepare data for table
    table_header = ["<b>Genome Name</b>"] + [f"<b>{param}</b>" for param in all_params]
    table_cells = []
    
    # Add traces for each genome
    for i, (genome, norm_genome, name) in enumerate(zip(genomes, normalized_genomes, names)):
        color = colors[i % len(colors)]
        
        # Extract values
        r_values = [norm_genome[param] for param in all_params]
        # Close the loop
        r_values_closed = r_values + [r_values[0]]
        theta_closed = all_params + [all_params[0]]
        
        # Add radar trace to the first row
        fig.add_trace(go.Scatterpolar(
            r=r_values_closed,
            theta=theta_closed,
            fill='toself',
            name=name,
            line=dict(color=color),
            fillcolor=color.replace('0.8', '0.2')
        ), row=1, col=1)
        
        # Add markers with actual values as hover info
        actual_values = []
        for param in all_params:
            if param == "max_inventory_limit":
                actual_values.append(f"{genome[param]:.0f}")
            else:
                actual_values.append(f"{genome[param]:.4f}")
                
        fig.add_trace(go.Scatterpolar(
            r=r_values,
            theta=all_params,
            mode='markers',
            marker=dict(size=10, color=color),
            name=f"{name} values",
            text=actual_values,
            hoverinfo="text+theta",
            hovertext=[f"{param}: {value}" for param, value in zip(all_params, actual_values)],
            showlegend=False
        ), row=1, col=1)
        
        # Add row to table cells
        table_cells.append([f"<b>{name}</b>"] + actual_values)
    
    # Prepare better column headers with line breaks to avoid squashing
    formatted_headers = ["<b>Genome<br>Name</b>"]
    for param in all_params:
        # Replace underscores with spaces and add line breaks if needed
        formatted_param = param.replace("_", "<br>")
        formatted_headers.append(f"<b>{formatted_param}</b>")
    
    # Add table trace to second row
    fig.add_trace(go.Table(
        header=dict(
            values=formatted_headers,
            font=dict(size=12, color='rgb(50, 50, 50)'),
            align="center",
            height=40,  # Increase header height
            fill_color='rgba(200, 200, 200, 0.5)',
            line_color='rgba(100, 100, 100, 0.5)'
        ),
        cells=dict(
            values=list(map(list, zip(*table_cells))),  # Transpose the table cells for plotly format
            font=dict(size=11),
            align="center",
            height=30,  # Set consistent cell height
            fill_color='rgba(240, 240, 240, 0.5)',
            line_color='rgba(100, 100, 100, 0.5)'
        ),
        columnwidth=[0.18] + [0.137] * len(all_params)  # Adjust column widths (wider for name column)
    ), row=2, col=1)
    
    # Configure layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(size=10)
            ),
            angularaxis=dict(
                tickfont=dict(size=12, color='rgb(50, 50, 50)'),
                # Rotate the label positions for better readability
                rotation=0,
                direction='clockwise'
            ),
            domain=dict(x=[0, 1], y=[0.45, 1])  # Adjust the domain for the polar plot
        ),
        title=dict(
            text=title,
            y=0.98,
            x=0.5,
            xanchor='center',
            yanchor='top',
            font=dict(size=20)
        ),
        margin=dict(l=60, r=60, t=100, b=20),
        height=950,  # Increased height to accommodate improved table
        width=950,   # Slightly wider for better table display
        legend=dict(
            x=1.05,
            y=0.7,
            font=dict(size=12)
        )
    )
    
    # Save the figure if output path is provided
    if output_path:
        try:
            if output_path.lower().endswith('.html'):
                fig.write_html(output_path)
                print(f"Interactive plot saved to {output_path}")
            else:
                # For PNG output, increase resolution for better quality
                fig.write_image(output_path, scale=2)  # Scale=2 doubles the resolution
                print(f"Static image saved to {output_path}")
        except Exception as e:
            print(f"Error saving plot: {e}")
    
    return fig

def compare_all_genomes_in_directory(directory_path, output_path=None):
    """
    Find all JSON files in a directory and compare them on a radar plot.
    
    Args:
        directory_path (str): Path to directory containing genome JSON files
        output_path (str, optional): Path to save the output plot
        
    Returns:
        plotly.graph_objects.Figure: The radar plot figure
    """
    # Find all JSON files in the directory
    genome_paths = []
    for file in os.listdir(directory_path):
        if file.endswith('.json'):
            genome_paths.append(os.path.join(directory_path, file))
    
    if not genome_paths:
        print(f"No JSON files found in {directory_path}")
        return None
    
    print(f"Found {len(genome_paths)} genome files")
    return compare_genomes(genome_paths, output_path)

class MatplotlibAnimator:
    """
    Class for animating stock price using matplotlib during simulation
    """
    def __init__(self, update_frequency=1, window_size=500, figsize=(12, 10)):
        """
        Initialize the matplotlib animator
        
        Args:
            update_frequency: How often to update the plot (in simulation steps)
            window_size: Maximum number of data points to display
            figsize: Figure size (width, height) in inches
        """
        self.update_frequency = update_frequency
        self.window_size = window_size
        self.figsize = figsize
        
        # Data storage
        self.steps = []
        self.prices = []
        self.spreads = []
        self.volumes = []
        self.hft_concentration = []
        
        # Setup the figure and subplots
        self.setup_plot()
        
    def setup_plot(self):
        """Setup matplotlib figure and subplots"""
        self.fig, self.axs = plt.subplots(4, 1, figsize=self.figsize, 
                                          gridspec_kw={'height_ratios': [3, 1, 1, 1]}, 
                                          sharex=True)
        
        # Price plot - remove duplicate legend
        self.price_line, = self.axs[0].plot([], [], 'b-', linewidth=2, label='Price')
        
        # Bid-ask spread plot
        self.spread_line, = self.axs[1].plot([], [], 'g-', alpha=0.7, label='Bid-Ask Spread')
        
        # Volume plot (separate)
        self.volume_line, = self.axs[2].plot([], [], 'c-', alpha=0.7, label='Trading Volume')
        
        # HFT concentration plot (separate)
        self.hft_line, = self.axs[3].plot([], [], 'm-', alpha=0.7, label='HFT Sell Concentration')
        
        # Setup plot labels and legends - only one legend for price
        self.axs[0].set_title('Stock Price Simulation')
        self.axs[0].set_ylabel('Price')
        self.axs[0].legend(loc='upper left')
        self.axs[0].grid(True)
        
        self.axs[1].set_ylabel('Spread')
        self.axs[1].legend(loc='upper left')
        self.axs[1].grid(True)
        
        self.axs[2].set_ylabel('Volume')
        self.axs[2].legend(loc='upper left')
        self.axs[2].grid(True)
        
        self.axs[3].set_ylabel('HFT Concentration')
        self.axs[3].set_xlabel('Simulation Step')
        self.axs[3].legend(loc='upper left')
        self.axs[3].grid(True)
        
        # Enable tight layout
        self.fig.tight_layout()
        
        # Remove the stats textbox in the top-left as it overlaps with the legend
        
        # Show the plot
        plt.ion()  # Turn on interactive mode
        plt.show()
    
    def update(self, step, price, marketplace=None):
        """
        Update the animation with new data
        
        Args:
            step: Current simulation step
            price: Current stock price
            marketplace: Optional marketplace instance to get additional data
        """
        # Only update at specified frequency
        if step % self.update_frequency != 0:
            return
            
        # Add new data
        self.steps.append(step)
        self.prices.append(price)
        
        # Optional market data
        spread = marketplace.get_current_spread() if marketplace else 0
        self.spreads.append(spread)
        
        # Get volume data (use 1.0 as placeholder if not available)
        if marketplace and hasattr(marketplace, 'get_volume'):
            volume = marketplace.get_volume() / 1000  # Scale down for visualization
        else:
            volume = 1.0
        self.volumes.append(volume)
        
        # Get HFT concentration if available
        if marketplace and hasattr(marketplace, 'hft_sell_concentration_history') and len(marketplace.hft_sell_concentration_history) > 0:
            hft_conc = marketplace.hft_sell_concentration_history[-1]
            # No annotations for high concentration values
        else:
            hft_conc = 0.5  # Neutral value if not available
        self.hft_concentration.append(hft_conc)
        
        # Limit data to window size
        if len(self.steps) > self.window_size:
            self.steps = self.steps[-self.window_size:]
            self.prices = self.prices[-self.window_size:]
            self.spreads = self.spreads[-self.window_size:]
            self.volumes = self.volumes[-self.window_size:]
            self.hft_concentration = self.hft_concentration[-self.window_size:]
        
        self.update_plot(step)
    
    def update_plot(self, current_step):
        """Update the matplotlib plot with current data"""
        # Update price line
        self.price_line.set_data(self.steps, self.prices)
        
        # Update spread line
        self.spread_line.set_data(self.steps, self.spreads)
        
        # Update volume line (separate plot)
        self.volume_line.set_data(self.steps, self.volumes)
        
        # Update HFT concentration line (separate plot)
        self.hft_line.set_data(self.steps, self.hft_concentration)
        
        # No stats display - keeping the visualization clean with just the plots
        
        # Adjust axis limits
        for ax in self.axs:
            ax.relim()
            ax.autoscale_view()
        
        # Draw the updated figure
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
    
    def close(self):
        """Close the matplotlib figure"""
        plt.ioff()
        plt.close(self.fig)