#!/usr/bin/env python
# parameter_sensitivity.py - Analyze the sensitivity of simulation results to parameter values

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from simulator import Simulator
from parameters import PARAMETERS

def run_sensitivity_analysis(param_names, param_values_list, output_dir="./sensitivity_results"):
    """
    Run sensitivity analysis for specified parameters.
    
    Args:
        param_names: List of parameter names to analyze
        param_values_list: List of lists containing parameter values to test
        output_dir: Directory to save the results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Loop through each parameter
    for i, param_name in enumerate(param_names):
        param_values = param_values_list[i]
        
        print(f"Running sensitivity analysis for parameter: {param_name}")
        print(f"Testing values: {param_values}")
        
        # Store results for this parameter
        flash_crashes = []
        crash_durations = []
        volatilities = []
        volumes = []
        spreads = []
        
        # Run simulation for each parameter value
        for value in param_values:
            # Create parameters with this value
            params = PARAMETERS.copy()
            params[param_name] = value
            
            # Force a reasonable number of simulations for sensitivity analysis
            params["MC"] = 10  # Run 10 replications for each parameter value
            
            print(f"Testing {param_name} = {value}")
            
            # Create and run simulator
            simulator = Simulator(parameters=params)
            results = simulator.run_simulations()
            
            # Store the results
            flash_crashes.append(results.get("avg_flash_crashes", 0))
            crash_durations.append(results.get("avg_crash_duration", 0))
            volatilities.append(results.get("avg_volatility", 0))
            volumes.append(results.get("avg_volume", 0))
            spreads.append(results.get("avg_bid_ask_spread", 0))
        
        # Create plots for this parameter
        create_parameter_plots(
            param_name,
            param_values,
            flash_crashes,
            crash_durations,
            volatilities,
            volumes,
            spreads,
            output_dir
        )

def create_parameter_plots(param_name, param_values, flash_crashes, crash_durations, 
                          volatilities, volumes, spreads, output_dir):
    """
    Create and save plots for a parameter's sensitivity analysis.
    
    Args:
        param_name: The parameter name
        param_values: List of parameter values tested
        flash_crashes: List of average flash crash counts
        crash_durations: List of average crash durations
        volatilities: List of average volatilities
        volumes: List of average volumes
        spreads: List of average bid-ask spreads
        output_dir: Directory to save the plots
    """
    # Format parameter name and values for display
    friendly_name = param_name.replace('_', ' ').title()
    
    # 1. Flash crashes plot
    plt.figure(figsize=(10, 6))
    plt.plot(param_values, flash_crashes, 'b-o', linewidth=2)
    plt.title(f'Sensitivity of Flash Crashes to {friendly_name}')
    plt.xlabel(friendly_name)
    plt.ylabel('Average Number of Flash Crashes')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/{param_name}_flash_crashes.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Crash durations plot
    plt.figure(figsize=(10, 6))
    plt.plot(param_values, crash_durations, 'r-o', linewidth=2)
    plt.title(f'Sensitivity of Crash Duration to {friendly_name}')
    plt.xlabel(friendly_name)
    plt.ylabel('Average Crash Duration (periods)')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/{param_name}_crash_durations.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Volatility plot
    plt.figure(figsize=(10, 6))
    plt.plot(param_values, volatilities, 'g-o', linewidth=2)
    plt.title(f'Sensitivity of Volatility to {friendly_name}')
    plt.xlabel(friendly_name)
    plt.ylabel('Average Volatility')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/{param_name}_volatility.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Volume plot
    plt.figure(figsize=(10, 6))
    plt.plot(param_values, volumes, 'm-o', linewidth=2)
    plt.title(f'Sensitivity of Trading Volume to {friendly_name}')
    plt.xlabel(friendly_name)
    plt.ylabel('Average Trading Volume')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/{param_name}_volume.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Bid-Ask Spread plot
    plt.figure(figsize=(10, 6))
    plt.plot(param_values, spreads, 'c-o', linewidth=2)
    plt.title(f'Sensitivity of Bid-Ask Spread to {friendly_name}')
    plt.xlabel(friendly_name)
    plt.ylabel('Average Bid-Ask Spread')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/{param_name}_spread.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Combined plot
    fig, axs = plt.subplots(5, 1, figsize=(12, 15), sharex=True)
    
    # Normalize each metric for better visualization
    normalized_metrics = [
        normalize(flash_crashes),
        normalize(crash_durations),
        normalize(volatilities),
        normalize(volumes),
        normalize(spreads)
    ]
    
    metrics_names = ['Flash Crashes', 'Crash Duration', 'Volatility', 'Volume', 'Bid-Ask Spread']
    colors = ['blue', 'red', 'green', 'magenta', 'cyan']
    
    for i, (metric, name, color) in enumerate(zip(normalized_metrics, metrics_names, colors)):
        axs[i].plot(param_values, metric, f'{color}-o', linewidth=2)
        axs[i].set_ylabel(name)
        axs[i].grid(True, alpha=0.3)
    
    axs[-1].set_xlabel(friendly_name)
    fig.suptitle(f'Sensitivity Analysis for {friendly_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(f"{output_dir}/{param_name}_combined.png", dpi=300, bbox_inches='tight')
    plt.close()

def normalize(values):
    """
    Normalize a list of values to the range [0, 1].
    
    Args:
        values: List of values to normalize
        
    Returns:
        Normalized values
    """
    min_val = min(values)
    max_val = max(values)
    
    if max_val == min_val:
        return [0.5] * len(values)
    
    return [(val - min_val) / (max_val - min_val) for val in values]

def main():
    parser = argparse.ArgumentParser(description='Run parameter sensitivity analysis')
    
    parser.add_argument('--params', required=True, type=str, 
                        help='Comma-separated list of parameter names to analyze')
    parser.add_argument('--values', required=True, type=str, 
                        help='Comma-separated list of parameter values to test (use semicolons to separate values for different parameters)')
    parser.add_argument('--output', type=str, default='./sensitivity_results',
                        help='Output directory for sensitivity analysis results')
    
    args = parser.parse_args()
    
    # Parse parameter names
    param_names = args.params.split(',')
    
    # Parse parameter values for each parameter
    param_values_lists = []
    values_groups = args.values.split(';')
    
    for i, values_str in enumerate(values_groups):
        # Parse the values for this parameter
        try:
            # Try to parse as float values
            values = [float(val.strip()) for val in values_str.split(',')]
            param_values_lists.append(values)
        except ValueError:
            print(f"Error: Invalid parameter values for parameter {param_names[i]}")
            return
    
    # Ensure we have values for each parameter
    if len(param_names) != len(param_values_lists):
        print("Error: Number of parameter names must match number of parameter value lists")
        return
    
    # Run sensitivity analysis
    run_sensitivity_analysis(param_names, param_values_lists, args.output)

if __name__ == "__main__":
    main()