#!/usr/bin/env python
# run_simulation.py - Entry point for running and testing the HFT market simulation

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any

# Import simulation components
from simulator import Simulator
from analysis_tools import analyze_flash_crashes, compare_scenarios, plot_price_series, plot_order_concentration, plot_bid_ask_spread

def run_baseline_simulation(config=None):
    """
    Run the baseline simulation with default parameters or given config.
    
    Args:
        config (dict, optional): Custom configuration for the simulation
        
    Returns:
        dict: Simulation results
    """
    if config is None:
        # Use default parameters
        from parameters import PARAMETERS
        config = PARAMETERS.copy()
    
    print("Running baseline simulation...")
    print(f"Configuration: {config}")
    
    # Create and run simulator
    simulator = Simulator(
        iterations=config.get("MC", 1),
        num_lf_traders=config.get("NL", 10000),
        num_hf_traders=config.get("NH", 100),
        parameters=config
    )
    
    results = simulator.run_simulations()
    print("Baseline simulation completed.")
    
    return results

def run_lf_only_simulation(config=None):
    """
    Run a simulation with only LF traders (no HFTs).
    
    Args:
        config (dict, optional): Custom configuration for the simulation
        
    Returns:
        dict: Simulation results
    """
    if config is None:
        # Use default parameters
        from parameters import PARAMETERS
        config = PARAMETERS.copy()
    
    # Modify config to set HF traders to 0
    lf_config = config.copy()
    lf_config["NH"] = 0
    
    print("Running LF-only simulation...")
    print(f"Configuration: {lf_config}")
    
    # Create and run simulator
    simulator = Simulator(
        iterations=lf_config.get("MC", 1),
        num_lf_traders=lf_config.get("NL", 10000),
        num_hf_traders=0,  # Force to zero for clarity
        parameters=lf_config
    )
    
    results = simulator.run_simulations()
    print("LF-only simulation completed.")
    
    return results

def run_hf_cancel_rate_simulations(cancel_rates=None, config=None):
    """
    Run simulations with different HFT order cancellation rates.
    
    Args:
        cancel_rates (list, optional): List of cancel rates to test
        config (dict, optional): Base configuration for the simulations
        
    Returns:
        dict: Dictionary of results for each cancel rate
    """
    if cancel_rates is None:
        cancel_rates = [1, 5, 10, 20]  # Default rates to test (trading periods)
    
    if config is None:
        # Use default parameters
        from parameters import PARAMETERS
        config = PARAMETERS.copy()
    
    results = {}
    
    for rate in cancel_rates:
        print(f"Running simulation with HFT cancel rate = {rate}...")
        
        # Modify config to set HF cancel rate
        hf_config = config.copy()
        hf_config["gamma_H"] = rate
        
        # Create and run simulator
        simulator = Simulator(
            iterations=hf_config.get("MC", 1),
            num_lf_traders=hf_config.get("NL", 10000),
            num_hf_traders=hf_config.get("NH", 100),
            parameters=hf_config
        )
        
        sim_results = simulator.run_simulations()
        results[rate] = sim_results
        print(f"Simulation with HFT cancel rate = {rate} completed.")
    
    return results

def analyze_simulation_results(results, output_folder="./results"):
    """
    Analyze and visualize simulation results.
    
    Args:
        results: Simulation results dictionary
        output_folder: Folder to save outputs
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get the first simulation results for analysis
    if "simulations" in results and 0 in results["simulations"]:
        sim_data = results["simulations"][0]
        
        # Analyze flash crashes
        if "flash_crashes" in sim_data and sim_data["flash_crashes"]:
            print(f"Found {len(sim_data['flash_crashes'])} flash crashes in simulation")
            
            # Save crash data
            with open(f"{output_folder}/flash_crashes.json", "w") as f:
                json.dump(sim_data["flash_crashes"], f, indent=2)
            
            # Basic statistics
            crash_durations = [crash["duration"] for crash in sim_data["flash_crashes"] 
                              if "duration" in crash and crash["duration"] is not None]
            
            if crash_durations:
                avg_duration = sum(crash_durations) / len(crash_durations)
                print(f"Average flash crash duration: {avg_duration:.2f} periods")
                
                # Plot price series with crash highlights
                plt.figure(figsize=(12, 6))
                price_series = sim_data["price_series"]
                plt.plot(price_series)
                
                # Highlight crash periods
                for crash in sim_data["flash_crashes"]:
                    start = crash["start"]
                    end = crash.get("recovery_step", start + 30)
                    plt.axvspan(start, end, color='red', alpha=0.3)
                
                plt.title("Price Series with Flash Crash Periods Highlighted")
                plt.xlabel("Trading Period")
                plt.ylabel("Price")
                plt.grid(True)
                plt.savefig(f"{output_folder}/price_series_with_crashes.png")
                plt.close()
                
        # Plot bid-ask spreads
        if "bid_ask_spreads" in sim_data and sim_data["bid_ask_spreads"]:
            plt.figure(figsize=(12, 6))
            spreads = sim_data["bid_ask_spreads"]
            plt.plot(spreads)
            
            # Highlight crash periods if available
            if "flash_crashes" in sim_data:
                for crash in sim_data["flash_crashes"]:
                    start = crash["start"]
                    end = crash.get("recovery_step", start + 30)
                    plt.axvspan(start, end, color='red', alpha=0.3)
            
            plt.title("Bid-Ask Spread During Simulation")
            plt.xlabel("Trading Period")
            plt.ylabel("Spread")
            plt.grid(True)
            plt.savefig(f"{output_folder}/bid_ask_spreads.png")
            plt.close()
            
        # Plot HFT concentration metrics
        if ("hft_sell_concentration" in sim_data and 
            "lft_buy_concentration" in sim_data):
            plt.figure(figsize=(12, 6))
            
            hft_sell = sim_data["hft_sell_concentration"]
            lft_buy = sim_data["lft_buy_concentration"]
            
            # Ensure they're the same length for plotting
            min_len = min(len(hft_sell), len(lft_buy))
            plt.plot(hft_sell[:min_len], label='HFT Sell Concentration', color='red')
            plt.plot(lft_buy[:min_len], label='LFT Buy Concentration', color='blue')
            
            # Highlight crash periods if available
            if "flash_crashes" in sim_data:
                for crash in sim_data["flash_crashes"]:
                    start = crash["start"]
                    end = crash.get("recovery_step", start + 30)
                    if start < min_len:  # Only highlight if within plot range
                        plt.axvspan(start, min(end, min_len-1), color='gray', alpha=0.3)
            
            plt.title("Order Concentration During Simulation")
            plt.xlabel("Trading Period")
            plt.ylabel("Concentration Ratio")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{output_folder}/order_concentration.png")
            plt.close()
    
    # Save the full results as JSON
    with open(f"{output_folder}/simulation_results.json", "w") as f:
        # Create a simplified version that's JSON-serializable
        serializable_results = {
            "num_simulations": results.get("num_simulations", 1),
            "avg_volatility": float(results.get("avg_volatility", 0)),
            "avg_flash_crashes": float(results.get("avg_flash_crashes", 0)),
            "avg_crash_duration": float(results.get("avg_crash_duration", 0))
        }
        json.dump(serializable_results, f, indent=2)

def run_parameter_sensitivity(parameter_name, parameter_values, base_config=None):
    """
    Run simulations with different values for a specific parameter.
    
    Args:
        parameter_name: Name of the parameter to vary
        parameter_values: List of values to test
        base_config: Base configuration to use
        
    Returns:
        dict: Dictionary of results for each parameter value
    """
    if base_config is None:
        # Use default parameters
        from parameters import PARAMETERS
        base_config = PARAMETERS.copy()
    
    results = {}
    
    for value in parameter_values:
        print(f"Running simulation with {parameter_name} = {value}...")
        
        # Modify config to set parameter value
        config = base_config.copy()
        config[parameter_name] = value
        
        # Create and run simulator
        simulator = Simulator(
            iterations=config.get("MC", 1),
            num_lf_traders=config.get("NL", 10000),
            num_hf_traders=config.get("NH", 100),
            parameters=config
        )
        
        sim_results = simulator.run_simulations()
        results[value] = sim_results
        print(f"Simulation with {parameter_name} = {value} completed.")
    
    return results

def compare_all_scenarios(results_dict, output_folder="./results"):
    """
    Compare and visualize results from different scenarios.
    
    Args:
        results_dict: Dictionary of scenario names and results
        output_folder: Folder to save outputs
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Extract key metrics for each scenario
    scenarios = []
    volatilities = []
    flash_crashes = []
    crash_durations = []
    
    for scenario, results in results_dict.items():
        scenarios.append(scenario)
        volatilities.append(results.get("avg_volatility", 0))
        flash_crashes.append(results.get("avg_flash_crashes", 0))
        crash_durations.append(results.get("avg_crash_duration", 0))
    
    # Create comparison dataframe
    df = pd.DataFrame({
        "Scenario": scenarios,
        "Volatility": volatilities,
        "Flash Crashes": flash_crashes,
        "Avg Crash Duration": crash_durations
    })
    
    # Save to CSV
    df.to_csv(f"{output_folder}/scenario_comparison.csv", index=False)
    
    # Plot comparisons
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    
    # Volatility
    axes[0].bar(scenarios, volatilities)
    axes[0].set_title("Average Volatility by Scenario")
    axes[0].set_ylabel("Volatility")
    axes[0].grid(axis='y')
    
    # Flash Crashes
    axes[1].bar(scenarios, flash_crashes)
    axes[1].set_title("Average Number of Flash Crashes by Scenario")
    axes[1].set_ylabel("Flash Crashes")
    axes[1].grid(axis='y')
    
    # Crash Duration
    axes[2].bar(scenarios, crash_durations)
    axes[2].set_title("Average Flash Crash Duration by Scenario")
    axes[2].set_ylabel("Duration (periods)")
    axes[2].grid(axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{output_folder}/scenario_comparison.png")
    plt.close()

def print_help():
    """Print help information about available commands"""
    print("\nHFT Market Simulation - Available Commands:")
    print("-------------------------------------------")
    print("baseline                - Run baseline simulation")
    print("lf_only                 - Run simulation with only LF traders")
    print("cancel_rates            - Test different HFT cancel rates")
    print("sensitivity PARAM VALUES - Test parameter sensitivity")
    print("                          Example: sensitivity theta 10,20,30,40")
    print("help                    - Display this help message")
    print("exit                    - Exit the program")
    print("\nExamples:")
    print("  baseline")
    print("  lf_only")
    print("  cancel_rates 1,5,10,20")
    print("  sensitivity NH 50,100,200,500")
    print("-------------------------------------------")

def parse_param_values(values_str):
    """Parse a comma-separated string of values into the appropriate types"""
    values = values_str.split(',')
    parsed_values = []
    
    for val in values:
        val = val.strip()
        try:
            # Try to convert to int
            parsed_values.append(int(val))
        except ValueError:
            try:
                # Try to convert to float
                parsed_values.append(float(val))
            except ValueError:
                # Keep as string
                parsed_values.append(val)
    
    return parsed_values

def main():
    """Main function to interactively run simulations"""
    print("HFT Market Simulation Runner")
    print("Type 'help' to see available commands")
    
    while True:
        try:
            command = input("\nEnter command: ").strip().lower()
            
            if command == "help":
                print_help()
            
            elif command == "exit":
                print("Exiting simulation runner")
                break
            
            elif command == "baseline":
                results = run_baseline_simulation()
                analyze_simulation_results(results)
                print("Baseline results saved to ./results folder")
            
            elif command == "lf_only":
                results = run_lf_only_simulation()
                analyze_simulation_results(results, "./results/lf_only")
                print("LF-only results saved to ./results/lf_only folder")
            
            elif command.startswith("cancel_rates"):
                parts = command.split()
                rates = [1, 5, 10, 20]  # Default
                
                if len(parts) > 1:
                    rates = parse_param_values(parts[1])
                
                results = run_hf_cancel_rate_simulations(rates)
                
                # Analyze each result separately
                for rate, result in results.items():
                    analyze_simulation_results(result, f"./results/cancel_rate_{rate}")
                
                # Compare all results
                compare_all_scenarios(
                    {f"Cancel Rate {r}": results[r] for r in rates},
                    "./results/cancel_rates_comparison"
                )
                print("Cancel rate results saved to ./results/cancel_rate_* folders")
            
            elif command.startswith("sensitivity"):
                parts = command.split()
                
                if len(parts) < 3:
                    print("Error: Missing parameter name or values")
                    print("Usage: sensitivity PARAM VALUES")
                    continue
                
                param_name = parts[1]
                param_values = parse_param_values(parts[2])
                
                print(f"Running sensitivity analysis for {param_name} with values {param_values}")
                results = run_parameter_sensitivity(param_name, param_values)
                
                # Analyze each result separately
                for value, result in results.items():
                    analyze_simulation_results(result, f"./results/sensitivity_{param_name}_{value}")
                
                # Compare all results
                compare_all_scenarios(
                    {f"{param_name}={v}": results[v] for v in param_values},
                    f"./results/sensitivity_{param_name}_comparison"
                )
                print(f"Sensitivity results saved to ./results/sensitivity_{param_name}_* folders")
            
            else:
                print(f"Unknown command: {command}")
                print("Type 'help' to see available commands")
        
        except KeyboardInterrupt:
            print("\nOperation cancelled. Type 'exit' to quit or try another command.")
        
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()