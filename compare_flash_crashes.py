#!/usr/bin/env python
# compare_flash_crashes.py - Compare flash crash statistics between simulation sets

import argparse
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import pandas as pd
import os
import json
import sys

def compare_flash_crashes(results_list, names_list, output_file=None):
    """
    Compare flash crash statistics between different simulation sets.
    
    Args:
        results_list: List of simulation results from run_simulations()
        names_list: List of names for each simulation set
        output_file: Optional path to save the output table
    """
    if len(results_list) != len(names_list):
        print("Error: Number of results sets must match number of names")
        return
    
    # Extract flash crash data
    data = []
    for i, (results, name) in enumerate(zip(results_list, names_list)):
        # Get number of simulations
        num_sims = results.get("num_simulations", 1)
        
        # Get average flash crashes
        avg_crashes = results.get("avg_flash_crashes", 0)
        
        # Count total crashes across all simulations in this set
        total_crashes = 0
        for sim_idx, sim_data in results.get("simulations", {}).items():
            crashes = sim_data.get("flash_crashes", [])
            total_crashes += len(crashes)
        
        # Store data
        data.append([name, num_sims, avg_crashes, total_crashes])
    
    # Create a formatted table
    headers = ["Simulation Set", "Number of MC Runs", "Avg. Flash Crashes", "Total Flash Crashes"]
    table = tabulate(data, headers=headers, tablefmt="grid")
    
    # Print the table
    print("\nFlash Crash Comparison:")
    print(table)
    
    # Save to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            f.write(table)
        print(f"Table saved to {output_file}")
    
    # Create a bar chart visualization
    create_bar_chart(data, headers, output_file)

def create_bar_chart(data, headers, output_file=None):
    """
    Create a bar chart visualization of flash crash statistics.
    
    Args:
        data: Table data
        headers: Column headers
        output_file: Base path for output file
    """
    # Extract data for plotting
    names = [row[0] for row in data]
    avg_crashes = [row[2] for row in data]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Bar chart for average flash crashes
    bars = plt.bar(names, avg_crashes, color='steelblue')
    
    # Add data labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height,
                 f'{height:.2f}',
                 ha='center', va='bottom', fontweight='bold')
    
    # Add titles and labels
    plt.title('Average Flash Crashes per Simulation Set', fontsize=16)
    plt.ylabel('Average Number of Flash Crashes')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save to file if specified
    if output_file:
        plot_file = output_file.rsplit('.', 1)[0] + '.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Chart saved to {plot_file}")
    
    plt.show()

def load_results(results_paths):
    """
    Load simulation results from files.
    
    Args:
        results_paths: List of paths to result JSON files
        
    Returns:
        List of loaded results
    """
    results_list = []
    
    for path in results_paths:
        try:
            with open(path, 'r') as f:
                results = json.load(f)
            results_list.append(results)
            print(f"Loaded results from {path}")
        except Exception as e:
            print(f"Error loading results from {path}: {e}")
    
    return results_list

def main():
    parser = argparse.ArgumentParser(description='Compare flash crash statistics between simulation sets')
    
    # Option to load results from files
    parser.add_argument('--files', type=str, help='Comma-separated list of result JSON files')
    
    # Names for each simulation set
    parser.add_argument('--names', type=str, required=True, 
                       help='Comma-separated list of names for each simulation set')
    
    # Output file
    parser.add_argument('--output', type=str, default='flash_crash_comparison.txt',
                       help='Output file to save the comparison table')
    
    args = parser.parse_args()
    
    # Parse names
    names_list = args.names.split(',')
    
    # Load results if files are specified
    if args.files:
        results_paths = args.files.split(',')
        results_list = load_results(results_paths)
    else:
        # If no files are specified, read from stdin
        print("No files specified. Please enter number of simulation results to read from stdin:")
        try:
            num_results = int(input())
            results_list = []
            
            for i in range(num_results):
                print(f"Enter JSON data for simulation {i+1}:")
                json_data = input()
                try:
                    results = json.loads(json_data)
                    results_list.append(results)
                except json.JSONDecodeError:
                    print(f"Error: Invalid JSON data for simulation {i+1}")
                    return
        except ValueError:
            print("Error: Please enter a valid number")
            return
    
    # Ensure we have the same number of names and results
    if len(results_list) != len(names_list):
        print(f"Error: Number of names ({len(names_list)}) must match number of result sets ({len(results_list)})")
        return
    
    # Compare flash crashes
    compare_flash_crashes(results_list, names_list, args.output)

if __name__ == "__main__":
    main()