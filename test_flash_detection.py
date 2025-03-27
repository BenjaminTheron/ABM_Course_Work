from simulator import Simulator
from parameters import PARAMETERS
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def main():
    params0 = PARAMETERS.copy()
    params1 = PARAMETERS.copy()
    params0["MC"] = 10
    params1["MC"] = 10
    params1["NH"] = 0

    test_detection(params0, "detection_with_hft")
    test_detection(params1, "detection_without_hft")

def test_detection(params, output_dir):
    """"
    Tests flash clash detection parameters on simulation with and without high frequency traders.

    """

    
    # Define different parameter configurations
    configs = [
        {"detection_window": 30, "lookback": 5, "name": "lf = 30 lb = 5"},
        {"detection_window": 40, "lookback": 5, "name": "lf = 40 lb = 5"},
        {"detection_window": 50, "lookback": 5, "name": "lf = 50 lb = 5"},
        {"detection_window": 30, "lookback": 10, "name": "lf = 30 lb = 10"},
        {"detection_window": 40, "lookback": 10, "name": "lf = 40 lb = 10"},
        {"detection_window": 50, "lookback": 10, "name": "lf = 50 lb = 10"}
    ]
    
    # Store results for each configuration
    all_results = []
    
    # Run simulations for each configuration
    for config in configs:
        simulator = Simulator(
            parameters=params,
            detection_window=config["detection_window"],
            lookback=config["lookback"],
            animate = True
        )
        
        results = simulator.run_simulations()
        
        # Add results to our list
        all_results.append({
            "name": config["name"],
            "detection_window": config["detection_window"],
            "lookback": config["lookback"],
            "avg_flash_crashes": results["avg_flash_crashes"],
            "avg_crash_duration": results.get("avg_crash_duration", 0)
        })
    
    # Convert results to a pandas DataFrame
    df = pd.DataFrame(all_results)
    
    # Create results directory if it doesn't exist
    os.makedirs("./results", exist_ok=True)
    os.makedirs("./results/" + output_dir, exist_ok=True)
    
    # Print the results table to console
    print("\nFlash Crash Analysis Results:")
    print(df)
    
    # Save results as CSV
    csv_file = "./results/" + output_dir + "/flash_crash_analysis.csv"
    df.to_csv(csv_file, index=False)
    print(f"\nResults saved as CSV to {csv_file}")
    
    # Create a matplotlib table visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = df.values
    column_labels = df.columns
    
    # Format the numeric columns
    for i in range(len(table_data)):
        for j in [3, 4]:  # Avg Flash Crashes and Avg Crash Duration columns
            table_data[i, j] = f"{table_data[i, j]:.2f}"
    
    # Create the table
    table = ax.table(
        cellText=table_data,
        colLabels=column_labels,
        loc='center',
        cellLoc='center',
        colColours=['#4472C4'] * len(column_labels),
        colLoc='center'
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)  # Adjust table scale
    
    # Change text color of header to white
    for j, cell in enumerate(table._cells[(0, j)] for j in range(len(column_labels))):
        cell.set_text_props(color='white', fontweight='bold')
    
    # Add alternating row colors
    for i in range(len(table_data)):
        for j in range(len(column_labels)):
            cell = table._cells[(i+1, j)]
            if i % 2 == 0:
                cell.set_facecolor('#E6F0FF')
            else:
                cell.set_facecolor('#FFFFFF')
    
    # Add title
    plt.title('Flash Crash Analysis Results', fontsize=16, pad=20)
    plt.tight_layout()
    
    # Save table as image
    table_file = "./results/" + output_dir + "/flash_crash_table.png"
    plt.savefig(table_file, dpi=300, bbox_inches='tight')
    print(f"Table visualization saved to {table_file}")
    
    # Create a bar chart visualization for comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Extract data for plot
    configs = df['name']
    flash_crashes = df['avg_flash_crashes']
    
    # Create the bar chart
    bars = ax.bar(configs, flash_crashes, color='#4472C4')
    
    # Add data labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),  # 3 points vertical offset
                   textcoords="offset points",
                   ha='center', va='bottom', fontweight='bold')
    
    # Customize the plot
    ax.set_ylabel('Average Number of Flash Crashes')
    ax.set_xlabel('Configuration')
    ax.set_title('Average Flash Crashes by Configuration', fontsize=16)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the bar chart
    chart_file = "./results/" + output_dir + "/flash_crash_chart.png"
    plt.savefig(chart_file, dpi=300, bbox_inches='tight')
    print(f"Bar chart saved to {chart_file}")

if __name__ == "__main__":
    main()