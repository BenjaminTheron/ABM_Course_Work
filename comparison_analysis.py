# comparison_analysis.py

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Import simulation components
from simulator import Simulator
from parameters import PARAMETERS
from genetic_evolution import GeneticAlgorithm

class MarketAnalysis:
    """
    Analyze and compare market conditions with and without market makers
    using the optimized parameters from genetic evolution.
    """
    
    def run_analysis(self):
        """
        Run the complete analysis pipeline:
        1. Load the best market maker genome
        2. Run baseline simulations (no market maker)
        3. Run market maker simulations
        4. Compare and analyze the results
        """
        # Load the best genome
        best_genome = self.load_best_genome()
        print(f"Loaded best genome: {best_genome}")
        
        # Run baseline simulations
        baseline_results = self.run_baseline_simulations()
        
        # Run market maker simulations
        mm_results = self.run_mm_simulations(best_genome)
        
        # Compare and analyze results
        self.compare_and_analyze(baseline_results, mm_results)
        
        # Export consolidated data for further analysis
        self.export_consolidated_data(baseline_results, mm_results)
        
        print(f"Analysis complete. Results saved to {self.output_dir}")
    
    def export_consolidated_data(self, baseline_results, mm_results):
        """
        Export consolidated data from all simulations for further analysis.
        
        Args:
            baseline_results: Results from baseline simulations
            mm_results: Results from market maker simulations
        """
        # Create data structures for price series and other time series
        price_data = []
        spread_data = []
        
        # Process baseline results
        for i, result in enumerate(baseline_results):
            sim_result = result["simulations"][0] if "simulations" in result else result
            
            # Price series
            if "price_series" in sim_result:
                for t, price in enumerate(sim_result["price_series"]):
                    price_data.append({
                        "condition": "baseline",
                        "simulation": i,
                        "timestep": t,
                        "price": price
                    })
            
            # Spread series
            if "bid_ask_spreads" in sim_result:
                for t, spread in enumerate(sim_result["bid_ask_spreads"]):
                    spread_data.append({
                        "condition": "baseline",
                        "simulation": i,
                        "timestep": t,
                        "spread": spread
                    })
        
        # Process market maker results
        for i, result in enumerate(mm_results):
            sim_result = result["simulations"][0] if "simulations" in result else result
            
            # Price series
            if "price_series" in sim_result:
                for t, price in enumerate(sim_result["price_series"]):
                    price_data.append({
                        "condition": "market_maker",
                        "simulation": i,
                        "timestep": t,
                        "price": price
                    })
            
            # Spread series
            if "bid_ask_spreads" in sim_result:
                for t, spread in enumerate(sim_result["bid_ask_spreads"]):
                    spread_data.append({
                        "condition": "market_maker",
                        "simulation": i,
                        "timestep": t,
                        "spread": spread
                    })
        
        # Convert to DataFrames and save
        price_df = pd.DataFrame(price_data)
        spread_df = pd.DataFrame(spread_data)
        
        price_df.to_csv(f"{self.output_dir}/price_series.csv", index=False)
        spread_df.to_csv(f"{self.output_dir}/spread_series.csv", index=False)
        
        # Generate additional visualizations
        
        # Price volatility over time
        plt.figure(figsize=(15, 8))
        
        # Calculate rolling volatility
        window_size = 50
        for condition in ["baseline", "market_maker"]:
            for sim in price_df[price_df["condition"] == condition]["simulation"].unique():
                data = price_df[(price_df["condition"] == condition) & (price_df["simulation"] == sim)]
                if len(data) > window_size:
                    # Calculate returns
                    prices = data.sort_values("timestep")["price"].values
                    returns = np.diff(prices) / prices[:-1]
                    
                    # Calculate rolling volatility
                    rolling_vol = pd.Series(returns).rolling(window=window_size).std().dropna().values
                    timesteps = data.sort_values("timestep")["timestep"].values[window_size:]
                    
                    plt.plot(timesteps, rolling_vol, alpha=0.5, 
                             color="blue" if condition == "baseline" else "red")
        
        # Add legend patches
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="blue", label="Baseline"),
            Patch(facecolor="red", label="Market Maker")
        ]
        plt.legend(handles=legend_elements)
        
        plt.title("Rolling Volatility Over Time")
        plt.xlabel("Time Step")
        plt.ylabel(f"Volatility (Rolling {window_size}-Period Std Dev of Returns)")
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{self.output_dir}/rolling_volatility.png")
        plt.close()
    def __init__(self, num_simulations=5, output_dir="./results/comparison"):
        """
        Initialize the market analysis.
        
        Args:
            num_simulations: Number of simulations to run for each condition
            output_dir: Directory to save analysis results
        """
        self.num_simulations = num_simulations
        self.output_dir = output_dir
        self.parameters = PARAMETERS.copy()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def load_best_genome(self, evolution_results_path="./results/genetic/best_genomes.csv"):
        """
        Load the best genome from evolution results.
        
        Args:
            evolution_results_path: Path to the evolution results CSV
            
        Returns:
            dict: Best genome parameters
        """
        try:
            # Load the evolution results
            df = pd.read_csv(evolution_results_path)
            
            # Get the last (best) genome
            best_genome = df.iloc[-1].to_dict()
            
            # Remove generation column if present
            if "generation" in best_genome:
                del best_genome["generation"]
            
            return best_genome
        except (FileNotFoundError, pd.errors.EmptyDataError):
            print(f"Could not load best genome from {evolution_results_path}")
            print("Using default market maker parameters")
            
            # Return default parameters
            return {
                "bid_spread_factor": 0.001,
                "ask_spread_factor": 0.001,
                "max_inventory_limit": 500,
                "hedge_ratio": 0.3,
                "order_size_multiplier": 0.1,
                "skew_factor": 0.01
            }
    
    def run_baseline_simulations(self):
        """
        Run baseline simulations without market makers.
        
        Returns:
            List of simulation results
        """
        print("Running baseline simulations (no market maker)...")
        
        baseline_results = []
        
        for i in range(self.num_simulations):
            print(f"Baseline simulation {i+1}/{self.num_simulations}")
            
            # Create simulator without market maker
            simulator = Simulator(
                iterations=1,
                num_lf_traders=self.parameters.get("NL", 10000),
                num_hf_traders=self.parameters.get("NH", 100),
                parameters=self.parameters,
                use_market_maker=False
            )
            
            # Run simulation
            results = simulator.run_simulations()
            baseline_results.append(results)
        
        return baseline_results
    
    def run_mm_simulations(self, mm_params):
        """
        Run simulations with market makers using the optimized parameters.
        
        Args:
            mm_params: Market maker parameters
            
        Returns:
            List of simulation results
        """
        print("Running market maker simulations...")
        
        mm_results = []
        
        for i in range(self.num_simulations):
            print(f"Market maker simulation {i+1}/{self.num_simulations}")
            
            # Create simulator with market maker
            simulator = Simulator(
                iterations=1,
                num_lf_traders=self.parameters.get("NL", 10000),
                num_hf_traders=self.parameters.get("NH", 100),
                parameters=self.parameters,
                use_market_maker=True,
                market_maker_params=mm_params
            )
            
            # Run simulation
            results = simulator.run_simulations()
            mm_results.append(results)
        
        return mm_results
    
    def compare_and_analyze(self, baseline_results, mm_results):
        """
        Compare and analyze the results from both types of simulations.
        
        Args:
            baseline_results: Results from baseline simulations
            mm_results: Results from market maker simulations
        """
        # Extract key metrics for comparison
        metrics = {
            "volatility": [],
            "flash_crashes": [],
            "crash_duration": [],
            "avg_spread": [],
            "volume": []
        }
        
        for condition in ["baseline", "market_maker"]:
            results = baseline_results if condition == "baseline" else mm_results
            
            for result in results:
                # Get the first (and only) simulation
                sim_result = result["simulations"][0] if "simulations" in result else result
                
                # Extract metrics
                metrics["volatility"].append((condition, sim_result["volatility"]))
                metrics["flash_crashes"].append((condition, sim_result["num_flash_crashes"]))
                metrics["crash_duration"].append((condition, sim_result["avg_crash_duration"]))
                metrics["avg_spread"].append((condition, sim_result["avg_bid_ask_spread"]))
                metrics["volume"].append((condition, sim_result.get("total_volume", 0)))
        
        # Convert to DataFrames for analysis
        dfs = {}
        for metric, values in metrics.items():
            dfs[metric] = pd.DataFrame(values, columns=["Condition", metric])
        
        # Generate visualizations
        self.generate_comparison_visualizations(dfs)
        
        # Run statistical tests
        self.run_statistical_tests(dfs)
        
        # Analyze flash crash characteristics
        self.analyze_flash_crashes(baseline_results, mm_results)
        
        # Analyze market maker activity
        self.analyze_market_maker(mm_results)
    
    def generate_comparison_visualizations(self, metric_dfs):
        """
        Generate visualizations comparing baseline and market maker conditions.
        
        Args:
            metric_dfs: Dictionary of DataFrames for each metric
        """
        # Create figure for all metrics
        fig = plt.figure(figsize=(15, 12))
        gs = GridSpec(3, 2, figure=fig)
        
        # Volatility
        ax1 = fig.add_subplot(gs[0, 0])
        sns.boxplot(x="Condition", y="volatility", data=metric_dfs["volatility"], ax=ax1)
        ax1.set_title("Price Volatility Comparison")
        ax1.set_ylabel("Volatility")
        
        # Flash crashes
        ax2 = fig.add_subplot(gs[0, 1])
        sns.boxplot(x="Condition", y="flash_crashes", data=metric_dfs["flash_crashes"], ax=ax2)
        ax2.set_title("Number of Flash Crashes")
        ax2.set_ylabel("Count")
        
        # Crash duration
        ax3 = fig.add_subplot(gs[1, 0])
        sns.boxplot(x="Condition", y="crash_duration", data=metric_dfs["crash_duration"], ax=ax3)
        ax3.set_title("Flash Crash Duration")
        ax3.set_ylabel("Duration (periods)")
        
        # Spread
        ax4 = fig.add_subplot(gs[1, 1])
        sns.boxplot(x="Condition", y="avg_spread", data=metric_dfs["avg_spread"], ax=ax4)
        ax4.set_title("Average Bid-Ask Spread")
        ax4.set_ylabel("Spread")
        
        # Volume
        ax5 = fig.add_subplot(gs[2, 0])
        sns.boxplot(x="Condition", y="volume", data=metric_dfs["volume"], ax=ax5)
        ax5.set_title("Trading Volume")
        ax5.set_ylabel("Volume")
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/market_conditions_comparison.png")
        plt.close()
        
        # Create individual visualizations
        for metric, df in metric_dfs.items():
            plt.figure(figsize=(8, 6))
            sns.barplot(x="Condition", y=metric, data=df)
            plt.title(f"{metric.replace('_', ' ').title()} Comparison")
            plt.savefig(f"{self.output_dir}/{metric}_comparison.png")
            plt.close()
    
    def run_statistical_tests(self, metric_dfs):
        """
        Run statistical tests to compare conditions.
        
        Args:
            metric_dfs: Dictionary of DataFrames for each metric
        """
        from scipy import stats
        
        # Create results DataFrame
        stats_results = pd.DataFrame(columns=["Metric", "Baseline_Mean", "MM_Mean", "Difference_Pct", "P_Value", "Significant"])
        
        for metric, df in metric_dfs.items():
            # Split by condition
            baseline = df[df["Condition"] == "baseline"][metric]
            mm = df[df["Condition"] == "market_maker"][metric]
            
            # Calculate means
            baseline_mean = baseline.mean()
            mm_mean = mm.mean()
            
            # Calculate percent difference
            if baseline_mean != 0:
                diff_pct = ((mm_mean - baseline_mean) / baseline_mean) * 100
            else:
                diff_pct = np.nan
            
            # Run t-test
            t_stat, p_value = stats.ttest_ind(baseline, mm)
            
            # Check significance
            significant = p_value < 0.05
            
            # Add to results
            stats_results = stats_results.append({
                "Metric": metric,
                "Baseline_Mean": baseline_mean,
                "MM_Mean": mm_mean,
                "Difference_Pct": diff_pct,
                "P_Value": p_value,
                "Significant": significant
            }, ignore_index=True)
        
        # Save results
        stats_results.to_csv(f"{self.output_dir}/statistical_comparison.csv", index=False)
        
        # Print results
        print("\nStatistical Comparison:")
        print(stats_results)
        
        # Create visualization of differences
        plt.figure(figsize=(10, 6))
        bars = plt.bar(stats_results["Metric"], stats_results["Difference_Pct"])
        
        # Color based on significance
        for i, bar in enumerate(bars):
            if stats_results.iloc[i]["Significant"]:
                if stats_results.iloc[i]["Difference_Pct"] > 0:
                    bar.set_color("green")
                else:
                    bar.set_color("red")
            else:
                bar.set_color("gray")
        
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.title("Percent Difference: Market Maker vs. Baseline")
        plt.ylabel("Percent Difference (%)")
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/percent_differences.png")
        plt.close()
    
    def analyze_flash_crashes(self, baseline_results, mm_results):
        """
        Analyze flash crash characteristics in detail.
        
        Args:
            baseline_results: Results from baseline simulations
            mm_results: Results from market maker simulations
        """
        # Extract all flash crashes
        baseline_crashes = []
        mm_crashes = []
        
        for results, crashes_list in [(baseline_results, baseline_crashes), (mm_results, mm_crashes)]:
            for result in results:
                sim_result = result["simulations"][0] if "simulations" in result else result
                if "flash_crashes" in sim_result:
                    for crash in sim_result["flash_crashes"]:
                        crashes_list.append(crash)
        
        # Create DataFrames
        baseline_df = pd.DataFrame(baseline_crashes) if baseline_crashes else pd.DataFrame()
        mm_df = pd.DataFrame(mm_crashes) if mm_crashes else pd.DataFrame()
        
        # Add condition labels
        if not baseline_df.empty:
            baseline_df["condition"] = "baseline"
        if not mm_df.empty:
            mm_df["condition"] = "market_maker"
        
        # Combine
        combined_df = pd.concat([baseline_df, mm_df], ignore_index=True)
        
        if combined_df.empty:
            print("No flash crashes detected in either condition.")
            return
        
        # Save crash data
        combined_df.to_csv(f"{self.output_dir}/flash_crash_details.csv", index=False)
        
        # Calculate percent drop
        if "pre_crash_price" in combined_df.columns and "low_price" in combined_df.columns:
            combined_df["percent_drop"] = (1 - combined_df["low_price"] / combined_df["pre_crash_price"]) * 100
        
        # Generate visualizations
        plt.figure(figsize=(10, 6))
        if "percent_drop" in combined_df.columns:
            sns.boxplot(x="condition", y="percent_drop", data=combined_df)
            plt.title("Flash Crash Severity")
            plt.ylabel("Price Drop (%)")
            plt.savefig(f"{self.output_dir}/crash_severity.png")
            plt.close()
        
        plt.figure(figsize=(10, 6))
        if "duration" in combined_df.columns:
            sns.boxplot(x="condition", y="duration", data=combined_df)
            plt.title("Flash Crash Duration")
            plt.ylabel("Duration (periods)")
            plt.savefig(f"{self.output_dir}/crash_duration.png")
            plt.close()
        
        # Statistical comparison
        if not baseline_df.empty and not mm_df.empty:
            from scipy import stats
            
            crash_stats = {}
            
            # Compare severity
            if "percent_drop" in combined_df.columns:
                baseline_drops = baseline_df["percent_drop"].dropna()
                mm_drops = mm_df["percent_drop"].dropna()
                
                if len(baseline_drops) > 0 and len(mm_drops) > 0:
                    t_stat, p_value = stats.ttest_ind(baseline_drops, mm_drops)
                    crash_stats["severity"] = {
                        "baseline_mean": baseline_drops.mean(),
                        "mm_mean": mm_drops.mean(),
                        "diff_pct": ((mm_drops.mean() - baseline_drops.mean()) / baseline_drops.mean()) * 100 if baseline_drops.mean() != 0 else np.nan,
                        "p_value": p_value,
                        "significant": p_value < 0.05
                    }
            
            # Compare duration
            if "duration" in combined_df.columns:
                baseline_durations = baseline_df["duration"].dropna()
                mm_durations = mm_df["duration"].dropna()
                
                if len(baseline_durations) > 0 and len(mm_durations) > 0:
                    t_stat, p_value = stats.ttest_ind(baseline_durations, mm_durations)
                    crash_stats["duration"] = {
                        "baseline_mean": baseline_durations.mean(),
                        "mm_mean": mm_durations.mean(),
                        "diff_pct": ((mm_durations.mean() - baseline_durations.mean()) / baseline_durations.mean()) * 100 if baseline_durations.mean() != 0 else np.nan,
                        "p_value": p_value,
                        "significant": p_value < 0.05
                    }
            
            # Save crash stats
            with open(f"{self.output_dir}/flash_crash_stats.json", "w") as f:
                json.dump(crash_stats, f, indent=2)