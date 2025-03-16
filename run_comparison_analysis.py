#!/usr/bin/env python
# run_comparison_analysis.py

"""
Script to run the comparison analysis between markets with and without market makers.
This script uses the best genome from genetic evolution to compare market conditions.
"""

import os
import argparse
from comparison_analysis import MarketAnalysis

def main():
    parser = argparse.ArgumentParser(description='Run comparison analysis of market conditions')
    parser.add_argument('--simulations', type=int, default=5, help='Number of simulations for each condition')
    parser.add_argument('--genome', type=str, default='./results/genetic/best_genomes.csv', help='Path to best genome CSV')
    parser.add_argument('--output', type=str, default='./results/comparison', help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print(f"Starting comparison analysis with {args.simulations} simulations per condition")
    
    # Initialize market analysis
    analysis = MarketAnalysis(
        num_simulations=args.simulations,
        output_dir=args.output
    )
    
    # Run analysis
    analysis.run_analysis()
    
    print(f"Analysis complete! Results saved to {args.output}")

if __name__ == "__main__":
    main()