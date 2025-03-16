#!/usr/bin/env python
# run_genetic_evolution.py

"""
Script to run the genetic evolution of market makers.
This script evolves a population of market makers to find the optimal parameters
for market making in the flash crash simulation.
"""

import os
import argparse
from genetic_evolution import GeneticAlgorithm

def main():
    parser = argparse.ArgumentParser(description='Run genetic evolution of market makers')
    parser.add_argument('--population', type=int, default=20, help='Population size')
    parser.add_argument('--generations', type=int, default=10, help='Number of generations')
    parser.add_argument('--mutation', type=float, default=0.1, help='Mutation rate')
    parser.add_argument('--elite', type=int, default=2, help='Number of elite genomes to keep')
    parser.add_argument('--output', type=str, default='./results/genetic', help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print(f"Starting genetic evolution with population={args.population}, generations={args.generations}")
    
    # Initialize genetic algorithm
    ga = GeneticAlgorithm(
        population_size=args.population,
        generations=args.generations,
        mutation_rate=args.mutation,
        elite_size=args.elite
    )
    
    # Run evolution
    best_genome = ga.run_evolution(output_dir=args.output)
    
    print(f"Evolution complete! Best genome: {best_genome}")
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()