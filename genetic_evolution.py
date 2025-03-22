# genetic_evolution.py

import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.cm as cm
from matplotlib.patches import Polygon
import seaborn as sns
from tqdm import tqdm
import kaleido
import plotly
import plotly.graph_objects as go

# Import simulation components
from simulator import Simulator
from market_maker import MarketMaker
from parameters import PARAMETERS

class GeneticAlgorithm:
    """
    Evolutionary algorithm to optimize market maker parameters
    """
    def __init__(self, params=PARAMETERS.copy(), population_size=20, generations=10, mutation_rate=0.1, elite_size=2):
        """
        Initialize the genetic algorithm
        
        Args:
            population_size: Number of market makers in each generation
            generations: Number of generations to evolve
            mutation_rate: Probability of mutation per gene
            elite_size: Number of best performers to keep unchanged
        """
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        
        # Genetic parameters bounds
        self.param_bounds = {
            "bid_spread_factor": (0.0001, 0.01),
            "ask_spread_factor": (0.0001, 0.01),
            "max_inventory_limit": (50, 1000),
            "hedge_ratio": (0.0, 1.0),
            "order_size_multiplier": (0.01, 0.3),
            "skew_factor": (0.0, 0.1)
        }
        
        # Track evolution history
        self.history = {
            "best_fitness": [],
            "avg_fitness": [],
            "best_genome": [],
            "population_diversity": []
        }
        
        # Simulation parameters
        self.simulation_params = params
        self.simulation_steps = self.simulation_params.get("T", 1200)
    
    def initialize_population(self):
        """Create initial random population of market makers"""
        population = []
        
        for i in range(self.population_size):
            # Generate random genome
            genome = {}
            for param, (min_val, max_val) in self.param_bounds.items():
                if param == "max_inventory_limit":
                    # Integer parameter
                    genome[param] = random.randint(min_val, max_val)
                else:
                    # Float parameter
                    genome[param] = random.uniform(min_val, max_val)
            
            population.append(genome)
        
        return population
    
    def evaluate_fitness(self, population, verbose=False):
        """
        Evaluate fitness of each genome in the population
        
        Args:
            population: List of genome dictionaries
            verbose: Whether to print progress
            
        Returns:
            List of (genome, fitness) tuples
        """
        results = []
        
        if verbose:
            population_iter = tqdm(population, desc="Evaluating population")
        else:
            population_iter = population
        
        for genome in population_iter:
            # Create simulator with market maker
            simulator = Simulator(
                iterations=1,
                num_lf_traders=self.simulation_params.get("NL", 10000),
                num_hf_traders=self.simulation_params.get("NH", 100),
                parameters=self.simulation_params,
                use_market_maker=True,
                market_maker_params=genome
            )
            
            # Run simulation
            results_data = simulator.run_simulations()
            
            # Get market maker fitness
            mm_fitness = simulator.market_maker.fitness
            
            # Store results
            results.append((genome, mm_fitness))
        
        return results
    
    def select_parents(self, population_fitness, num_parents):
        """
        Select parents for next generation using tournament selection
        
        Args:
            population_fitness: List of (genome, fitness) tuples
            num_parents: Number of parents to select
            
        Returns:
            List of selected genomes
        """
        selected_parents = []
        
        # Sort by fitness (descending)
        sorted_population = sorted(population_fitness, key=lambda x: x[1], reverse=True)
        
        # Keep elites
        elites = [genome for genome, _ in sorted_population[:self.elite_size]]
        selected_parents.extend(elites)
        
        # Tournament selection for the rest
        while len(selected_parents) < num_parents:
            # Select random individuals for tournament
            tournament_size = 3
            tournament = random.sample(sorted_population, tournament_size)
            
            # Get the fittest
            winner = max(tournament, key=lambda x: x[1])[0]
            selected_parents.append(winner)
        
        return selected_parents
    
    def crossover(self, parent1, parent2):
        """
        Perform crossover between two parents
        
        Args:
            parent1: First parent genome
            parent2: Second parent genome
            
        Returns:
            New genome from crossover
        """
        child = {}
        
        # Randomly choose genes from each parent
        for param in self.param_bounds.keys():
            # 50% chance to inherit from each parent
            if random.random() < 0.5:
                child[param] = parent1[param]
            else:
                child[param] = parent2[param]
        
        return child
    
    def mutate(self, genome):
        """
        Mutate a genome
        
        Args:
            genome: The genome to mutate
            
        Returns:
            Mutated genome
        """
        mutated = genome.copy()
        
        for param, (min_val, max_val) in self.param_bounds.items():
            # Each parameter has a chance to mutate
            if random.random() < self.mutation_rate:
                if param == "max_inventory_limit":
                    # Integer parameter
                    mutated[param] = random.randint(min_val, max_val)
                else:
                    # Float parameter - gaussian mutation
                    current_val = mutated[param]
                    range_size = max_val - min_val
                    mutation_strength = range_size * 0.2  # 20% of range
                    
                    # Apply gaussian mutation
                    new_val = current_val + random.gauss(0, mutation_strength)
                    
                    # Clamp to valid range
                    mutated[param] = max(min_val, min(max_val, new_val))
        
        return mutated
    
    def create_next_generation(self, population_fitness):
        """
        Create the next generation using selection, crossover and mutation
        
        Args:
            population_fitness: List of (genome, fitness) tuples
            
        Returns:
            Next generation population
        """
        # Select parents
        parents = self.select_parents(population_fitness, self.population_size)
        
        # Create new generation
        next_generation = []
        
        # Add elites directly
        sorted_population = sorted(population_fitness, key=lambda x: x[1], reverse=True)
        elites = [genome for genome, _ in sorted_population[:self.elite_size]]
        next_generation.extend(elites)
        
        # Fill the rest with crossover and mutation
        while len(next_generation) < self.population_size:
            # Select two parents for crossover
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            
            # Perform crossover and mutation
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            
            next_generation.append(child)
        
        return next_generation
    
    def calculate_diversity(self, population):
        """
        Calculate population diversity
        
        Args:
            population: List of genomes
            
        Returns:
            Diversity score (average pairwise distance)
        """
        if len(population) <= 1:
            return 0
        
        # Normalize genomes
        normalized_genomes = []
        for genome in population:
            norm_genome = {}
            for param, (min_val, max_val) in self.param_bounds.items():
                # Normalize to [0, 1]
                norm_genome[param] = (genome[param] - min_val) / (max_val - min_val)
            normalized_genomes.append(norm_genome)
        
        # Calculate average pairwise distance
        total_distance = 0
        num_pairs = 0
        
        for i in range(len(normalized_genomes)):
            for j in range(i+1, len(normalized_genomes)):
                # Euclidean distance
                dist = 0
                for param in self.param_bounds.keys():
                    dist += (normalized_genomes[i][param] - normalized_genomes[j][param])**2
                
                dist = np.sqrt(dist)
                total_distance += dist
                num_pairs += 1
        
        avg_distance = total_distance / num_pairs if num_pairs > 0 else 0
        return avg_distance
    
    def run_evolution(self, output_dir="./results/genetic"):
        """
        Run the evolutionary algorithm
        
        Args:
            output_dir: Directory to save results
            
        Returns:
            Best genome found
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize population
        population = self.initialize_population()
        
        # Evolution loop
        for generation in range(self.generations):
            print(f"Generation {generation+1}/{self.generations}")
            
            # Evaluate fitness
            population_fitness = self.evaluate_fitness(population, verbose=True)
            
            # Get statistics
            fitnesses = [fitness for _, fitness in population_fitness]
            avg_fitness = np.mean(fitnesses)
            best_fitness = max(fitnesses)
            best_genome = population_fitness[fitnesses.index(best_fitness)][0]
            
            diversity = self.calculate_diversity(population)
            
            # Update history
            self.history["best_fitness"].append(best_fitness)
            self.history["avg_fitness"].append(avg_fitness)
            self.history["best_genome"].append(best_genome)
            self.history["population_diversity"].append(diversity)
            
            print(f"Best fitness: {best_fitness:.2f}, Avg fitness: {avg_fitness:.2f}")
            print(f"Best genome: {best_genome}")
            print(f"Population diversity: {diversity:.4f}")
            
            # Create next generation (except for last generation)
            if generation < self.generations - 1:
                population = self.create_next_generation(population_fitness)
            
            # Save checkpoints
            self.save_results(output_dir, generation)
        
        # Return best genome from final generation
        return self.history["best_genome"][-1]
    
    def save_results(self, output_dir, generation):
        """
        Save evolution results and visualizations
        
        Args:
            output_dir: Directory to save results
            generation: Current generation number
        """
        # Save history to CSV
        history_df = pd.DataFrame({
            "generation": list(range(generation + 1)),
            "best_fitness": self.history["best_fitness"],
            "avg_fitness": self.history["avg_fitness"],
            "diversity": self.history["population_diversity"]
        })
        
        history_df.to_csv(f"{output_dir}/evolution_history.csv", index=False)
        
        # Save best genomes
        best_genomes_df = pd.DataFrame(self.history["best_genome"])
        best_genomes_df["generation"] = list(range(generation + 1))
        best_genomes_df.to_csv(f"{output_dir}/best_genomes.csv", index=False)
        
        # Generate and save visualizations
        self.generate_visualizations(output_dir)
    
    def generate_visualizations(self, output_dir):
        """
        Generate visualizations of the evolution process
        
        Args:
            output_dir: Directory to save visualizations
        """
        # Plot fitness history
        plt.figure(figsize=(10, 6))
        plt.plot(self.history["best_fitness"], label="Best Fitness")
        plt.plot(self.history["avg_fitness"], label="Average Fitness")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title("Evolution of Fitness")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_dir}/fitness_history.png")
        plt.close()
        
        # Plot diversity
        plt.figure(figsize=(10, 6))
        plt.plot(self.history["population_diversity"])
        plt.xlabel("Generation")
        plt.ylabel("Diversity (avg pairwise distance)")
        plt.title("Population Diversity")
        plt.grid(True)
        plt.savefig(f"{output_dir}/diversity_history.png")
        plt.close()
        
        # Plot evolution of parameters
        if self.history["best_genome"]:
            param_history = {param: [] for param in self.param_bounds.keys()}
            
            for genome in self.history["best_genome"]:
                for param in self.param_bounds.keys():
                    param_history[param].append(genome[param])
            
            plt.figure(figsize=(12, 8))
            for param, values in param_history.items():
                plt.plot(values, label=param)
            
            plt.xlabel("Generation")
            plt.ylabel("Parameter Value")
            plt.title("Evolution of Best Genome Parameters")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{output_dir}/parameter_evolution.png")
            plt.close()
            
            # Generate radar plot for best genome
            if len(self.history["best_genome"]) > 0:
                best_genome = self.history["best_genome"][-1]
                self.plot_genome_radar(best_genome, f"{output_dir}/best_genome_radar.png")
    
    def plot_genome_radar(self, genome, output_path):
        """
        Create a radar plot of a genome using Plotly
        
        Args:
            genome: The genome to visualize
            output_path: Where to save the plot
        """
        # Normalize genome values to [0, 1]
        normalized_genome = {}
        for param, (min_val, max_val) in self.param_bounds.items():
            normalized_genome[param] = (genome[param] - min_val) / (max_val - min_val)
        
        # Setup radar plot
        params = list(self.param_bounds.keys())
        
        # Format the actual values for display
        actual_values = []
        for param in params:
            value = genome[param]
            if param == "max_inventory_limit":
                actual_values.append(f"{value:.0f}")
            else:
                actual_values.append(f"{value:.4f}")
        
        # Create the radar chart using Plotly
        fig = go.Figure()
        
        # Add the radar trace
        fig.add_trace(go.Scatterpolar(
            r=[normalized_genome[param] for param in params] + [normalized_genome[params[0]]],  # Close the loop
            theta=params + [params[0]],  # Close the loop
            fill='toself',
            name='Normalized Values',
            line=dict(color='rgba(32, 128, 255, 0.8)', width=2),
            fillcolor='rgba(32, 128, 255, 0.2)'
        ))
        
        # Add markers for each parameter with the actual value as hover text
        fig.add_trace(go.Scatterpolar(
            r=[normalized_genome[param] for param in params],
            theta=params,
            mode='markers+text',
            name='Parameters',
            marker=dict(color='rgba(32, 128, 255, 1)', size=10),
            text=actual_values,
            textposition="top center",
            hoverinfo="text",
            hovertext=[f"{param}: {value}" for param, value in zip(params, actual_values)]
        ))
        
        # Configure the layout
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickfont=dict(size=10)
                ),
                angularaxis=dict(
                    tickfont=dict(size=12, color='rgb(50, 50, 50)'),
                )
            ),
            showlegend=False,
            title={
                'text': "Market Maker Genome",
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=20)
            },
            margin=dict(l=80, r=80, t=100, b=80),
            height=700,
            width=700,
        )
        
        # Save the figure as PNG
        fig.write_image(output_path)