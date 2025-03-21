import os
import argparse
import json
from genetic_evolution import GeneticAlgorithm
from parameters import PARAMETERS

def main():
    parser = argparse.ArgumentParser(description='Run genetic evolution of market makers under varying cancellation')
    parser.add_argument('--genome_num', type=int, default=1, help='Cance')
    args = parser.parse_args()
    genome_num = args.genome_num
    output_dir = "./genome_" + str(genome_num) + "_results"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("./genomes", exist_ok=True)
    genetic_dir = os.path.join(output_dir,"evolution_results")
    file_name = "genome_" + str(genome_num) + ".json"
    genome_path = os.path.join("./genomes",file_name)

    if genome_num == 0:
        params = PARAMETERS.copy()
        params["gamma_H"] = 1

        # Initialize genetic algorithm
        ga = GeneticAlgorithm(
            params=params
        )

        best_genome = ga.run_evolution(output_dir=genetic_dir)

        genome_log = {"genome_num": genome_num, "best_genome": best_genome}

        with open(genome_path, 'w') as f:
            json.dump(genome_log, f, indent=4)

    if genome_num == 1:
        params = PARAMETERS.copy()
        params["gamma_H"] = 5

        # Initialize genetic algorithm
        ga = GeneticAlgorithm(
            params=params,
            population_size=3,
            generations=3
        )

        best_genome = ga.run_evolution(output_dir=genetic_dir)

        genome_log = {"genome_num": genome_num, "best_genome": best_genome}

        with open(genome_path, 'w') as f:
            json.dump(genome_log, f, indent=4)

    if genome_num == 2:
        params = PARAMETERS.copy()
        params["gamma_H"] = 10

        # Initialize genetic algorithm
        ga = GeneticAlgorithm(
            params=params
        )

        best_genome = ga.run_evolution(output_dir=genetic_dir)

        genome_log = {"genome_num": genome_num, "best_genome": best_genome}

        with open(genome_path, 'w') as f:
            json.dump(genome_log, f, indent=4)
    
    if genome_num == 3:
        params = PARAMETERS.copy()
        params["NH"] = 0

        # Initialize genetic algorithm
        ga = GeneticAlgorithm(
            params=params
        )

        best_genome = ga.run_evolution(output_dir=genetic_dir)

        genome_log = {"genome_num": genome_num, "best_genome": best_genome}

        with open(genome_path, 'w') as f:
            json.dump(genome_log, f, indent=4)


    
if __name__ == "__main__":
    main()