
from simulator import Simulator
from parameters import PARAMETERS

params = PARAMETERS.copy()
params["NH"] = 0
simulator = Simulator(parameters=params)
results = simulator.run_simulations(animate=True)




run_sensitivity_analysis(param_names, param_values_list, output_dir="./sensitivity_results")




ga = GeneticAlgorithm(params, population_size=20, generations=10, mutation_rate=0.1, elite_size=2)
best_genome = ga.run_evolution(genetic_dir)
simulator = Simulator(use_market_maker=True, market_maker_params=best_genome)
simulator.run_simulations(animate=True)

