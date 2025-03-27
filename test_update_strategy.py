from parameters import PARAMETERS
from simulator import Simulator
from analysis_tools import create_performance_bar_plots, stylized_facts_plots
from compare_flash_crashes import compare_flash_crashes

def main():

    # run monte carlo sim without continuous strategy update
    params0 = PARAMETERS.copy()
    params0["MC"] = 10
    params0["continuous"] = False

    # run monte carlo sim with continuous strategy update
    params1 = PARAMETERS.copy()
    params1["MC"] = 10
    params1["continuous"] = True

    # run monte carlo sim without continuous and without hf traders
    params2 = PARAMETERS.copy()
    params2["MC"] = 10
    params2["NH"] = 0
    params0["continuous"] = False

    # run monte carlo sim with continuous and without hft traders
    params3 = PARAMETERS.copy()
    params3["MC"] = 10
    params3["NH"] = 0
    params2["continuous"] = True
    
    # CONSIDER: what output directories?

    simulator0 = Simulator(parameters= params0)
    results0 = simulator0.run_simulations()

    simulator1 = Simulator(parameters= params1)
    results1 = simulator1.run_simulations()

    simulator2 = Simulator(parameters= params2)
    results2 = simulator2.run_simulations()

    simulator3 = Simulator(parameters= params3)
    results3 = simulator3.run_simulations()

    stylized_facts_plots(results0, output_dir='./figures/stylized_facts0')
    stylized_facts_plots(results1, output_dir='./figures/stylized_facts1')
    stylized_facts_plots(results2, output_dir='./figures/stylized_facts2')
    stylized_facts_plots(results3, output_dir='./figures/stylized_facts3')

    result_names = ["discontinuous", "continuous"]
    simulation_results = [results0, results1]
    create_performance_bar_plots(simulation_results, result_names, output_dir='./figures/update_performance')

    result_names2 = ["discontinuous with hft", "continuous with hft", "discontinous without hft", "continuous without hft"]
    simulation_results2 = [results0, results1, results2, results3]
    compare_flash_crashes(simulation_results2, result_names2, "update_comparison")


if __name__ == "__main__":
    main()
