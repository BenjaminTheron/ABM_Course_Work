from simulator import Simulator
from parameters import PARAMETERS
from analysis_tools import create_performance_bar_plots, stylized_facts_plots


def main():

    parameters = PARAMETERS.copy()
    simulator = Simulator(parameters)
    results = simulator.run_simulations(animate=True)

    #parameters = PARAMETERS.copy()
    #parameters["MC"] = 1
    #simulator = Simulator(
    #    parameters=parameters
    #)
    
    #results = simulator.run_simulations()
    #stylized_facts_plots(results)

if __name__ == "__main__":
    main()