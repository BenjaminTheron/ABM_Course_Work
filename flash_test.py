from simulator import Simulator
from parameters import PARAMETERS
from analysis_tools import create_performance_bar_plots


def main():
    results_names = ["test","testy"]
    simulator = Simulator(
        iterations=1,
        num_lf_traders=10000,
        num_hf_traders=100,
        parameters=PARAMETERS.copy(),
        use_market_maker=False
    )
    results0 = simulator.run_simulations()

    simulator1 = Simulator(
        iterations=1,
        num_lf_traders=10000,
        num_hf_traders=100,
        parameters=PARAMETERS.copy(),
        use_market_maker=False
    )
    results1 = simulator1.run_simulations()

    results = [results0,results1]
    
    create_performance_bar_plots(results, results_names, output_dir='./figures')
    print(results["avg_flash_crashes"])

if __name__ == "__main__":
    main()