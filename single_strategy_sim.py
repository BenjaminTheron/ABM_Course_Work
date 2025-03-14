from simulator import Simulator

def single_strategy_simulator(trader_type, filename, num_traders, iterations):
    """
    Performs a systematic experiment of a single trader type, given a number of traders,
    a number of simulations to run for each increment and the trader type to evaluate.
    Finally, the results are written to a file with the provided filename in csv format
    """
    print(f"{trader_type} simulation beginning...")
    # Create a blank dictionary to hold the distribution of trader types
    trader_type_weights = {
        "aggressive" : 0,
        "passive" : 0,
        "momentum" : 0,
        "random" : 0,
        "fundamental_up" : 0,
        "fundamental_down" : 0
    }

    # Increase the number of a given trader type by 10% each time a block of simulations is run
    for i in range(10,110,10):
        # i is the test percentage, with the weights below using the decimal form of i
        # Update the variable trader type
        trader_type_weights[trader_type] = i/100
        remaining_percentage = (1 - i/100) / 5

        # Update the distribution of the remaining trader weights for this iteration
        for other_trader in trader_type_weights.keys():
            if other_trader != trader_type:
                trader_type_weights[other_trader] = remaining_percentage

        # At each test percentage, run 50 simulations with a given number of traders
        simulator = Simulator(iterations=iterations, num_traders=num_traders)
        results = simulator.run_simulations(trader_type_weights)

        # Stores the results in a string
        current_results = f"{i/100},{results['num_simulations']},{results['avg_total_trades']},{results["avg_trade_size"]}\n"

        # Stores the results to a provided file in csv format
        with open(filename, "a") as file:
            file.write(current_results)

    print(f"{trader_type} simulation ending...")

if __name__ == "__main__":
    num_traders = 150
    iterations = 50 # The number of simulations run for each value being examined
    # Runs the experiments for the aggressive trader strategy
    single_strategy_simulator("aggressive", "results/aggressive_results.txt", num_traders, iterations)
    # Runs the experiments for the passive trader strategy
    single_strategy_simulator("passive", "results/passive_results.txt", num_traders, iterations)
    # Runs the experiments for the momentum trader strategy
    single_strategy_simulator("momentum", "results/momentum_results.txt", num_traders, iterations)
    # Runs the experiments for the random trader strategy
    single_strategy_simulator("random", "results/random_results.txt", num_traders, iterations)
    # Runs the experiments for the fundamental_up trader strategy
    single_strategy_simulator("fundamental_up", "results/fundamental_up_results.txt", num_traders, iterations)
    # Runs the experiments for the fundamental_down trader strategy
    single_strategy_simulator("fundamental_down", "results/fundamental_down_results.txt", num_traders, iterations)
