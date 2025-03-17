from simulator import Simulator, Simple_Simulator

if __name__ == "__main__":
    num_of_traders = 50
    # For this simulation an equal proportion of traders is picked
    # The weights/ likelihood for a trader to act a given way
    trader_weights = {
        "aggressive": 1/6,
        "passive": 1/6,
        "momentum": 1/6,
        "random": 1/6,
        "fundamental_up": 1/6,
        "fundamental_down": 1/6,
    }
    
    # Create and run a simulation with 10 iterations and 20 traders
    simulator = Simple_Simulator(iterations=10, num_traders=num_of_traders)
    results = simulator.run_simulations(trader_weights, 100)

    # Output the results of the simulation
    print("Simulation completed:")
    print(f"Total simulations ran: {results["num_simulations"]}")
    # Outputs the overall market results
    print(f"Average trades executed per simulation: {results["avg_total_trades"]}")
    print(f"Average trade size per simulation: {results["avg_trade_size"]}")
    print(f"Average market high: ${results["avg_highest_price"]}")
    print(f"Average market low: ${results["avg_lowest_price"]}")
    print(f"Average price change over the simulation: ${results["avg_price_change"]}\n")

    # Outputs the trader specific results
    for trade_type in trader_weights.keys():
        print("For the", trade_type, "strategy:")
        print(f"The average PnL per simulation: {results[trade_type]["avg_pnl"]}")
        print(f"The average trade size per simulation: {results[trade_type]["avg_trade_size"]}")
        print(f"The average trade volume per simulation: {results[trade_type]["avg_num_trades"]}")
        print(f"The average ratio of bid to ask trades: {results[trade_type]["avg_bid/ask_ratio"]}")
        print(f"The average number of traders using the strategy: {results[trade_type]["avg_traders"]}")

    # Print results
    print("Simulation completed:")
    print(f"Total simulations: {results['num_simulations']}")
    print(f"Average trades per simulation: {results['avg_total_trades']}")
    print(f"Average trade size: {results["avg_trade_size"]}")