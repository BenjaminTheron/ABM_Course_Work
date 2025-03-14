from simulator import Simulator

if __name__ == "__main__":
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
    simulator = Simulator(iterations=10, num_traders=100)
    results = simulator.run_simulations(trader_weights)
    
    # Print results
    print("Simulation completed:")
    print(f"Total simulations: {results['num_simulations']}")
    print(f"Average trades per simulation: {results['avg_total_trades']}")
    print(f"Average trade size: {results["avg_trade_size"]}")