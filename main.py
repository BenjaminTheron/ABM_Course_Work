from simulator import Simulator

if __name__ == "__main__":
    simulator = Simulator(iterations=5, num_traders=10)
    results = simulator.run_simulations()
    
    # Print results
    print("Simulation completed:")
    print(f"Total simulations: {results['num_simulations']}")
    print(f"Average trades per simulation: {results['avg_total_trades']}")
    print(f"Average trade size: {results["avg_trade_size"]}")