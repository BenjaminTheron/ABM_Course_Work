#!/usr/bin/env python
# run_animated_simulation.py - Run simulator with command line arguments

import argparse
from simulator import Simulator

def main():
    parser = argparse.ArgumentParser(description='Run market simulation with animation')
    
    # Add all parameters from PARAMETERS dictionary
    parser.add_argument('--MC', type=int, help='Monte Carlo replications')
    parser.add_argument('--T', type=int, help='Number of trading sessions')
    parser.add_argument('--NL', type=int, help='Number of low-frequency traders')
    parser.add_argument('--NH', type=int, help='Number of high-frequency traders')
    parser.add_argument('--theta', type=float, help="LF traders' trading frequency mean")
    parser.add_argument('--theta_min', type=float, help='Min trading frequency')
    parser.add_argument('--theta_max', type=float, help='Max trading frequency')
    parser.add_argument('--alpha_c', type=float, help="Chartists' order size parameter")
    parser.add_argument('--sigma_c', type=float, help="Chartists' shock standard deviation")
    parser.add_argument('--alpha_f', type=float, help="Fundamentalists' order size parameter")
    parser.add_argument('--sigma_f', type=float, help="Fundamentalists' shock standard deviation")
    parser.add_argument('--sigma_y', type=float, help='Fundamental value shock standard deviation')
    parser.add_argument('--delta', type=float, help='Price drift parameter')
    parser.add_argument('--sigma_z', type=float, help="LF traders' price tick standard deviation")
    parser.add_argument('--zeta', type=float, help="LF traders' intensity of switching")
    parser.add_argument('--gamma_L', type=int, help="LF traders' resting order periods")
    parser.add_argument('--gamma_H', type=int, help="HF traders' resting order periods")
    parser.add_argument('--eta_min', type=float, help="HF traders' activation threshold min")
    parser.add_argument('--eta_max', type=float, help="HF traders' activation threshold max")
    parser.add_argument('--lamb', dest='lambda_param', type=float, help="Market volumes weight in HF traders' order size distribution")
    parser.add_argument('--kappa_min', type=float, help="HF traders' order price distribution support min")
    parser.add_argument('--kappa_max', type=float, help="HF traders' order price distribution support max")
    parser.add_argument('--randomise_HF', type=bool, help='Whether to randomize HF trader activation')
    
    args = parser.parse_args()
    
    # Import default parameters
    from parameters import PARAMETERS
    parameters = PARAMETERS.copy()
    
    # Update parameters with command line arguments if provided
    for key, value in vars(args).items():
        if value is not None:
            parameters[key] = value
    
    # Create simulator with parameters
    simulator = Simulator(parameters=parameters)
    
    # Run with animation enabled
    results = simulator.run_simulations(animate=True)
    
    print("Simulation completed!")
    print(f"Number of flash crashes: {results.get('avg_flash_crashes', 0)}")
    print(f"Average crash duration: {results.get('avg_crash_duration', 0)}")

if __name__ == "__main__":
    main()