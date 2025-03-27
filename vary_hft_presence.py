import json
from simulator import Simulator
from analysis_tools import create_performance_bar_plots
from parameters import PARAMETERS

def main():
    params0 = PARAMETERS.copy()
    params0["MC"] = 20
    with open("./genomes/genome_0.json)", 'r') as f:
            genome_hft = json.load(f)
    simulator0 = Simulator(params0,use_market_maker=True, market_maker_params=genome_hft)
    results0 = simulator0.run_simulations()

    params1 = PARAMETERS.copy()
    params1["NH"] = 0
    params1["MC"] = 20
    with open("./genomes/genome_0.json)", 'r') as f:
            genome_without_hft = json.load(f)
    simulator0 = Simulator(params0,use_market_maker=True, market_maker_params=genome_hft)
    results0 = simulator0.run_simulations()

    names = ["with hft", "without hft"]
    create_performance_bar_plots