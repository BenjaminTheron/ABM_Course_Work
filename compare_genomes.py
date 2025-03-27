from analysis_tools import compare_genomes
import json
import matplotlib.pyplot as plt

def main():
    paths = ["./genomes/genome_0.json", "./genomes/genome_1.json"]
    compare_genomes(paths,"./results/genome_comparison.png")
    #paths[2] = "./genomes/genome_3.json"
    #compare_genomes(paths,"./results/genome_comparison.png")

    names = ["with hft 0", "with hft 1", "with hft 2", "without hft"]
    fitness = [15216.43, 40385.32, 567.56, 210133.81]
    fig, ax = plt.subplots()
    bar_container = ax.bar(names, fitness)
    ax.set(ylabel='Fitness', title='Best Genome Fitness')
    ax.bar_label(bar_container)

    plt.savefig("./fitnesscomparison.png")

if __name__ == "__main__":
    main()