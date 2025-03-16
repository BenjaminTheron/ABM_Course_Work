## Running the System

### Genetic Evolution

```bash
python run_genetic_evolution.py --population 20 --generations 10
```

This will evolve a population of market makers over multiple generations, producing the optimal parameters for market making. This will take over an hour.

### Market Comparison

```bash
python run_comparison_analysis.py --simulations 5
```

This compares market conditions with and without the optimized market maker, analyzing metrics like volatility, flash crashes, and bid-ask spreads.
