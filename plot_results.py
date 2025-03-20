import matplotlib.pyplot as plt
import numpy as np
""" Plot the results of each of the run experiments """

# Stores the name of each file holding experiment results
files = ['aggressive_results.txt', 'passive_results.txt',
        'momentum_results.txt', 'random_results.txt',
        'fundamental_up_results.txt', 'fundamental_down_results.txt']

save_files = ['plots/agro_plot', 'plots/passive_plot',
             'plots/momentum_plot', 'plots/random_plot',
             'plots/f_up_plot', 'plots/f_down_plot']

trader_types = ['aggressive', 'passive', 'momentum', 'random', 'fundamental_up', 'fundamental_down']

# Tracks the wider trader results being examined
current_trader = 0 # 0 = aggressive, 1 = passive, 2 = momentum, 3 = random, 4 = fundamental_up, 5 = fundamental_down
# Tracks the trader being examined locally within the results file
local_trader = 0

for file in files:
    # Stores the pnl, trade size and number of trades for the given trader type
    trader_pnl_results = []
    trader_tsize_results = []
    trader_num_trades_results = []
    # Stores the average pnl, trade size and number of trades for the other trader types
    other_trader_pnl_results = {
        'aggressive': [],
        'passive' : [],
        'momentum' : [],
        'random': [],
        'fundamental_up': [],
        'fundamental_down': []
    }
    other_trader_tsize_results = {
        'aggressive': [],
        'passive' : [],
        'momentum' : [],
        'random': [],
        'fundamental_up': [],
        'fundamental_down': []
    }
    other_trader_num_trades_results = {
        'aggressive': [],
        'passive' : [],
        'momentum' : [],
        'random': [],
        'fundamental_up': [],
        'fundamental_down': []
    }

    # Stores the average high and low market price data across the simulation
    average_market_highs = []
    average_market_lows = []
    average_market_difference = []

    with open('results/' + file, 'r') as f:
        # Iterate over each line in the file
        for line in f:
            current_line = line.strip()
            # Converts the current line into a list of values separated by commas
            current_line = current_line.split(',')

            # Add the market wide data for the simulation at the current proportion
            # Validate we are looking at the market data by checking that the number of simulations is 100
            if current_line[1] == '100':
                average_market_highs.append(current_line[4])
                average_market_lows.append(current_line[5])
                average_market_difference.append(current_line[6])

            # Check which trader type (if any) we are currently looking at
            if current_line[0] in trader_types:
                # All results files follow the same order 
                if current_trader == local_trader:
                    # Extracts the pnl, trade size and number of trades values
                    trader_pnl_results.append(current_line[1])
                    trader_tsize_results.append(current_line[2])
                    trader_num_trades_results.append(current_line[3])
                else:
                    # Extracts the pnl, trade size and number of trade values for the other traders
                    other_trader_pnl_results[current_line[0]].append(current_line[1])
                    other_trader_tsize_results[current_line[0]].append(current_line[2])
                    other_trader_num_trades_results[current_line[0]].append(current_line[3])

                # Move to the next trader
                local_trader += 1
                # After the fundamental_up trader, we loop back to the aggressive trader
                if local_trader > 5:
                    local_trader = 0

    # Array to store the x-axis values -> proportion of population abiding by strategy
    proportion_array = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    # An array of arrays to store the pnl data of every type in one location
    pnl_data = np.zeros((len(trader_types), 10))
    # An array of arrays to store the trade size data for every type
    tsize_data = np.zeros((len(trader_types), 10))
    # An array of arrays to store the number of trades data for every type
    num_trades_data = np.zeros((len(trader_types), 10))

    for i in range(0,len(trader_types)):
        if i == current_trader:
            # If we're looking at the investigated trading strategy
            pnl_data[i] = trader_pnl_results
            tsize_data[i] = trader_tsize_results
            num_trades_data[i] = trader_num_trades_results
        else:
            # Otherwise add the pnl data for the other trading strategies
            pnl_data[i] = other_trader_pnl_results[trader_types[i]]
            tsize_data[i] = other_trader_tsize_results[trader_types[i]]
            num_trades_data[i] = other_trader_num_trades_results[trader_types[i]]

    # Plot the results for the PnL file
    plt.figure(figsize=(12,10))
    plt.title(f"PnL Performance of the {trader_types[current_trader]} Strategy:")
    plt.xlabel("Proportion of the Population Trading this way (%)")
    plt.ylabel("PnL ($)")
    plt.plot(proportion_array, pnl_data.T, label=trader_types)
    plt.legend()
    plt.savefig(save_files[current_trader] + '_pnl.pdf')
    
    # Plot the results for the trade size file
    plt.figure(figsize=(12,10))
    plt.title(f"Average Trade Sizes Made by the {trader_types[current_trader]} Strategy:")
    plt.xlabel("Proportion of the Population Trading this way (%)")
    plt.ylabel("Trade Size")
    plt.plot(proportion_array, tsize_data.T, label=trader_types)
    plt.legend()
    plt.savefig(save_files[current_trader] + '_tsize.pdf')

    # Plot the results for the number of trades file
    plt.figure(figsize=(12,10))
    plt.title(f"Average Number of Trades Made by the {trader_types[current_trader]} Strategy:")
    plt.xlabel("Proportion of the Population Trading this way (%)")
    plt.ylabel("Average Number of Trades")
    plt.plot(proportion_array, num_trades_data.T, label=trader_types)
    plt.legend()
    plt.savefig(save_files[current_trader] + '_num_trades.pdf')

    # Plot the average highest price reached across the simulation at each proportion
    plt.figure(figsize=(12,10))
    plt.title(f"Average Market High Reached with the {trader_types[current_trader]} Strategy:")
    plt.xlabel("Proportion of the Population Trading this way (%)")
    plt.ylabel("Market Price ($)")
    plt.plot(proportion_array, average_market_highs)
    plt.savefig(save_files[current_trader] + '_market_high.pdf')
    # Plot the average minimum price reached across the simulation at each proportion
    plt.figure(figsize=(12,10))
    plt.title(f"Average Market Low Reached with the {trader_types[current_trader]} Strategy:")
    plt.xlabel("Proportion of the Population Trading this way (%)")
    plt.ylabel("Market Price ($)")
    plt.plot(proportion_array, average_market_lows)
    plt.savefig(save_files[current_trader] + '_market_low.pdf')
    # Plot the average Start to Close price difference across the simulation
    plt.figure(figsize=(12,10))
    plt.title(f"Average Open-Close Difference Reached with the {trader_types[current_trader]} Strategy:")
    plt.xlabel("Proportion of the Population Trading this way (%)")
    plt.ylabel("Price Difference ($)")
    plt.plot(proportion_array, average_market_difference)
    plt.savefig(save_files[current_trader] + '_price_difference.pdf')

    current_trader += 1

# Plots the baseline values
# The x axis is the trader type
average_type_pnl = []
average_type_tsize = []
average_type_num_trades = []
market_data = []

with open('results/baseline_results.txt', 'r') as file:
    current_trader = 0
    for line in file:
        current_line = line.strip()
        # Converts the current line into a list of values separated by commas
        current_line = current_line.split(',')

        # Validate whether you're looking at the market data or trader type data
        if current_line[1] == '100':
            # Extract the relevant market data
            market_data.append(float(current_line[4]))
            market_data.append(float(current_line[5]))
            market_data.append(float(current_line[6]))
        elif current_line[0] in trader_types:
            # Extract the relevant type specific data
            average_type_pnl.append(float(current_line[1]))
            average_type_tsize.append(float(current_line[2]))
            average_type_num_trades.append(float(current_line[3]))
        
        current_trader += 1

# The y-value is the corresponding, pnl, trade size and number of trades submitted by the given trader type
# PnL plot
plt.figure(figsize=(12,10))
plt.title(f"The Baseline PnL of Each Trading Strategy")
plt.xlabel("Trader Type")
plt.ylabel("PnL ($)")
plt.plot(trader_types, average_type_pnl)
plt.savefig('plots/baseline_pnl.pdf')
# Trade size plot
plt.figure(figsize=(12,10))
plt.title(f"The Baseline Average Trade Size of Each Trading Strategy")
plt.xlabel("Trader Type")
plt.ylabel("Trade Size")
plt.plot(trader_types, average_type_tsize)
plt.savefig('plots/baseline_trade_size.pdf')
# Num trades plot
plt.figure(figsize=(12,10))
plt.title(f"The Baseline Average Number of Trades Made by Each Trading Strategy")
plt.xlabel("Trader Type")
plt.ylabel("Number of Trades")
plt.plot(trader_types, average_type_num_trades)
plt.savefig('plots/baseline_trade_size.pdf')

# For the market values, there is only a single value for each so all are plotted on the same graph

# The x-axis is the market value
x_values = ['Market High', 'Market Low', 'Start-Close Price Difference']
plt.figure(figsize=(12,10))
plt.title(f"Baseline Market Data")
plt.xlabel("Market Data Point")
plt.ylabel("Price ($)")
plt.plot(x_values, market_data)
plt.savefig('plots/baseline_market_values.pdf')

# Conclusions to draw -> which distribution of strategies had the highest highs, which had the highest average PnL, which had the highest lows?
# Severe limitations with each of the strategies to discuss.