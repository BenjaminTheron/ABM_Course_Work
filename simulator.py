import time
import pandas as pd
from auctioneer import Auctioneer
from trader import Trader
from marketplace import MarketPlace

class Simulator:
    def __init__(self, iterations, num_traders):
        self.iterations = iterations # number of times to simulate the marketplace
        self.num_traders = num_traders # number of traders in each simulation of the marketplace
        self.performance_log = dict() # stores a log of each 

    def Run_Simulations(self):
        for x in range(0, self.iterations):
            self.performance_log[x] = self.Simulate_Marketplace()

        return self.Generate_Performance_Log()

    def Simulate_Marketplace(self):
        # Initialise the LOB (as two separate data frames)
        
        
        # Initialise the traders (an array is used to store each trader)
        # TODO: alter this to give different traders different strategies/ parameters
        traders_array = []
        for x in range(0, self.num_traders):
            trader = Trader(x, "aggresive", "full", 10_000)
            traders_array.append(trader)
        
        # Initialise the Auctioneer
        # TODO: add more auctioneers / alter parameters
        auctioneer = Auctioneer(0,0,0,0,0)
        
        # Initialise the Marketplace
        # TODO: add more markets
        marketplace = MarketPlace()
        
        # Run everything over a specified time horizon
        # All trading days start at 9:00 and close at 17:00.
        # (Every half second is treated as a minute -> simulation is run for 4 minutes)
        start_time = time.time()
        while time.time() - start_time <= 240:
            # Have the auctioneer match any orders
            # Have the traders generate and submit any new orders (only one order allowed at a time)
            
        # At the end of the simulation close out each position, paying each trader a set amount for each stock
        # Their P&L is essentially their returns over the period (budget at end - budget at start)/ budget at start
        

    def Generate_Performance_Log(self):
        """
        Stores the results of the simulation in a file
        """
        print()