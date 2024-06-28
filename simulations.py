# Import the required modules
from requirement2 import *
from requirement1 import *

# Define the number of days for the simulation
n_days = 500

# Define the number of users per day
users_per_day = [10 for i in range(n_days)] #20 users per day

# Define the click-through rates (CTRs) for the company and competitors
ctrs = [0.8, 0.5, 0.9, 1] # company + competitors

# Define the lambdas for the slots
lambdas = [1, 0.9] # 2 slots

# Define the budget for the bidding agent
budget = 10000

# Define the cost of the product
product_cost = 0.1

# Define the valuation of the product
valuation = 1

# Define the type of bidding agent
ucb_bidding_agent = False  #if true then the bidding agent is ucb-like, else it is multiplicative pacing

# Define the number of trials for the simulation
n_trials = 10

# Create a dictionary with all the problem parameters
problem_params = {"n_days" : n_days, "users_per_day" : users_per_day, "ctrs" : ctrs, "lambdas" : lambdas, 
                  "ucb_bidding_agent" : ucb_bidding_agent, "budget" : budget, "product_cost" : product_cost, "valuation" : valuation}     

# Set the random seed for reproducibility
set_seeds(100)

# Create an instance of Requirement2 with the given parameters
req = Requirement2(problem_params, n_trials)

# Run the pricing simulation with a specified parameter
req.run_pricing(80) 

