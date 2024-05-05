from requirement2 import *
from requirement1 import *

n_days = 5000
users_per_day = [10 for i in range(n_days)] #20 users per day
ctrs = [1 for i in range(4)] # company + competitors
lambdas = [1 for i in range(2)] # 2 slots
budget = 10000
product_cost = 0.1
valuation = 1
ucb_bidding_agent = False  #if true then the bidding agent is ucb-like, else it is multiplicative pacing

n_trials = 20


problem_params = {"n_days" : n_days, "users_per_day" : users_per_day, "ctrs" : ctrs, "lambdas" : lambdas, 
                  "ucb_bidding_agent" : ucb_bidding_agent, "budget" : budget, "product_cost" : product_cost, "valuation" : valuation}     

set_seeds(200)
req = Requirement2(problem_params, n_trials)
req.run() 