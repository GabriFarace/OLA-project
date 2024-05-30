from interactions import *
from requirement4 import *
 
n_days = 80
users_per_day = [10 for i in range(n_days)] #20 users per day
lambdas = [0.5, 0.6, 0.7, 0.8] # 2 slots
budget = 200
product_cost = 0.1
valuation = 1
ucb_bidding_agent = True  #if true then the bidding agent is ucb-like, else it is multiplicative pacing
 
set_seeds(18) 
       
n_users = 1000
#valuations = [0.5, 0.6, 0.7, 0.8, 0.9]
valuations = [0.8, 0.8, 0.8]

problem_params = {"n_users" : n_users, "lambdas" : lambdas,
                 "budget" : budget, "valuations" : valuations}    
 
req = Requirement4(problem_params)
req.run()