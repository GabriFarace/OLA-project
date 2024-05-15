import numpy as np
from utils import *
 
def find_rank(bid, m_t):
    for i in range(len(m_t)):
        if bid >= m_t[i]:
            return i
    return -1
       
 
class BiddingAgent:
    def __init__(self):
        pass
 
    def bid(self):
       pass
 
    def update(self, lambdas, slot, c_t, m_t=None):
        pass

    def get_utility(self):
        return np.array(self.utility).sum()
 
class UCBBiddingAgentExpert(BiddingAgent):
    def __init__(self, valuation, available_bids, budget, T):
        self.available_bids = np.linspace(0,valuation,11)
        self.valuation = valuation
        self.budget = budget
        self.T = T
       
        self.rho = self.budget/self.T
        self.action_t = None
        self.average_utilities = np.zeros(len(available_bids))
        self.average_costs = np.zeros(len(available_bids))
        self.n_pulls = np.zeros(len(available_bids))
        self.t = 0
        self.utility = []
        self.log_win = np.zeros(5)
 
    def bid(self):
        if self.budget < 1:
            self.action_t = 0
           
        elif self.t == 0:
            self.action_t = np.random.choice(range(len(self.available_bids)))
           
        else:
            ucb_utility_values = self.average_utilities + np.sqrt(2 * np.log(self.t) / self.t)
            ucb_cost_values = self.average_costs - np.sqrt(2 * np.log(self.t) / self.t)
            gamma, fun = solve_linear_program(ucb_utility_values, ucb_cost_values, self.rho)            
            # print(self.t)
            # print(gamma)
            # print(ucb_utility_values)
            # print(ucb_cost_values)
            # print()
            self.action_t = np.random.choice(range(len(self.available_bids)), p=gamma)
            
        return self.available_bids[self.action_t]
 
    def update(self, lambdas, slot, c_t, m_t=None):
       
        # Update bidding strategy
        self.t += 1
        #print(self.valuation, self.t)
        for i, b in enumerate(self.available_bids):
            rank = find_rank(b, m_t)
            if rank != -1:
                f = (self.valuation-b)*lambdas[rank]
                c = b
            else:
                f = c = 0
            #print(rank, f, c)
            self.average_utilities[i] += (f - self.average_utilities[i]) / self.t
            self.average_costs[i] += (c - self.average_costs[i]) / self.t
        self.log_win[slot] += 1
        #print()
        # Update bidder status
        f_t = 0
        if slot != -1:
            f_t = self.valuation*lambdas[slot]-c_t
        #print(f_t, lambdas[slot], c_t )
        self.utility.append(f_t)
        self.budget -= c_t

    def get_utility(self):
        return np.array(self.utility).sum()
    
class MultiplicativePacingAgent(BiddingAgent):
    def __init__(self, valuation, budget, T, learning_rate=0.1):
        self.valuation = valuation
        self.budget = budget
        self.T = T
        self.learning_rate = learning_rate
        self.rho = self.budget / self.T
        self.lmbd = 1
        self.t = 0
        self.utility = 0
    
    def bid(self):
        if self.budget < 1:
            return 0
        return self.valuation / (self.lmbd + 1)
    
    def update(self, lambdas, slot, c_t, m_t=None):
        self.lmbd = np.clip(self.lmbd - self.learning_rate * (self.rho - c_t), a_min=0, a_max=1/self.rho)
        self.budget -= c_t
        if slot != -1:
            self.utility.append(self.valuation*lambdas[slot] - c_t)
        else:
            self.utility.append(0)
        
    
    def get_utility(self):
        return np.array(self.utility).sum()