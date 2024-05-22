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
        self.utility = []

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
    

class FFMultiplicativePacingAgent(BiddingAgent):
    def __init__(self, bids_set, valuation, budget, T, learning_rate):
        self.bids_set = bids_set
        self.K = len(bids_set)
        self.hedge = HedgeAgent(self.K, np.sqrt(np.log(self.K) / T)) # learning rate from theory
        self.valuation = valuation
        self.budget = budget
        self.T = T
        self.learning_rate = learning_rate
        self.rho = self.budget / self.T
        self.lmbd = 1
        self.t = 0
        self.utility = []
    
    def bid(self):
        if self.budget < 1:
            return 0
        return self.bids_set[self.hedge.pull_arm()]
    
    def update(self, lambdas, slot, c_t, m_t):
        # Update the Hedge agent
        f_t_full = np.zeros(len(self.bids_set))
        c_t_full = np.zeros(len(self.bids_set))
        for i,b in enumerate(self.bids_set):
            rank = find_rank(b, m_t)
            if rank != -1:
                f_t_full[i] = (self.valuation-b)*lambdas[rank]
                c_t_full[i] = b*lambdas[rank]
            else:
                f_t_full[i] = c_t_full[i] = 0

        L = f_t_full - self.lmbd * (c_t_full - self.rho)
        L_range = 2 + (1 - self.rho) / self.rho
        self.hedge.update((2 - L) / L_range) # Hedge needs losses in [0,1]

        # Update the Lagrangian multiplier
        self.lmbd = np.clip(self.lmbd - self.learning_rate * (self.rho - c_t), a_min=0, a_max=1/self.rho)

        # Update the budget
        self.budget -= c_t 

        # Update utility
        if slot != -1:
            self.utility.append(self.valuation*lambdas[slot] - c_t)
        else:
            self.utility.append(0)
    
    def get_utility(self):
        return np.array(self.utility).sum()
