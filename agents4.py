import numpy as np
from utils import solve_linear_program, find_rank


# Virtual class for the bidding agents
class BiddingAgent:
    def __init__(self):
        pass

    def bid(self):
        pass

    def update(self, lambdas, slot, c_t, m_t=None):
        pass

    def get_utility(self):
        return np.array(self.utility).sum()

    def get_log(self):
        return self.log_bids, self.log_slots


class UCBBiddingAgentExpert(BiddingAgent):
    def __init__(self, available_bids, valuation, budget, T):
        self.available_bids = np.linspace(0,valuation,101)
        self.valuation = valuation
        self.budget = budget
        self.T = T
        self.rho = self.budget/self.T
        self.action_t = None
        self.n_pulls = np.zeros(len(self.available_bids))
        self.t = 0

        self.utility = []
        self.log_win = np.zeros(5)
        self.log_bids = []
        self.log_slots = []
        self.average_utilities = np.zeros(len(self.available_bids))
        self.average_costs = np.zeros(len(self.available_bids))

        self.name = "UCB_classic " +str(self.valuation)
        self.color = "red"

    def get_name(self):
        return self.name

    def get_color(self):
        return self.color

    def bid(self):
        if self.budget < 1:
            self.action_t = 0

        elif self.t == 0:
            self.action_t = np.random.choice(range(len(self.available_bids)))

        else:
            ucb_utility_values = self.average_utilities  #+np.sqrt(2*np.log(self.t)/self.t)
            ucb_cost_values = self.average_costs #-np.sqrt(2*np.log(self.t)/self.t)
            gamma, fun = solve_linear_program(ucb_utility_values, ucb_cost_values, self.rho)
            self.action_t = np.random.choice(range(len(self.available_bids)), p=gamma)

        self.log_bids.append(self.available_bids[self.action_t])
        return self.available_bids[self.action_t]

    def update(self, lambdas, slot, c_t, m_t=None):
        self.t += 1
        for i, b in enumerate(self.available_bids):
            rank = find_rank(b, m_t)
            if rank != -1:
                f = (self.valuation-b)*lambdas[rank]
                c = b
            else:
                f = c = 0
            self.average_utilities[i] += (f - self.average_utilities[i]) / self.t
            self.average_costs[i] += (c - self.average_costs[i]) / self.t

        f_t = 0
        if slot != -1:
            f_t = self.valuation*lambdas[slot]-c_t

        self.utility.append(f_t)
        self.budget -= c_t
        self.log_slots.append(slot)

    def get_utility(self):
        return np.array(self.utility).sum()

    def get_log(self):
        return self.log_bids, self.log_slots


class UCBBiddingAgentExpertUpdateRho(BiddingAgent):
    def __init__(self, available_bids, valuation, budget, T):
        self.available_bids = np.linspace(0,valuation,101)
        self.valuation = valuation
        self.budget = budget
        self.T = T
        self.rho = self.budget/self.T
        self.action_t = None
        self.n_pulls = np.zeros(len(self.available_bids))
        self.t = 0

        self.utility = []
        self.log_win = np.zeros(5)
        self.log_bids = []
        self.log_slots = []
        self.average_utilities = np.zeros(len(self.available_bids))
        self.average_costs = np.zeros(len(self.available_bids))

        self.name = "UCB_rho " +str(self.valuation)
        self.color = "orange"

    def get_name(self):
        return self.name

    def get_color(self):
        return self.color

    def bid(self):
        if self.budget < 1:
            self.action_t = 0

        elif self.t == 0:
            self.action_t = np.random.choice(range(len(self.available_bids)))

        else:
            ucb_utility_values = self.average_utilities  #+np.sqrt(2*np.log(self.t)/self.t)
            ucb_cost_values = self.average_costs #-np.sqrt(2*np.log(self.t)/self.t)
            gamma, fun = solve_linear_program(ucb_utility_values, ucb_cost_values, self.rho)
            self.rho = self.budget/ (self.T-self.t)
            self.action_t = np.random.choice(range(len(self.available_bids)), p=gamma)

        self.log_bids.append(self.available_bids[self.action_t])
        return self.available_bids[self.action_t]

    def update(self, lambdas, slot, c_t, m_t=None):
        self.t += 1
        for i, b in enumerate(self.available_bids):
            rank = find_rank(b, m_t)
            if rank != -1:
                f = (self.valuation-b)*lambdas[rank]
                c = b
            else:
                f = c = 0
            self.average_utilities[i] += (f - self.average_utilities[i]) / self.t
            self.average_costs[i] += (c - self.average_costs[i]) / self.t

        f_t = 0
        if slot != -1:
            f_t = self.valuation*lambdas[slot]-c_t

        self.utility.append(f_t)
        self.budget -= c_t
        self.log_slots.append(slot)

    def get_utility(self):
        return np.array(self.utility).sum()

    def get_log(self):
        return self.log_bids, self.log_slots


class MultiplicativePacingBiddingAgent(BiddingAgent):
    def __init__(self, valuation, budget, T, learning_rate=0.1):
        self.valuation = valuation
        self.budget = budget
        self.T = T
        self.learning_rate = learning_rate
        self.rho = self.budget/self.T
        self.lmbd = 1
        self.t = 0
        self.utility = []

        self.log_bids = []
        self.log_slots = []

        self.name = "Multiplicative " +str(self.valuation)
        self.color = "blue"

    def get_name(self):
        return self.name

    def get_color(self):
        return self.color

    def bid(self):
        if self.budget < 1:
            return 0

        self.log_bids.append(self.valuation/(self.lmbd+1))
        return self.valuation/(self.lmbd+1)

    def update(self, lambdas, slot, c_t, m_t=None):
        self.lmbd = np.clip(self.lmbd-self.learning_rate*(self.rho-c_t), a_min=0, a_max=1/self.rho)
        self.budget -= c_t
        if slot != -1:
            self.utility.append(self.valuation*lambdas[slot]-c_t)
        else:
            self.utility.append(0)
        self.log_slots.append(slot)

    def get_utility(self):
        return np.array(self.utility).sum()

    def get_log(self):
        return self.log_bids, self.log_slots


class HedgeAgent:
    def __init__(self, K, learning_rate):
        self.K = K
        self.learning_rate = learning_rate
        self.weights = np.ones(K)
        self.x_t = np.ones(K)/K         # Normalized weights (uniform distribution)
        self.action_t = None
        self.t = 0

    def pull_arm(self):
        self.x_t = self.weights / self.weights.sum()
        self.action_t = np.random.choice(np.arange(self.K), p=self.x_t)
        return self.action_t

    def update(self, loss_t):
        self.weights *= np.exp(-self.learning_rate * loss_t)
        self.t += 1


class FFMultiplicativePacingBiddingAgent(BiddingAgent):
    def __init__(self, bids_set, valuation, budget, T, learning_rate=0.1):
        self.bids_set = np.linspace(0,valuation,101)
        self.K = len(self.bids_set)
        self.hedge = HedgeAgent(self.K, np.sqrt(np.log(self.K)/T))
        self.valuation = valuation
        self.budget = budget
        self.T = T
        self.learning_rate = learning_rate
        self.rho = self.budget / self.T
        self.lmbd = 1
        self.t = 0
        self.utility = []

        self.log_bids = []
        self.log_slots = []

        self.name = "FF_multiplicative " +str(self.valuation)
        self.color = "green"

    def get_name(self):
        return self.name

    def get_color(self):
        return self.color

    def bid(self):
        if self.budget < 1:
            return 0

        self.log_bids.append(self.bids_set[self.hedge.pull_arm()])
        return self.bids_set[self.hedge.pull_arm()]

    def update(self, lambdas, slot, c_t, m_t):
        f_t_full = np.zeros(len(self.bids_set))
        c_t_full = np.zeros(len(self.bids_set))
        for i,b in enumerate(self.bids_set):
            rank = find_rank(b, m_t)
            if rank != -1:
                f_t_full[i] = (self.valuation-b)*lambdas[rank]
                c_t_full[i] = b*lambdas[rank]
            else:
                f_t_full[i] = c_t_full[i] = 0

        L = f_t_full-self.lmbd*(c_t_full-self.rho)
        L_range = np.max(L)-np.min(L)
        if L_range < 1e-8:
            L_range = 1
        self.hedge.update(1+(np.min(L)-L)/L_range) # Hedge needs losses in [0,1]

        self.lmbd = np.clip(self.lmbd-self.learning_rate*(self.rho-c_t), a_min=0, a_max=1/self.rho)
        self.budget -= c_t
        self.t += 1

        if slot != -1:
            self.utility.append(self.valuation*lambdas[slot] - c_t)
        else:
            self.utility.append(0)

        self.log_slots.append(slot)

    def get_utility(self):
        return np.array(self.utility).sum()

    def get_log(self):
        return self.log_bids, self.log_slots
