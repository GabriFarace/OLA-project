# AGENTS

# Virtual class
class Agent:
    def __init__(self):
        pass

    def pull_arm(self):
        pass

    def update(self, r_t):
        pass


# Stochastic MABs

# Random
class RandomAgent(Agent):
    def __init__(self, K, T, seed):
        np.random.seed(seed)
        self.action_sequence = np.random.choice(np.arange(K), size=T)
        self.a_t = None  # Last action performed (useless for random agent)
        self.action_history = np.array([])
        self.t = 0

    def pull_arm(self):
        self.a_t = self.action_sequence[self.t]
        return self.a_t

    def update(self, r_t):
        self.t += 1
        self.action_history = np.append(self.action_history, self.a_t)


# ETC      
class ETCAgent(Agent):
    def __init__(self, K, T, T0):
        self.K = K
        self.T = T
        self.T0 = T0
        self.a_t = None
        self.avg_reward = np.zeros(K)
        self.N_pulls = np.zeros(K)
        self.t = 0

    def pull_arm(self):
        if self.t <= self.T0 * self.K:
            self.a_t = self.t % self.K
        else:
            self.a_t = np.argmax(self.avg_reward)
        return self.a_t

    def update(self, r_t):
        self.N_pulls[self.a_t] += 1
        if self.t <= self.T0 * self.K:
            self.avg_reward[self.a_t] += (r_t - self.avg_reward[self.a_t]) / self.N_pulls[self.a_t]
        self.t += 1


# UCB1      
class UCB1Agent(Agent):
    def __init__(self, K, T, range=1):
        self.K = K
        self.T = T
        self.range = range
        self.a_t = None
        self.avg_rewards = np.zeros(K)
        self.N_pulls = np.zeros(K)
        self.t = 0

    def pull_arm(self):
        if self.t < self.K:
            self.a_t = self.t
        else:
            ucbs = self.avg_rewards + self.range * np.sqrt(2 * np.log(self.T) / self.N_pulls)
            # Substituting the time horizon T with the actual time t, the algo still works and it even performs better
            self.a_t = np.argmax(ucbs)
        return self.a_t

    def update(self, r_t):
        self.N_pulls[self.a_t] += 1
        self.avg_rewards[self.a_t] += (r_t - self.avg_rewards[self.a_t]) / self.N_pulls[self.a_t]
        self.t += 1


# Thompson Sampling        
class TSAgent(Agent):
    def __init__(self, K):
        self.K = K
        self.a_t = None
        self.N_pulls = np.zeros(K)
        self.alpha, self.beta = np.ones(K), np.ones(K)

    # The following implementation is just for Bernoulli distribution
    def pull_arm(self):
        theta = np.random.beta(self.alpha, self.beta)
        self.a_t = np.argmax(theta)
        return self.a_t

    def update(self, r_t):
        self.alpha[self.a_t] += r_t
        self.beta[self.a_t] += 1 - r_t
        self.N_pulls[self.a_t] += 1


# Adversarial MABs

# Hedge
class HedgeAgent(Agent):
    def __init__(self, K, learning_rate):
        self.K = K
        self.learning_rate = learning_rate
        self.weights = np.ones(K)
        self.a_t = None
        self.x_t = np.ones(K)/K
        self.t = 0

    def pull_arm(self):
        self.x_t = self.weights/sum(self.weights)
        self.a_t = np.random.choice(np.arange(self.K), p=self.x_t)
        return self.a_t

    def update(self, l_t):
        self.weights *= np.exp(-self.learning_rate*l_t)
        self.t += 1
        

# EXP3      
class EXP3Agent(Agent):
    def __init__(self, K, learning_rate):
        self.K = K
        self.learning_rate = learning_rate
        self.weights = np.ones(K)
        self.a_t = None
        self.x_t = np.ones(K)/K
        self.N_pulls = np.zeros(K)
        self.t = 0

    def pull_arm(self):
        self.x_t = self.weights/sum(self.weights)
        self.a_t = np.random.choice(np.arange(self.K), p=self.x_t)
        return self.a_t

    def update(self, l_t):
        l_t_tilde = l_t/self.x_t[self.a_t]
        self.weights[self.a_t] *= np.exp(-self.learning_rate*l_t_tilde)
        self.N_pulls[self.a_t] += 1
        self.t += 1


# Gaussian Process

class GPUCBAgent(Agent):
    def __init__(self, T, discr=100):
        self.T = T
        self.arms = np.linspace(0, 1, discr)
        self.gp = RBF_GP(scale=2).fit()
        self.a_t = None
        self.action_hist = np.array([])
        self.reward_hist = np.array([])
        self.mu_t = np.zeros(discr)
        self.sigma_t = np.zeros(discr)
        self.gamma = lambda t: np.log(t+1)**2 
        self.beta = lambda t: 1 + 0.5*np.sqrt(2 * (self.gamma(t) + 1 + np.log(T)))
        self.N_pulls = np.zeros(discr)
        self.t = 0
    
    def pull_arm(self):
        self.mu_t, self.sigma_t = self.gp.predict(self.arms) 
        ucbs = self.mu_t + self.beta(self.t) * self.sigma_t
        self.a_t = np.argmax(ucbs)
        return self.arms[self.a_t]
    
    def update(self, r_t):
        self.N_pulls[self.a_t] += 1
        self.action_hist = np.append(self.action_hist, self.arms[self.a_t])
        self.reward_hist = np.append(self.reward_hist, r_t)
        self.gp = self.gp.fit(self.arms[self.a_t], r_t)
        self.t += 1


# BIDDERS

# Budget Pacing
class MultiplicativePacingAgent:
    def __init__(self, valuation, budget, T, eta):
        self.valuation = valuation
        self.budget = budget
        self.eta = eta
        self.T = T
        self.rho = self.budget/self.T
        self.lmbd = 1
        self.t = 0

    def bid(self):
        if self.budget < 1:
            return 0
        return self.valuation/(self.lmbd+1)
    
    def update(self, f_t, c_t):
        self.lmbd = np.clip(self.lmbd-self.eta*(self.rho-c_t), 
                            a_min=0, a_max=1/self.rho)
        self.budget -= c_t


# Full-Feedback (Hedge)
class FFMultiplicativePacingAgent:
    def __init__(self, bids_set, valuation, budget, T, eta):
        self.bids_set = bids_set
        self.K = len(bids_set)
        self.hedge = HedgeAgent(self.K, np.sqrt(np.log(self.K)/T))
        self.valuation = valuation
        self.budget = budget
        self.eta = eta
        self.T = T
        self.rho = self.budget/self.T
        self.lmbd = 1
        self.t = 0

    def bid(self):
        if self.budget < 1:
            return 0
        return self.bids_set[self.hedge.pull_arm()]
    
    def update(self, f_t, c_t, m_t):
        # update hedge
        f_t_full = np.array([(self.valuation-b)*int(b >= m_t) for b in self.bids_set])
        c_t_full = np.array([b*int(b >= m_t) for b in self.bids_set])
        L = f_t_full - self.lmbd*(c_t_full-self.rho)
        range_L = 2+(1-self.rho)/self.rho
        self.hedge.update((2-L)/range_L) # hedge needs losses in [0,1]
        # update lagrangian multiplier
        self.lmbd = np.clip(self.lmbd-self.eta*(self.rho-c_t), 
                            a_min=0, a_max=1/self.rho)
        # update budget
        self.budget -= c_t
