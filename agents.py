### AGENTS ###

import numpy as np

# Base Agent class
class Agent:
    def __init__(self):
        pass

    def pull_arm(self):
        pass

    def update(self, reward_t):
        pass


# Stochastic MABs
class RandomAgent(Agent):
    def __init__(self, K, T, seed):
        np.random.seed(seed)
        self.action_sequence = np.random.choice(np.arange(K), size=T)
        self.action_t = None
        self.action_history = np.array([])
        self.t = 0
    
    def pull_arm(self):
        self.action_t = self.action_sequence[self.t]
        return self.action_t
    
    def update(self, reward_t):
        self.action_history = np.append(self.action_history, self.action_t)
        self.t += 1

class GreedyAgent:
    def __init__(self, K, T):
        self.K = K
        self.T = T
        self.action_t = None
        self.average_rewards = np.inf * np.ones(K)
        self.n_pulls = np.zeros(K)
        self.t = 0
    
    def pull_arm(self):
        self.action_t = np.argmax(self.average_rewards)
        return self.action_t

    def update(self, reward_t):
        self.n_pulls[self.action_t] += 1
        if self.n_pulls[self.action_t] == 1:
            self.average_rewards[self.action_t] = reward_t
        else:
            self.average_rewards[self.action_t] += (reward_t - self.average_rewards[self.action_t]) / self.n_pulls[self.action_t]
        self.t += 1

class ETCAgent(Agent):
    def __init__(self, K, T, T0):
        self.K = K
        self.T = T
        self.T0 = T0
        self.action_t = None
        self.average_rewards = np.zeros(K)
        self.n_pulls = np.zeros(K)
        self.t = 0
    
    def pull_arm(self):
        if self.t <= self.T0 * self.K:
            self.action_t = self.t % self.K
        else:
            self.action_t = np.argmax(self.average_rewards)
        
        return self.action_t
    
    def update(self, reward_t):
        self.n_pulls[self.action_t] += 1
        if self.t <= self.T0 * self.K:
            self.average_rewards[self.action_t] += (reward_t - self.average_rewards[self.action_t]) / self.n_pulls[self.action_t]
        self.t += 1

class UCB1Agent(Agent):
    def __init__(self, K, T, range=1, anytime=False):
        self.K = K
        self.T = T
        self.range = range
        self.anytime = anytime
        self.action_t = None
        self.average_rewards = np.zeros(K)
        self.n_pulls = np.zeros(K)
        self.t = 0

    def pull_arm(self):
        if self.t < self.K:
            self.action_t = self.t
        else:
            if self.anytime:
                ucb_values = self.average_rewards + self.range * np.sqrt(2 * np.log(self.t) / self.n_pulls)
            else:
                ucb_values = self.average_rewards + self.range * np.sqrt(2 * np.log(self.T) / self.n_pulls)
            self.action_t = np.argmax(ucb_values)
        return self.action_t
    
    def update(self, reward_t):
        self.n_pulls[self.action_t] += 1
        self.average_rewards[self.action_t] += (reward_t - self.average_rewards[self.action_t]) / self.n_pulls[self.action_t]
        self.t += 1
      
class TSAgent(Agent):
    def __init__(self, K):
        self.K = K
        self.action_t = None
        self.n_pulls = np.zeros(K)
        self.alpha, self.beta = np.ones(K), np.ones(K)

    def pull_arm(self):
        theta = np.random.beta(self.alpha, self.beta)
        self.action_t = np.argmax(theta)
        return self.action_t
    
    def update(self, reward_t):
        self.alpha[self.action_t] += reward_t
        self.beta[self.action_t] += 1 - reward_t
        self.n_pulls[self.action_t] += 1


# Adversarial MABs
class HedgeAgent(Agent):
    def __init__(self, K, learning_rate):
        self.K = K
        self.learning_rate = learning_rate
        self.weights = np.ones(K)
        self.x_t = np.ones(K) / K # normalized weights (initial uniform distribution)
        self.action_t = None
        self.t = 0

    def pull_arm(self):
        self.x_t = self.weights / self.weights.sum()
        self.action_t = np.random.choice(np.arange(self.K), p=self.x_t)
        return self.action_t

    def update(self, loss_t):
        self.weights *= np.exp(-self.learning_rate * loss_t)
        self.t += 1
    
class EXP3Agent(Agent):
    def __init__(self, K, learning_rate):
        self.K = K
        self.learning_rate = learning_rate
        self.weights = np.ones(K)
        self.x_t = np.ones(K) / K
        self.action_t = None
        self.n_pulls = np.zeros(K)
        self.t = 0

    def pull_arm(self):
        self.x_t = self.weights / self.weights.sum()
        self.action_t = np.random.choice(np.arange(self.K), p=self.x_t)
        return self.action_t
    
    def update(self, loss_t):
        loss_t_adjusted = loss_t / self.x_t[self.action_t]
        self.weights[self.action_t] *= np.exp(-self.learning_rate * loss_t_adjusted)
        self.n_pulls[self.action_t] += 1
        self.t += 1


# PRICING AGENTS

# Base Pricing Agent class
class PricingAgent:
    def __init__(self):
        pass

    def set_price(self):
        pass

    def update(self, reward_t):
        pass

# Gaussian Process based agents
class GPUCBAgent(PricingAgent):
    # From the agent's POV, the action set is [0,1]. If the actual actions are outside this
    # set, we can always perform a rescaling outside the class.

    def __init__(self, T, discretization=100, scale=1):
        self.T = T
        self.arms = np.linspace(0, 1, discretization)
        self.gp = RBFGaussianProcess(scale).fit()
        self.action_t = None
        self.action_history = np.array([])
        self.reward_history = np.array([])
        self.mu_t = np.zeros(discretization)
        self.sigma_t = np.zeros(discretization)
        self.gamma = lambda t: np.log(t + 1) ** 2
        self.beta = lambda t: 1 + 0.5 * np.sqrt(2 * (self.gamma(t) + 1 + np.log(T)))
        self.n_pulls = np.zeros(discretization)
        self.t = 0

    def set_price(self):
        self.mu_t, self.sigma_t = self.gp.predict(self.arms)
        ucbs = self.mu_t + self.beta(self.t) * self.sigma_t
        self.action_t = np.argmax(ucbs) # index of the price the agent chooses
        return self.arms[self.action_t] # returns the price the agent chooses
    
    def update(self, reward_t):
        self.n_pulls[self.action_t] += 1
        self.action_history = np.append(self.action_history, self.arms[self.action_t])
        self.reward_history = np.append(self.reward_history, reward_t)
        self.gp = self.gp.fit(self.arms[self.action_t], reward_t)
        self.t += 1

class RBFGaussianProcess:
    def __init__(self, scale=1, reg=1e-2):
        self.scale = scale 
        self.reg = reg
        self.k_xx_inv = None

    def rbf_kernel_incr_inv(self, B, C, D):
        temp = np.linalg.inv(D - C @ self.k_xx_inv @ B)
        block1 = self.k_xx_inv + self.k_xx_inv @ B @ temp @ C @ self.k_xx_inv
        block2 = - self.k_xx_inv @ B @ temp
        block3 = - temp @ C @ self.k_xx_inv
        block4 = temp
        res1 = np.concatenate((block1, block2), axis=1)
        res2 = np.concatenate((block3, block4), axis=1)
        res = np.concatenate((res1, res2), axis=0)
        return res

    def rbf_kernel(self, a, b):
        a_ = a.reshape(-1, 1)
        b_ = b.reshape(-1, 1)
        output = -1 * np.ones((a_.shape[0], b_.shape[0]))
        for i in range(a_.shape[0]):
            output[i, :] = np.power(a_[i] - b_, 2).ravel()
        return np.exp(-self.scale * output)
    
    def fit(self, x=np.array([]), y=np.array([])):
        x,y = np.array(x),np.array(y)
        if self.k_xx_inv is None:
            self.y = y.reshape(-1,1)
            self.x = x.reshape(-1,1)
            k_xx = self.rbf_kernel(self.x, self.x) + self.reg * np.eye(self.x.shape[0])
            self.k_xx_inv = np.linalg.inv(k_xx)
        else:
            B = self.rbf_kernel(self.x, x)
            self.x = np.vstack((self.x, x))
            self.y = np.vstack((self.y, y))
            self.k_xx_inv = self.rbf_kernel_incr_inv(B, B.T, np.array([1 + self.reg]))

        return self

    def predict(self, x_predict):
        k = self.rbf_kernel(x_predict, self.x)

        mu_hat = k @ self.k_xx_inv @ self.y
        sigma_hat = 1 - np.diag(k @ self.k_xx_inv @ k.T)

        return mu_hat.ravel(), sigma_hat.ravel()


# BIDDING AGENTS

# Base Bidding Agent class
class BiddingAgent:
    def __init__(self):
        pass

    def bid(self):
        pass

    def update(self, f_t, c_t, m_t=None):
        pass

class MultiplicativePacingAgent(BiddingAgent):
    def __init__(self, valuation, budget, T, learning_rate=0.1):
        self.valuation = valuation
        self.budget = budget
        self.T = T
        self.learning_rate = learning_rate
        self.rho = self.budget / self.T
        self.lmbd = 1
        self.t = 0
    
    def bid(self):
        if self.budget < 1:
            return 0
        return self.valuation / (self.lmbd + 1)
    
    def update(self, f_t, c_t, m_t=None):
        self.lmbd = np.clip(self.lmbd - self.learning_rate * (self.rho - c_t), a_min=0, a_max=1/self.rho)
        self.budget -= c_t

class FFMultiplicativePacingAgent:
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
    
    def bid(self):
        if self.budget < 1:
            return 0
        return self.bids_set[self.hedge.pull_arm()]
    
    def update(self, f_t, c_t, m_t):
        # Update the Hedge agent
        f_t_full = np.array([(self.valuation - b) * int(b >= m_t) for b in self.bids_set])
        c_t_full = np.array([b * int(b >= m_t) for b in self.bids_set])
        L = f_t_full - self.lmbd * (c_t_full - self.rho)
        L_range = 2 + (1 - self.rho) / self.rho
        self.hedge.update((2 - L) / L_range) # Hedge needs losses in [0,1]

        # Update the Lagrangian multiplier
        self.lmbd = np.clip(self.lmbd - self.learning_rate * (self.rho - c_t), a_min=0, a_max=1/self.rho)

        # Update the budget
        self.budget -= c_t 
