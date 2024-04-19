import numpy as np

# Base Environment class
class Environment:
    def __init__(self):
        pass

    def round(self, action_t):
        pass

# Environments for stochastic and adversarial bandits
class BernoulliEnvironment(Environment):
    def __init__(self, probs, T, seed=None):
        np.random.seed(seed)
        self.K = len(probs)
        self.rewards = np.random.binomial(n=1, p=probs, size=(T, self.K)) # T x K matrix: rewards[t][k] is the reward of action k at time t
        self.t = 0

    def round(self, action_t):
        reward_t = self.rewards[self.t][action_t]
        self.t += 1
        return reward_t

class BinomialEnvironment(Environment):
    def __init__(self, probs, T, interval_range, seed=None):
        np.random.seed(seed)
        self.K = len(probs)
        self.rewards = np.random.binomial(n=interval_range, p=probs, size=(T, self.K))
        self.t = 0

    def round(self, action_t):
        reward_t = self.rewards[self.t][action_t]
        self.t += 1
        return reward_t

class GreedyEnvironment(Environment):
    def __init__(self, T, seed=None):
        np.random.seed(seed)
        self.probs = np.array([0.5, 0.25])
        self.K = len(self.probs)
        self.rewards = np.zeros((T, self.K))
        self.rewards[0, 0] = 0
        self.rewards[1:, 0] = np.random.binomial(n=1, p=self.probs[0], size=T-1)
        self.rewards[:, 1] = self.probs[1]
        self.t = 0
    
    def round(self, action_t):
        reward_t = self.rewards[self.t, action_t]
        self.t += 1
        return reward_t

class AdversarialExpertEnvironment(Environment):
    def __init__(self, loss_sequence):
        self.loss_sequence = loss_sequence
        self.t = 0

    def round(self):                        # no need for a specific arm, the learner observes the loss of every action each round
        loss_t = self.loss_sequence[self.t] # we return the whole loss vector
        self.t += 1
        return loss_t

class AdversarialBanditEnvironment(Environment):
    def __init__(self, loss_sequence):
        self.loss_sequence = loss_sequence
        self.t = 0
    
    def round(self, action_t):                          # we need to receive a specific arm
        loss_t = self.loss_sequence[self.t, action_t]   # we only return the loss of the chosen arm
        self.t += 1
        return loss_t

# Environment for pricing problems
class PricingEnvironment(Environment):
    def __init__(self, conversion_probability, cost):
        self.conversion_probability = conversion_probability
        self.cost = cost
    
    def round(self, price_t, n_t):
        demand_t = np.random.binomial(n=n_t, p=self.conversion_probability(price_t))
        reward_t = demand_t * (price_t - self.cost)
        return demand_t, reward_t