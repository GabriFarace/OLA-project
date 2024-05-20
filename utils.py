import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy import stats

def set_seeds(seed):
    """
    Sets the random seeds for reproducibility.

    Parameters:
        - seed : int
            The seed to set for reproducibility.
    """

    np.random.seed(seed)

    
def plot_regret(all_cumulative_regrets, label, T, n_epochs):
    average_cumulative_regret = all_cumulative_regrets.mean(axis=0)
    cumulative_regret_std = all_cumulative_regrets.std(axis=0)
    
    plt.plot(np.arange(T), average_cumulative_regret, label=label)
    plt.fill_between(np.arange(T),
                    average_cumulative_regret-cumulative_regret_std/np.sqrt(n_epochs),
                    average_cumulative_regret+cumulative_regret_std/np.sqrt(n_epochs),
                    alpha=0.3)
    plt.legend()
    
    plt.show()
    
def solve_linear_program(f, ct, rho):
    c = -f
    A_ub = [ct]
    b_ub = [rho]
    A_eq = [np.ones(len(f))]
    b_eq = [1]
    res = optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0,1))
    return res.x, -res.fun



def get_clairvoyant_truthful(B, my_valuation, lam, m_t, n_users):
    ## I compute my sequence of utilities at every round
    payment = m_t*lam
    utility = ((my_valuation*lam) - payment)*(my_valuation>=m_t)
    ## Now I have to find the sequence of m_t summing up to budget B and having the maximum sum of utility
    ## In second price auctions, I can find the sequence **greedily**:
    sorted_round_utility = np.flip(np.argsort(utility)) # sorted rounds, from most profitable to less profitable
    clairvoyant_utilities = np.zeros(n_users)
    clairvoyant_bids= np.zeros(n_users)
    clairvoyant_payments = np.zeros(n_users)
    c = 0
    i = 0
    while c <= B-1 and i < n_users:
        clairvoyant_bids[sorted_round_utility[i]] = my_valuation
        clairvoyant_utilities[sorted_round_utility[i]] = utility[sorted_round_utility[i]]
        clairvoyant_payments[sorted_round_utility[i]] = payment[sorted_round_utility[i]]
        c += payment[sorted_round_utility[i]]
        i+=1
    return clairvoyant_bids, clairvoyant_utilities, clairvoyant_payments    

def get_clairvoyant_OPT(my_valuation, ctr, B, n_users, m_t, available_bids, lambdas):
    
      my_valuations = np.array([])
      lambdas_b = np.array([])
      for b in available_bids:
          lambda_b = 0
          count = 0
          y = np.count_nonzero(b*ctr > m_t[0])
          for i in range(len(lambdas)):
              x = np.count_nonzero(b*ctr >= m_t[i]);
              if(i != 0):
                 x -= y
                 y = x
              count += x
              lambda_b += x*lambdas[i]
          if (count != 0):    
              lambda_b = lambda_b/count  
          my_valuations = np.append(my_valuations, my_valuation*lambda_b*ctr)
          lambdas_b = np.append(lambdas_b, b*lambda_b*ctr)
          
      win_probabilities = np.array([sum(b*ctr >= m_t[len(lambdas) - 1])/n_users for b in available_bids])
      f = (my_valuations-lambdas_b)*win_probabilities
      ct = lambdas_b*win_probabilities
      rho = B/n_users
      gamma, value = solve_linear_program(f, ct, rho)
      expected_clairvoyant_utilities = [value for u in range(n_users)]
      expected_clairvoyant_bids = [sum(available_bids*gamma*win_probabilities) for u in range(n_users)]
      return expected_clairvoyant_bids, expected_clairvoyant_utilities   
         



def get_bidding_adversarial_sequence(n_competitors, n_users, alpha, beta):
    bids = lambda n_competitors, a, b: np.random.uniform(a, b, n_competitors)
    a_s = np.random.beta(alpha, beta, n_users)/2
    b_s = 1 - np.random.beta(alpha, beta, n_users)/2
    
    bids_sequence = [bids(n_competitors, a_s[i], b_s[i]) for i in range(n_users)]
    bids_sequence = np.array(bids_sequence)

    return bids_sequence

def get_pricing_adversarial_sequence(prices, max_visits, n_days, alpha, beta):
    conversion_probability = lambda p, theta: 1-theta*p
    thetas = np.random.beta(alpha, beta, n_days)
    
    demands_t_p_n = [[[
        np.random.binomial(n=i+1, p=conversion_probability(price, thetas[k])) 
        for i in range(max_visits)] 
        for price in  prices] 
        for k in range(n_days)]
    return demands_t_p_n    
        

'''plt.plot(m_t)
plt.xlabel('$t$')
plt.ylabel('$m_t$')
plt.title('Sequence of m_t')
plt.show()

plt.plot(bidding_agent_utilities)
plt.xlabel('$t$')
plt.ylabel('$f_t$')
plt.title('Bidding Agent Utilies')
plt.show()

plt.plot(self.bidding_agent_bids)
plt.xlabel('$t$')
plt.ylabel('$b_t$')
plt.title('Agent Bids')
plt.legend()
plt.show()  

plt.plot(np.cumsum(self.bidding_agent_payments))
plt.xlabel('$t$')
plt.ylabel('$\sum c_t$')
plt.legend()
plt.title('Cumulative Payments of bidding Agent')
plt.show()'''    