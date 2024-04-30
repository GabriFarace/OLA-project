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



def get_clairvoyant_truthful(B, my_valuation, m_t, n_users):
    ## I compute my sequence of utilities at every round
    utility = (my_valuation-m_t)*(my_valuation>=m_t)
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
        clairvoyant_payments[sorted_round_utility[i]] = m_t[sorted_round_utility[i]]
        c += m_t[sorted_round_utility[i]]
        i+=1
    return clairvoyant_bids, clairvoyant_utilities, clairvoyant_payments    

def get_clairvoyant_OPT(my_valuation, B, n_users, win_probabilities, available_bids):
      f = (my_valuation-available_bids)*win_probabilities
      ct = available_bids*win_probabilities
      rho = B/n_users
      gamma, value = solve_linear_program(f, ct, rho)
      expected_clairvoyant_utilities = [value for u in range(n_users)]
      expected_clairvoyant_bids = [sum(available_bids*gamma*win_probabilities) for u in range(n_users)]
      return expected_clairvoyant_bids, expected_clairvoyant_utilities   
         
      