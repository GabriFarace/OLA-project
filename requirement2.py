import numpy as np
from auctions import *
from environments import *
from agents import *
from utils import *
from scipy import stats
from interactions import *

''' Requirement classes '''       
class Requirement2:        
    def __init__(self, params, n_trials):
        # Parameters of the problem
        self.n_days = params["n_days"]
        self.users_per_day = params["users_per_day"]
        self.ctrs = params["ctrs"]
        self.lambdas = params["lambdas"]
        self.valuation = params["valuation"]
        self.product_cost = params["product_cost"]
        self.budget = params["budget"]
        self.n_trials = n_trials
        
    def run(self):
        # Discretized prices 
        K = int(1/(self.n_days**(-1/3)))  # discretization as prescribed by the theory
        min_price, max_price = 0, 1
        prices = np.linspace(min_price, max_price, K)
        
        # The first ctr is the one of the company
        n_competitors = len(self.ctrs)-1
        
        # Number of users for  the bidding algorithm accross all days
        n_users = np.sum(self.users_per_day)  
        
        # Discretized bids for the adversarial problem
        available_bids = np.linspace(0,1,int(1/(n_users**(-1/3))))
    
        
        ''' DEFINE THE ADVERSARIAL COMPONENTS'''
        # Define the sequence of demands for each day, price and number of visits demands_t_p_n
        max_visits = np.max(self.users_per_day)
        demands_t_p_n = get_pricing_adversarial_sequence(prices, max_visits, self.n_days, 1, 1)
        
        # Define the adversarial sequence of bids of the competitors
        bids_sequence = get_bidding_adversarial_sequence(n_competitors, n_users, 1, 1)
        
        
        ''' COMPUTE THE TWO CLAIRVOYANTS'''
        
        # Compute the pricing clairvoyant

        reward_function = lambda price, n_sales: (price-self.product_cost)*n_sales
        
        
        # Compute the bidding clairvoyant

                  
        # Compute it by solving the linear program 
        m_t = bids_sequence.max(axis=1)
        win_probabilities = np.array([sum(b > m_t)/n_users for b in available_bids])
        expected_bidding_clairvoyant_bids, expected_bidding_clairvoyant_utilities = get_clairvoyant_OPT(self.valuation, self.budget, n_users, win_probabilities, available_bids)
        
        
        
        ''' DEFINE THE LOGGING VARIABLE AND START THE TRIALS'''
        
        pricing_all_cumulative_regret = []
        bidding_all_cumulative_regret = []
        
        
        for i in range(self.n_trials):
            '''DEFINE THE COMPANY'''
            
            # Define the pricing agent
            pricing_agent = EXP3PricingAgent(prices, self.n_days)
            
            # Define the bidding agent
            learning_rate = 1/np.sqrt(n_users)
            bidding_agent = FFMultiplicativePacingAgent(available_bids, self.valuation, self.budget, n_users, learning_rate)

            
            company = Company(pricing_agent, bidding_agent, self.valuation, self.product_cost)
            
            
            '''DEFINE THE PUBLISHER I.E. THE AUCTION TYPE (IN THIS CASE A NON-TRUTHFUL AUCTION)'''
            
            # Define the auction type
            auction = GeneralizedFirstPriceAuction(self.ctrs, self.lambdas)
            
            publisher = Publisher(auction)
            
            
            '''DEFINE THE ADVERSARIAL ENVIRONMENT I.E. THE ADVERSARIAL COMPETITORS AND THE ADVERSARIAL PRICING ENVIRONMENT'''
            
        
            pricing_environment = AdversarialPricingEnvironment(demands_t_p_n, self.product_cost, self.n_days, prices)
            
            
            competitors = AdversarialCompetitors(bids_sequence, n_users)
            
            ''' START THE INTERACTION'''
            
            interaction = Interaction(company, publisher, competitors, pricing_environment)
            
            bidding_agent_utilities, m_t, pricing_agent_rewards, visits_per_day = interaction.simulates_n_days(self.n_days, self.users_per_day)
            
            
            ''' COMPUTE THE PRICING CLAIRVOYANT FOR THIS TRIAL'''
            
            demands_t_p = [[
                (demands_t_p_n[i][j][visits_per_day[i]-1])/visits_per_day[i]
                for j in range(len(prices))] 
                for i in range(self.n_days)] 
                
            loss_curve_t = 1 - np.array([np.array(reward_function(prices, np.array(demands_t_p[i]))) for i in range(self.n_days)])
            
            # Best arm(price) in hindsight
            best_arm = np.argmin(loss_curve_t.sum(axis=0))
            expected_pricing_clairvoyant_losses = loss_curve_t[:, best_arm]
            
            ''' LOGGING '''
            pricing_all_cumulative_regret.append(np.cumsum((1 - pricing_agent_rewards) - expected_pricing_clairvoyant_losses))
            bidding_all_cumulative_regret.append(np.cumsum(expected_bidding_clairvoyant_utilities - bidding_agent_utilities))
        
        pricing_all_cumulative_regret = np.array(pricing_all_cumulative_regret)
        bidding_all_cumulative_regret = np.array(bidding_all_cumulative_regret)
        
        
        ''' PLOT THE AVERAGE REGRETS OF THE PRICING AND BIDDING AGENT'''
        plot_regret(pricing_all_cumulative_regret, "Pricing EXP3 Agent Average Regret", self.n_days, self.n_trials)
        
        
        plot_regret(bidding_all_cumulative_regret, "Bidding FFMultiplicativePacing Agent Average Regret", n_users, self.n_trials)
    
    
   
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        