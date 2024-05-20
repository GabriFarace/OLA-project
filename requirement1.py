import numpy as np
from auctions import *
from environments import *
from agents import *
from utils import *
from scipy import stats
from interactions import *

''' Requirement classes '''       
class Requirement1:        
    def __init__(self, params, n_trials):
        # Parameters of the problem
        self.n_days = params["n_days"]
        self.users_per_day = params["users_per_day"]
        self.ctrs = params["ctrs"]
        self.lambdas = params["lambdas"]
        self.valuation = params["valuation"]
        self.product_cost = params["product_cost"]
        self.budget = params["budget"]
        self.bidding_agent_type= params["ucb_bidding_agent"]
        self.n_trials = n_trials
        
    def run(self):
        ''' DEFINE THE STOCHASTIC COMPONENTS'''
        # Define a conversion probability function
        conversion_probability = lambda p: 1-p
        
        # Define a distribution over the bids (uniform)
        distribution = lambda n_competitors: np.random.uniform(0, 1, n_competitors)
        
        # The first ctr is the one of the company
        n_competitors = len(self.ctrs)-1
        
        # Number of users for  the bidding algorithm accross all days
        n_users = np.sum(self.users_per_day)
        
        ''' COMPUTE THE TWO CLAIRVOYANTS'''
        
        # Compute the pricing clairvoyant
        K = int(1/(self.n_days**(-1/3)))  # discretization as prescribed by the theory
        min_price, max_price = 0, 1
        reward_function = lambda price, n_sales: (price-self.product_cost)*n_sales
        
        prices = np.linspace(min_price, max_price, K)
        profit_curve = reward_function(prices, conversion_probability(prices))
        best_price_index = np.argmax(profit_curve)
        expected_pricing_clairvoyant_rewards = np.repeat(profit_curve[best_price_index], self.n_days)
        
        '''# Compute the bidding clairvoyant
        #The distribution of the max of k uniform random variables is a beta variable with alpha = k and beta = 1, expected_value = alpha/beta+alpha
        m_t = [n_competitors/(1 + n_competitors) for i in range(n_users)]
        m_t = np.array(m_t)
        
        # 1) Compute it greedily (only in truthful auctions)
        expected_bidding_clairvoyant_bids, expected_bidding_clairvoyant_utilities, expected_bidding_clairvoyant_payments = get_clairvoyant_truthful(self.budget, self.valuation, m_t, n_users)
          
        # 2) Compute it by solving the linear program 
        win_probabilities = stats.beta.cdf(available_bids, n_competitors, 1)
        expected_bidding_clairvoyant_bids, expected_bidding_clairvoyant_utilities = get_clairvoyant_OPT(self.valuation, self.budget, n_users, win_probabilities, available_bids)
        
        since neither of those two worked the computation is done by using the true m_t vectors in the loop below
        '''
        
        ''' DEFINE THE LOGGING VARIABLE AND START THE TRIALS'''
        
        available_bids = np.linspace(0,1,int(1/(n_users**(-1/3))))
        pricing_all_cumulative_regret = []
        bidding_all_cumulative_regret = []
        
        
        for i in range(self.n_trials):
            '''DEFINE THE COMPANY'''
            
            # Define the pricing agent
            pricing_agent = GPUCBAgent(self.n_days, K)
            
            # Define the bidding agent
            if (not self.bidding_agent_type):
                bidding_agent = MultiplicativePacingAgent( self.valuation , self.budget, n_users)
            else:
                bidding_agent = UCBBiddingAgent( available_bids, self.budget, n_users)
            
            company = Company(pricing_agent, bidding_agent, self.valuation , self.ctrs[0] , self.product_cost)
            
            
            '''DEFINE THE PUBLISHER I.E. THE AUCTION TYPE (IN THIS CASE A TRUTHFUL AUCTION)'''
            
            # Define the auction type
            auction = SecondPriceAuction(self.ctrs, self.lambdas)
            
            publisher = Publisher(auction)
            
            
            '''DEFINE THE STOCHASTIC ENVIRONMENT I.E. THE STOCHASTIC COMPETITORS AND THE STOCHASTIC PRICING ENVIRONMENT'''
            
        
            pricing_environment = StochasticPricingEnvironment(conversion_probability, self.product_cost)
            
            
            competitors = StochasticCompetitors(n_competitors, distribution)
            
            ''' START THE INTERACTION'''
            
            interaction = Interaction(company, publisher, competitors, pricing_environment)
            
            bidding_agent_utilities, m_t, pricing_agent_rewards, visits_per_day = interaction.simulates_n_days(self.n_days, self.users_per_day)
            
            
            ''' COMPUTE THE BIDDING CLAIRVOYANT FOR THIS TRIAL'''
            expected_bidding_clairvoyant_bids, expected_bidding_clairvoyant_utilities, expected_bidding_clairvoyant_payments = get_clairvoyant_truthful(self.budget, self.valuation * self.ctrs[0] , self.lambdas[0], m_t, n_users)
            
            ''' LOGGING '''
            pricing_all_cumulative_regret.append(np.cumsum(expected_pricing_clairvoyant_rewards - pricing_agent_rewards))
            bidding_all_cumulative_regret.append(np.cumsum(expected_bidding_clairvoyant_utilities - bidding_agent_utilities))
        
        pricing_all_cumulative_regret = np.array(pricing_all_cumulative_regret)
        bidding_all_cumulative_regret = np.array(bidding_all_cumulative_regret)
        
        
        ''' PLOT THE AVERAGE REGRETS OF THE PRICING AND BIDDING AGENT'''
        plot_regret(pricing_all_cumulative_regret, "Pricing GPUCB Agent Average Regret", self.n_days, self.n_trials)
        
        if (self.bidding_agent_type):
            type_of_bidding_agent = "UCB-like"
        else:
            type_of_bidding_agent = "Multiplicative Pacing"
        
        plot_regret(bidding_all_cumulative_regret, f"Bidding {type_of_bidding_agent} Agent Average Regret", n_users, self.n_trials)
    