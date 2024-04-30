# Classes needed to model the interaction between all the components of the system

import numpy as np
from auctions import *
from environments import *
from agents import *
from utils import *
from scipy import optimize
from scipy import stats
import matplotlib.pyplot as plt

'''Environment for pricing problems'''
class PricingEnvironment:
    def __init__(self):
        pass

    def round(self, price_t, n_t):
        pass

class StochasticPricingEnvironment(PricingEnvironment):
    def __init__(self, conversion_probability, cost):
        self.conversion_probability = conversion_probability
        self.cost = cost
    
    def round(self, price_t, n_t):
        demand_t = np.random.binomial(n=n_t, p=self.conversion_probability(price_t))
        reward_t = demand_t * (price_t - self.cost)
        return demand_t, reward_t    

class AdversarialPricingEnvironment(PricingEnvironment):
    def __init__(self, conversion_probabilities, cost, T):
        self.conversion_probabilities = conversion_probabilities
        self.cost = cost
        self.T = T
        self.t = 0
        
    def round(self, price_t, n_t):
        demand_t = np.random.binomial(n=n_t, p=self.conversion_probabilities[self.t](price_t))
        reward_t = demand_t * (price_t - self.cost)
        self.t += 1
        return demand_t, reward_t    


'''Competitors for the auction problem'''
class Competitors:

    def __init__(self, *args, **kwargs):
        pass

    def get_bids(self):
        pass

class StochasticCompetitors(Competitors):

    def __init__(self, n_competitors, distribution):

        self.n_competitors = n_competitors
        self.distribution = distribution

    def get_bids(self):

        # Sample bids from the distribution
        bids = self.distribution(self.n_competitors)

        # Get the maximum bid
        m_t = np.max(bids)

        return bids, m_t
    
class AdversarialCompetitors(Competitors):

    def __init__(self, bids_sequence, T):

        self.bids_sequence = bids_sequence
        self.T = T
        self.t = 0

    def get_bids(self):

        bids = self.bids_sequence[:, self.t]

        # Get the maximum bid
        m_t = np.max(bids)
        
        self.t += 1

        return bids, m_t     

''' Company class '''
class Company:

    def __init__(self, pricing_agent, bidding_agent, valuation, product_cost):

        self.pricing_agent = pricing_agent
        self.bidding_agent = bidding_agent
        self.valuation = valuation
        self.product_cost = product_cost
    
    def set_price(self):

        return self.pricing_agent.set_price()
    
        # Here add the changes if we consider the value depend on the price, assigning a new value to the valuation variable

    def bid(self):

        return self.bidding_agent.bid()
    
    
    def update_pricing_strategy(self, purchase_reward):

        self.pricing_agent.update(purchase_reward)

    def update_bidding_strategy(self, auction_results):

        # Compute utility and cost
        f_t, c_t = self._compute_utility_and_cost(auction_results)
        m_t = auction_results['m_t']

        # Update bidding strategy
        self.bidding_agent.update(f_t, c_t, m_t)
    
        return f_t, c_t
    
    def _compute_utility_and_cost(self, auction_results):

        #payment could be m_t in case of truthful auction, or the bid itself in case of non truthful auctions
        win = auction_results['company_win']
        payment = auction_results['company_payment']

        # Compute utility
        f_t = (self.valuation - payment) * win

        # Compute cost
        c_t = payment * win

        return f_t, c_t

'''This class includes the logic of the specific auctions and return the results for the company for each round'''
class Publisher:

    # The company index is always 0, as the company bid is the first one
    COMPANY_INDEX = 0

    def __init__(self, auction):

        self.auction = auction

    
    def round(self, bid, competitor_bids, m_t):

        # Append the competitor bids to the company's bid
        bids = np.append(bid, competitor_bids)

        # Run the auction
        winners, payments_per_click = self.auction.round(bids)
        
        # Check if the company won the auction, and if so, the slot in which the ad was shown
        company_win = self.COMPANY_INDEX in winners
        company_slot = -1
        company_payment = 0
        if company_win:   
            for i in range(len(winners)):
                if winners[i] == self.COMPANY_INDEX:
                    company_slot = i
                    break
            company_payment = payments_per_click[company_slot]

        # Get the company's click through rate
        ctrs = self.auction.get_click_through_rates()
        company_ctr = ctrs[self.COMPANY_INDEX]

        # Simulate the click outcome (True if the user clicked on the ad) only if the company actually won some slot
        click_outcome = False
        
        if company_win:
            click_outcome = np.random.rand() < company_ctr    

        auction_results = {
            'company_win': company_win,
            'company_slot': company_slot,
            'company_payment': company_payment,
            'm_t': m_t
        }

        return auction_results, click_outcome
 
''' Interaction class '''    
class Interaction:
    def __init__(self, company, publisher, competitors, pricing_environment):
        self.company = company
        self.publisher = publisher
        self.competitors = competitors
        self.pricing_environment = pricing_environment
    
        # Logging variables
        self.m_t = np.array([])
        self.bidding_agent_utilities = np.array([])
        self.bidding_agent_payments = np.array([])
        self.bidding_agent_bids = np.array([])
        self.pricing_agent_rewards = np.array([])
    
    def simulates_n_days(self, n_days, users_per_day):
        # Simulates n_days of interaction
        
        for i in range(n_days):
            self._day(users_per_day[i])
    
        # return necessary data for computing clairvoyant and plotting results    
        return self.bidding_agent_utilities, self.m_t, self.pricing_agent_rewards    
    
    def _day(self, n_users):
        # Simulates a day, made up of multiple auctions

        # Company sets a price and initialize a counter of visits
        price = self.company.set_price()
        n_visits = 0

        # Company faces a series of auctions
        for user in range(n_users):
            # Company and Competitors bids
            bid = self.company.bid()
            competitor_bids, m_t_round = self.competitors.get_bids() 
            
            # Publisher runs the auction and simulates the click outcome
            auction_results, click_outcome = self.publisher.round(bid, competitor_bids, m_t_round)

            # Update the company's bidding strategy and get utility and payment of the company for this round
            f_t, c_t = self.company.update_bidding_strategy(auction_results)

            # Increment counter of visits if user clicked on the ad
            if click_outcome:
                n_visits += 1
                
            # Logging
            self.bidding_agent_bids = np.append(self.bidding_agent_bids, bid)
            self.bidding_agent_utilities = np.append(self.bidding_agent_utilities, f_t)
            self.bidding_agent_payments = np.append(self.bidding_agent_payments, c_t)
            self.m_t = np.append(self.m_t, m_t_round)
                
        # Get the reward for this day for the pricing agent  
        if (n_visits == 0):
            n_visits = 1
        demand_t, reward_t = self.pricing_environment.round(price, n_visits)  
        
        # Update the company's pricing strategy
        pricing_reward = reward_t/n_visits
        self.company.update_pricing_strategy(pricing_reward)
        
        # Logging
        self.pricing_agent_rewards = np.append(self.pricing_agent_rewards, pricing_reward)
        
        
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
            
            company = Company(pricing_agent, bidding_agent, self.valuation, self.product_cost)
            
            
            '''DEFINE THE PUBLISHER I.E. THE AUCTION TYPE (IN THIS CASE A TRUTHFUL AUCTION)'''
            
            # Define the auction type
            auction = SecondPriceAuction(self.ctrs)
            
            publisher = Publisher(auction)
            
            
            '''DEFINE THE STOCHASTIC ENVIRONMENT I.E. THE STOCHASTIC COMPETITORS AND THE STOCHASTIC PRICING ENVIRONMENT'''
            
        
            pricing_environment = StochasticPricingEnvironment(conversion_probability, self.product_cost)
            
            
            competitors = StochasticCompetitors(n_competitors, distribution)
            
            ''' START THE INTERACTION'''
            
            interaction = Interaction(company, publisher, competitors, pricing_environment)
            
            bidding_agent_utilities, m_t, pricing_agent_rewards = interaction.simulates_n_days(self.n_days, self.users_per_day)
            
            
            ''' COMPUTE THE BIDDING CLAIRVOYANT FOR THIS TRIAL'''
            expected_bidding_clairvoyant_bids, expected_bidding_clairvoyant_utilities, expected_bidding_clairvoyant_payments = get_clairvoyant_truthful(self.budget, self.valuation, m_t, n_users)
            
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
    
    

        


