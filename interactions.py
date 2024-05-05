# Classes needed to model the interaction between all the components of the system

import numpy as np
from auctions import *
from environments import *
from agents import *
from utils import *
from scipy import stats



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
        self.visits_per_day = []
    
    def simulates_n_days(self, n_days, users_per_day):
        # Simulates n_days of interaction
        
        for i in range(n_days):
            self._day(users_per_day[i])
    
        # return necessary data for computing clairvoyant and plotting results    
        return self.bidding_agent_utilities, self.m_t, self.pricing_agent_rewards, self.visits_per_day    
    
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
        self.visits_per_day.append(n_visits)
