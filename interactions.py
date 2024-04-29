# Classes needed to model the interaction between all the components of the system

import numpy as np

class Customer:
    """
    Represents a customer in the market. The customer behaves following a specified demand curve.
    """

    def __init__(self, demand_curve):
        """
        Initializes a customer with a givenv demand curve.

        Parameters:
            - demand_curve : function
                A function that takes a price as input and returns the probability of the customer buying the product at that price.
                Defined on the interval [0, 1], returns a value in [0, 1].
        """

        self.demand_curve = demand_curve
    
    def get_demand_curve(self):
        """
        Returns the demand curve of the customer.
        """

        return self.demand_curve

    def set_demand_curve(self, demand_curve):
        """
        Sets the demand curve of the customer.
        Use this if the customer's demand curve changes over time.
        """

        self.demand_curve = demand_curve
    
    def check_purchase(self, price):
        """
        Checks if the customer will purchase the product at a given price.

        Parameters:
            - price : float
                The price of the product.

        Returns:
            - product_purchased : int
                1 if the customer purchases the product, 0 otherwise.
        """

        return np.random.binomial(1, self.demand_curve(price))

class Company:
    """
    Represents the company (us in these experiments). The company has a pricing agent and a bidding agent.
    """

    def __init__(self, pricing_agent, bidding_agent, valuation, product_cost):
        """
        Initializes a company with the required components.

        Parameters:
            - pricing_agent : PricingAgent
                The pricing agent of the company.
            - bidding_agent : BiddingAgent
                The bidding agent of the company.
            - valuation : float
                The valuation of the product.
            - product_cost : float
                The cost of the product.
        """
        self.pricing_agent = pricing_agent
        self.bidding_agent = bidding_agent
        self.valuation = valuation
        self.product_cost = product_cost
    
    def set_price(self):
        """
        Sets the price of the product.
        """

        return self.pricing_agent.set_price()

    def bid(self):
        """
        Places a bid in the auction.
        """

        return self.bidding_agent.bid()
    
    def update_pricing_strategy(self, purchase_reward: int):
        """
        Updates the pricing strategy of the company.

        Parameters:
            - purchase_reward : int
                Whether the customer purchased the product or not. 1 if the customer purchased the product, 0 otherwise.
        """

        self.pricing_agent.update(purchase_reward)

    def update_bidding_strategy(self, auction_results):
        """
        Updates the bidding strategy of the company.

        Parameters:
            - auction_results : dict
                A dictionary containing the results of the auction.
        """

        # Compute utility and cost
        f_t, c_t = self._compute_utility_and_cost(auction_results)
        m_t = auction_results['m_t']

        # Update bidding strategy
        self.bidding_agent.update(f_t, c_t, m_t)
    
    def _compute_utility_and_cost(self, auction_results):
        """
        Computes the utility and cost of the company in the auction.

        Parameters:
            - auction_results : dict
                A dictionary containing the results of the auction.

        Returns:
            - f_t : float
                The utility of the company.
            - c_t : float
                The cost of the auction.
        """

        win = auction_results['company_win']
        m_t = auction_results['m_t']

        # Compute utility
        f_t = (self.valuation - m_t) * win

        # Compute cost
        c_t = m_t * win

        return f_t, c_t

class Publisher:
    """
    Represents the publisher in the system.
    The publisher takes care of the auctions and the users' behavior.
    """

    # The company index is always 0, as the company bid is the first one
    COMPANY_INDEX = 0

    def __init__(self, auction, competitors):
        """
        Initializes the publisher with the required components.

        Parameters:
            - auction : Auction
                The type of auctions to be run.
            - competitors : Competitors
                The competitors in the market.
        """

        self.auction = auction
        self.competitors = competitors
    
    def round(self, bid):
        """
        Runs a round of the auction.

        Parameters:
            - bid : float
                The bid of the company.
        
        Returns:
            - auction_results : dict
                A dictionary containing the results of the auction.
            - click_outcome : bool
                Whether the user clicked on the ad or not. True if the user clicked on the ad, False otherwise.
        """

        # Get the competitor bids and the maximum among them
        competitor_bids, m_t = self.competitors.get_bids()

        # Append the competitor bids to the company's bid
        bids = np.append(bid, competitor_bids)

        # Run the auction
        winners, payments_per_click = self.auction.round(bids)
        
        # Check if the company won the auction, and if so, the slot in which the ad was shown
        company_win = self.COMPANY_INDEX in winners
        company_slot = np.where(winners == self.COMPANY_INDEX)[0][0] if company_win else -1

        # Get the company's click through rate
        ctrs = self.auction.get_click_through_rates()
        company_ctr = ctrs[self.COMPANY_INDEX]

        # Simulate the click outcome (True if the user clicked on the ad)
        click_outcome = np.random.rand() < company_ctr

        auction_results = {
            'company_win': company_win,
            'company_slot': company_slot,
            'm_t': m_t
        }

        return auction_results, click_outcome
    
class Competitors:
    """
    Defines the competitors in the market.
    """

    def __init__(self, *args, **kwargs):
        pass

    def get_bids(self):
        pass

class StochasticCompetitors(Competitors):
    """
    Defines the competitors in the market as stochastic agents.
    Simulates hte behavior of a specified number of competitors according to a specified distribution.
    """

    def __init__(self, n_competitors, distribution):
        """
        Initializes the competitors with the required components.

        Parameters:
            - n_competitors : int
                The number of competitors in the market.
            - distribution : function
                A function that takes the number of competitors as input and returns a list of bids.
                Example: distribution = lambda n_competitors: np.random.uniform(0, 1, n_competitors)
        """

        self.n_competitors = n_competitors
        self.distribution = distribution

    def get_bids(self):
        """
        Gets the bids of the competitors in the market.

        Returns:
            - bids : np.array
                An array containing the bids of the competitors.
            - m_t : float
                The maximum bid among the competitors.
        """

        # Sample bids from the distribution
        bids = self.distribution(self.n_competitors)

        # Get the maximum bid
        m_t = np.max(bids)

        return bids, m_t

