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

