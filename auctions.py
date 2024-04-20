### AUCTIONS ###

import numpy as np

# Base Auction class
class Auction:
    def __init__(self, *args, **kwargs):
        pass

    def get_winners(self, bids):
        pass

    def get_payments_per_click(self, winners, values, bids):
        pass

    def round(self, bids):
        winners, values = self.get_winners(bids) # allocation mechanism
        payments_per_click = self.get_payments_per_click(winners, values, bids) # payment rule
        return winners, payments_per_click

# Truthful Auctions
class SecondPriceAuction(Auction):
    def __init__(self, click_through_rates):
        self.click_through_rates = click_through_rates
        self.n_advertisers = len(self.click_through_rates)
    
    def get_winners(self, bids):
        advertisers_values = self.click_through_rates * bids
        advertisers_ranking = np.argsort(advertisers_values)
        winner = advertisers_ranking[-1]
        return winner, advertisers_values
    
    def get_payments_per_click(self, winners, values, bids):
        advertisers_ranking = np.argsort(values)
        second_winner = advertisers_ranking[-2]
        payment = values[second_winner] / self.click_through_rates[winners]
        return payment.round(2)

class VCGAuction(Auction):
    def __init__(self, click_through_rates, lambdas):
        self.click_through_rates = click_through_rates
        self.lambdas = lambdas
        self.n_advertisers = len(self.click_through_rates)
        self.n_slots = len(self.lambdas)
    
    def get_winners(self, bids):
        advertisers_values = self.click_through_rates * bids
        advertisers_ranking = np.argsort(advertisers_values)
        winners = advertisers_ranking[-self.n_slots:]
        winners_values = advertisers_values[winners]
        return winners, winners_values
    
    def get_payments_per_click(self, winners, values, bids):
        payments_per_click = np.zeros(self.n_slots)
        for i, winner in enumerate(winners):
            Y = sum(np.delete(values, i) * self.lambdas[-self.n_slots + 1:])
            X = sum(np.delete(values * self.lambdas, i))
            payments_per_click[i] = (Y - X) / (self.lambdas[i] * self.click_through_rates[winner])
        return payments_per_click.round(2)

# Non-Truthful Auctions
class FirstPriceAuction(Auction):
    def __init__(self, click_through_rates):
        self.click_through_rates = click_through_rates
        self.n_advertisers = len(self.click_through_rates)

    def get_winners(self, bids):
        advertisers_values = self.click_through_rates * bids
        advertisers_ranking = np.argsort(advertisers_values)
        winner = advertisers_ranking[-1]
        return winner, advertisers_values

    def get_payments_per_click(self, winners, values, bids):
        payment = bids[winners]
        return payment.round(2)

class GeneralizedFirstPriceAuction(Auction):
    def __init__(self, click_through_rates, lambdas):
        self.click_through_rates = click_through_rates
        self.lambdas = lambdas
        self.n_advertisers = len(self.click_through_rates)
        self.n_slots = len(self.lambdas)
    
    def get_winners(self, bids):
        advertisers_values = self.click_through_rates * bids
        advertisers_ranking = np.argsort(advertisers_values)
        winners = advertisers_ranking[-self.n_slots:]
        winners = winners[::-1]
        winners_values = advertisers_values[winners]
        return np.array(winners), winners_values
    
    def get_payments_per_click(self, winners, values, bids):
        payments_per_click = np.array(bids)[winners]
        return [payment.round(2) for payment in payments_per_click]