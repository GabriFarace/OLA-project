from agents4 import *
from auctions import *
from utils import *
import matplotlib.pyplot as plt

class DebugBidder():
    def __init__(self, K):
        self.K = K
        self.valuation = 1
 
    def bid(self):
        return self.K+0.1*np.random.binomial(1,0.5)
 
def find_index(winners, i):
    for j in range(len(winners)):
        if winners[j] == i:
            return j
    return -1
 
class Requirement4:       
    def __init__(self, params):
        type_of_bidders = 4
        # Parameters of the problem
        self.n_users = params["n_users"]    
        self.lambdas = np.sort(np.array(params["lambdas"]))[::-1]
        self.valuations = params["valuations"]
        self.budget = params["budget"]
        self.bidders_per_type = len(self.valuations)
        self.ad_qualities = np.ones(self.bidders_per_type * type_of_bidders )
    
    def run(self):
 
        bids_set =  np.linspace(0,1,11) # np.linspace(0,1,int(self.n_users**(1/3)))
        bidders = []
        bids_log = []
        n_slots = len(self.lambdas)
 
        for i in range( self.bidders_per_type ):
            bidders.append(UCBBiddingAgentExpert( self.valuations[i], bids_set, self.budget, self.n_users ))
            bidders.append(UCBBiddingAgentExpertUpdateRho( self.valuations[i], self.budget, self.n_users ))
            bidders.append(MultiplicativePacingAgent(self.valuations[i], self.budget, self.n_users))
            bidders.append(FFMultiplicativePacingAgent(bids_set, self.valuations[i], self.budget, self.n_users))
 
        n_bidders = len(bidders)
 
        auction = GeneralizedFirstPriceAuction(self.ad_qualities, self.lambdas)
 
        for u in range(self.n_users):
            bids = []
            for bidder in bidders:
                bids.append(bidder.bid())

            bids_log.append(bids)
            winners, payment_per_click = auction.round(bids)
            payment = payment_per_click*self.lambdas
 
            for i,bidder in enumerate(bidders):
                rank = find_index(winners, i)
                c_t = 0 
                if rank != -1:
                    c_t = payment[rank]
                m_t = np.delete(bids, i)
                m_t = np.sort(m_t)[::-1]
                m_t = m_t[:n_slots]
 
                bidder.update(self.lambdas,rank,c_t,m_t)

        print("finished bidding rounds")

        # Instantiate the clairvoyants
        bids_log = np.array(bids_log)
        bids_log.reshape(self.n_users,n_bidders)
 
        clvoy_per_round_utility = []
 
        for i, b in enumerate(bidders):
            m_t = np.delete(bids_log, i, axis=1)
            m_t = np.sort(m_t, axis=1)[:,::-1]
            m_t = m_t[:,:n_slots]
 
            available_bids = np.linspace(0,b.valuation,101)
            win_prob =  np.array([np.sum(bd > m_t, axis=0)/self.n_users for bd in available_bids])
            diff_prob = win_prob[:,1:]-win_prob[:,:-1]
            win_prob = np.append(win_prob[:,:1], diff_prob, axis=1)
            #print(win_prob)
            avg_lambdas = np.dot(win_prob,self.lambdas)
           
            f = (b.valuation-available_bids)*avg_lambdas
            ct = available_bids*avg_lambdas
            rho = self.budget/self.n_users
 
            gamma, per_round_utility = solve_linear_program(f, ct, rho)
            clvoy_per_round_utility.append(per_round_utility)
       
        print("\nfinished computing clairvoyant")

        print("\nAgents' utility")
        for i, bidder in enumerate(bidders):
            print(i, bidder.get_utility(), clvoy_per_round_utility[i]*self.n_users)

        np.savetxt("output.csv", bids_log, delimiter=",")

        log_utility = []
        for b in bidders:
            log_utility.append( b.utility )
        log_utility = np.array(log_utility)
        log_utility = log_utility.cumsum(axis=1)
        np.savetxt("utility.csv", log_utility)

        clvoy_utility = np.ones((12,1000)) * np.reshape(clvoy_per_round_utility,(12,1))
        clvoy_utility = clvoy_utility.cumsum(axis=1)
        np.savetxt("clvoy_utility", clvoy_utility)
        # Compute cumulative regret for each bidder
        # Should we average on bidders of the same type?

        #plt.plot( bids_log[:,3] )
        plt.figure(figsize = [17, 4.8])
        # plt.plot( bidders[3].utility )
        plt.plot( bids_log[:,4] )
        plt.plot( bids_log[:,5] )
        #plt.plot( bids_log[:,11] )
        plt.savefig("./output3.png")
        # Plot
