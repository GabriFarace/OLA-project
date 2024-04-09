# Online Learning Applications Project

This repository contains the source code of the project for the Online Learning Applications course at Politecnico di Milano.

The goal of this project is to design online learning algorithms to **handle a marketing campaign to sell products**. This includes:
- an **advertising** campaign
- a **pricing** problem


## Overview of the Interactions
A company runs a marketing campaign for some months.
The interaction goes as follows:

> At each day:
> - The company chooses a price $p$ (which remains fixed for the day).
> - The company faces a sequence of auctions. For each auction:
>     - The company chooses a bid $b$
>     - An ad slot is (possibly) assigned to the company depending on $b$, the competing bids, and the auction format
>     - If the ad is clicked, a user visits the company's web page
>     - The user buys the company's product with a probability that depends on the price $p$


## Project Requirements
The project comprises of four requirements:


### <u>Requirement 1 - Stochastic Environment</u>

Build a **stochastic** environment. At a high level, it should include:
- A distribution over the bids of the other agents (**competitors**)
- A function specifying the probability with which a user buys for every price (**demand curve**)

***Pricing Algorithm***  
Build a pricing strategy using the continuous set of prices $p \in [0,1]$ and **Gaussian Processes**.

***Bidding Algorithm***  
Consider a sequence of **second-price auctions**. Build two learning algorithms to deal with the bidding problem:
- a primal-dual algorithm for **truthful** auctions
- a UCB-like algorithm

---

### <u>Requirement 2 - Adversarial Environment</u>

Build a **highly non-stationary** environment. At a high level, it should include:
- A sequence of competing bids of the other agents (e.g. sampled from a distribution that changes *quickly* over time)
- For each day, a function specifying the probability with which a user buys for every price (the demand curve also changes *quickly* over time)ù

***Pricing Algorithm***  
Build a pricing strategy **discretizing** the continuous set of prices $p \in [0,1]$.

***Bidding Algorithm***  
Consider a **generalized first-price auction**. Build a learning algorithm to deal with the bidding problem. In particular:
- a primal-dual algorithm for **non-truthful** auctions.

---

### <u>Requirement 3 - Two Extensions for Pricing</u>

We extend the pricing problem along two directions. since we focus only on pricing, we directly consider a demand curve **with noise**.

Build a **non-stationary** environment for the pricing problem. At a high level:
- Days are partitioned in intervals
- The demand curve is different in each interval
- The **noisy demand curve** specifies how many customers will buy for every price depending on the current interval

***Pricing Algorithm***  
Build a pricing strategy using the discretization of the prices $p \in [0,1]$ and:
- **Sliding Window**
- **CUSUM**

***BONUS POINT***  
Build a **two-item stochastic** pricing environment.  
In particular, the company proposes two items $i_1$ and $i_2$. The model should include:
- A noisy demand curve $D(p_1, p_2) + \eta$ that specifies how many users will buy each product depending on *the two prices*

***Goal***: build a regret minimizer for the continuous action set $[0,1]^2$ using **two-dimensional Gaussian Processes**.

---

### <u>Requirement 4 - Compare Different Bidding Algorithms</u>

The goal is to compare different algorithms that we have seen for bidding and let them play one against the others in a **generalized first-price auction**. In particular, consider:
- a primal-dual algorithm for **truthful** auctions
- a primal-dual algorithm for **non-truthful** auctions
- a UCB-like algorithm