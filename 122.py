#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 15:18:42 2023

@author: james
"""

# =============================================================================
# 122. Best Time to Buy and Sell Stock II
# =============================================================================



# =============================================================================
# https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/solutions/75924/most-consistent-ways-of-dealing-with-the-series-of-stock-problems/
# 
# dynamic Programming
# 
# =============================================================================



"""
T[i][k][0] = max(T[i-1][k][0], T[i-1][k][1] + prices[i])
T[i][k][1] = max(T[i-1][k][1], T[i-1][k-1][0] - prices[i]) = max(T[i-1][k][1], T[i-1][k][0] - prices[i])
"""


prices = [7,1,5,3,6,4]



profit_ik0, profit_ik1 = 0, -prices[0]

for i in range(1, len(prices)):
    profit_ik0_old = profit_ik0
    
    profit_ik0 = max(profit_ik0, profit_ik1 + prices[i])
    profit_ik1 = max(profit_ik1, profit_ik0_old - prices[i])













# =============================================================================
# # Greedy
# =============================================================================

# prices = [7,1,5,3,6,4]

# profit = 0

# for i in range(1, len(prices)):
#     profit += max(0, prices[i] - prices[i-1])
    










# =============================================================================
# 
# =============================================================================



# # Dynamic Programming 

# prices = [7,1,5,3,6,4]

# res = 0

# profit = [[0 for _ in range(3)] for _ in range(len(prices))]


# # profit[0][0] means there is no buy or sell action 
# # profit[0][1] means buy a stock
# # profit[0][2] means sell a stock

# profit[0][0], profit[0][1], profit[0][2] = 0, -prices[0], 0




# for i in range(1, len(prices)):
#     profit[i][0] = profit[i-1][0]
#     profit[i][1] = max(profit[i-1][1], profit[i-1][0] - prices[i])
#     profit[i][2] = profit[i-1][1] + prices[i]
#     res = max(profit[i][0],profit[i][2],res)
