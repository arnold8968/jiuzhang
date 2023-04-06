#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 14:11:22 2023

@author: james
"""

# =============================================================================
# You are given an array prices where prices[i] is the price of a given stock on the ith day.

# You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.

# Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.

 

# Example 1:

# Input: prices = [7,1,5,3,6,4]
# Output: 5
# Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
# Note that buying on day 2 and selling on day 1 is not allowed because you must buy before you sell.
# Example 2:

# Input: prices = [7,6,4,3,1]
# Output: 0
# Explanation: In this case, no transactions are done and the max profit = 0.

# =============================================================================

prices = [7,1,5,3,6,4]

profit_0, profit_1 = 0, float('-inf')

for p in prices:
    profit_0 = max(profit_0, profit_1+p)
    profit_1 = max(profit_1, -p)











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













# =============================================================================
# 
# =============================================================================



# prices = [7,1,5,3,6,4]

# # prices = [7,6,4,3,1]

# prices = [1,2]

# buy, sell, profit = prices[0], 0, 0

# for i in range(0, len(prices)):
#     current = prices[i]
#     buy = min(buy, prices[i])
#     # sell = max(sell, prices[i])
#     profit = max(profit, prices[i]-buy)