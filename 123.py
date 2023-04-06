#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 16:10:54 2023

@author: james
"""

# =============================================================================
# 123. Best Time to Buy and Sell Stock III
# =============================================================================



"""
https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/solutions/75924/most-consistent-ways-of-dealing-with-the-series-of-stock-problems/


T[i][2][0] = max(T[i-1][2][0], T[i-1][2][1] + prices[i])
T[i][2][1] = max(T[i-1][2][1], T[i-1][1][0] - prices[i])
T[i][1][0] = max(T[i-1][1][0], T[i-1][1][1] + prices[i])
T[i][1][1] = max(T[i-1][1][1], -prices[i])

"""

prices = [3,2,6,5,0,3]


profit_i10 = 0
profit_i11 = float('-inf')
profit_i20 = 0
profit_i21 = float('-inf')


for p in prices:
    profit_i20 = max(profit_i20, profit_i21 + p)
    profit_i21 = max(profit_i21, profit_i10 - p)
    profit_i10 = max(profit_i10, profit_i11 + p)
    profit_i11 = max(profit_i11, -p)















# =============================================================================
# 
# =============================================================================

# prices = [3,2,6,5,0,3]


# """
# Using dynamic programming to solve this problem 
# dp[i][k][j] = dp[i-1] +- prices[i]
# # # profit[i][k][j]. i means the ith day. k means the transaction time. j means whether hold stock

# i means the ith day 
# j means whether or not contain stock
# k means kth action time
# """

# profit = [[[0 for _ in range(2)] for _ in range(3)] for _ in range(len(prices))]

# # initial status for dynamic programming 
# profit[0][0][0], profit[0][0][1] = 0, -prices[0]
# profit[0][1][0], profit[0][1][1] = float('-inf'), float('-inf')
# profit[0][2][0], profit[0][2][1] = float('-inf'), float('-inf')


# # transi

# for i in range(1, len(prices)):
#     p = prices[i]
#     profit[i][0][0] = profit[i-1][0][0]
    
#     profit[i][0][1] = max(profit[i-1][0][1], profit[i-1][0][0]-prices[i])
#     print(profit[i][0][1])
    
#     profit[i][1][0] = max(profit[i-1][1][0], profit[i-1][0][1]+prices[i])
#     print(profit[i][1][0])
    
#     profit[i][1][1] = max(profit[i-1][1][1], profit[i-1][1][0]-prices[i])
#     print(profit[i][1][1])
    
#     profit[i][2][0] = max(profit[i-1][2][0], profit[i-1][1][1]+prices[i])
#     print(profit[i][2][0])
    
#     print(profit)















# =============================================================================
# 
# =============================================================================



# import sys

# prices = [3,2,6,5,0,3]


# profit = [[[0 for _ in range(2)] for _ in range(3)] for _ in range(len(prices))]

# res = 0

# # profit[i][k][j]. i means the ith day. k means the transaction time. j means whether hold stock

# profit[0][0][0], profit[0][0][1] = 0, -prices[0]
# profit[0][1][0], profit[0][1][1] = -sys.maxsize, -sys.maxsize
# profit[0][2][0], profit[0][2][1] = -sys.maxsize, -sys.maxsize


# for i in range(1, len(prices)):
#     profit[i][0][0] = profit[i-1][0][0]
#     profit[i][0][1] = max(profit[i-1][0][1], profit[i-1][0][0] - prices[i])
    
#     profit[i][1][0] = max(profit[i-1][1][0], profit[i-1][0][1] + prices[i])
#     profit[i][1][1] = max(profit[i-1][1][1], profit[i-1][1][0] - prices[i])
    
#     profit[i][2][0] = max(profit[i-1][2][0], profit[i-1][1][1] + prices[i])
    
# max(profit[-1][0][0], profit[-1][1][0], profit[-1][2][0])
    