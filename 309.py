#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 11:24:20 2023

@author: james
"""

# =============================================================================
# 309. Best Time to Buy and Sell Stock with Cooldown
# 
# Input: prices = [1,2,3,0,2]
# Output: 3
# Explanation: transactions = [buy, sell, cooldown, buy, sell]
# Example 2:
# 
# Input: prices = [1]
# Output: 0
# =============================================================================

# prices = [1,2,3,0,2]
prices = [2,1]
# prices = [1]

# profit[i][0] means colldown
# profit[i][1] means buy
# profit[i][2] means sell

if len(prices) <= 1: 0

profit = [[0 for _ in range(3)] for _ in range(len(prices))]

profit[0][0], profit[0][1], profit[0][2] = 0, -prices[0], float('-inf')

for i in range(1, len(prices)):
    profit[i][0] = max(profit[i-1][0], profit[i-1][2])
    profit[i][1] = max(profit[i-1][1], profit[i-1][0] - prices[i])
    profit[i][2] = profit[i-1][1] + prices[i]
    
print(profit[-1][2],profit[-1][1], profit[-1][0])