#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 15:21:00 2023

@author: james
"""

# =============================================================================
# 188. Best Time to Buy and Sell Stock IV
# You are given an integer array prices where prices[i] is the price of a given stock on the ith day, and an integer k.
# 
# Find the maximum profit you can achieve. You may complete at most k transactions: i.e. you may buy at most k times and sell at most k times.
# 
# Note: You may not engage in multiple transactions simultaneously (i.e., you must sell the stock before you buy again).
# =============================================================================




k = 2
prices = [3,2,6,5,0,3]

profit = [[0 for _ in range(3)] for _ in range(len(prices))]

profit[0][0], profit[0][1],  = 0, -prices[0]

# for i in range(len(prices)):
#     for k in range(k):
#         dp[i, k, 