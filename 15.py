#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 23:30:31 2023

@author: james
"""

# def threeSum(self, nums: List[int]) -> List[List[int]]:
#     hashmap = {}
#     for index, num in enumerate(nums):
#         hashmap[num] = index
#     res = []
#     for i in range(len(nums)-1):
#         for j in range(i+1, len(nums)):
#             target = 0 - nums[i] - nums[j]
#             if target in hashmap and hashmap[target] > j:
#                 res.append([nums[i], nums[j], target])
#     return res



nums = [-1,0,1,2,-1,-4]


hashmap = {}
nums.sort()
for index, num in enumerate(nums):
    hashmap[num] = index
res = []
for i in range(len(nums)-1):
    for j in range(i+1, len(nums)):
        target = 0 - nums[i] - nums[j]
        if target in hashmap and hashmap[target] > j and [nums[i], nums[j], target] not in res:
            res.append([nums[i], nums[j], target])