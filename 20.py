#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 15:51:08 2023

@author: james
"""

def isValid(s):
    pare_dict = {')': '(', ']': '[', '}': '{'}
    stack = []
    # for pare in s:
    #     if pare not in pare_dict:
    #         stack.append(pare)
    #     elif not stack or pare_dict[pare] != stack.pop():
    #         return False
    # return True
    # # pare_map = {')': '(', ']': '[', '}': '{'}
    # for pare in s:
    #     if pare not in 




s = "(]"

pare_dict = {')': '(', ']': '[', '}': '{'}
stack = []
for pare in s:
    if pare not in pare_dict:
        stack.append(pare)
    elif pare_dict[pare] != stack.pop():
        print('False')


# pare_map = {')': '(', ']': '[', '}': '{'}