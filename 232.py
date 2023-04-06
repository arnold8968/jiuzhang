#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 17:36:09 2023

@author: james
"""

class MyQueue:

    def __init__(self):
        self.inputStack = []
        self.outputStack = []

    def push(self, x: int) -> None:
        self.inputStack.append(x)
        

    def pop(self) -> int:
        n = len(self.inputStack)
        for i in range(n):
            self.outputStack.append(self.inputStack.pop())
        print(self.outputStack)
        res = self.outputStack.pop()
        print(res)
        for i in range(n-1):
            self.inputStack.append(self.outputStack.pop())
        return res
        

    def peek(self) -> int:
        n = len(self.inputStack)
        for i in range(n):
            self.outputStack.append(self.inputStack.pop())
        res = self.outputStack[0]
        for i in range(n):
            self.inputStack.append(self.outputStack.pop())
        
        return res

    def empty(self) -> bool:
        return len(self.inputStack) == 0


# Your MyQueue object will be instantiated and called as such:
obj = MyQueue()

obj.push(1)
obj.push(2)

param_2 = obj.pop()
param_3 = obj.peek()
param_4 = obj.empty()