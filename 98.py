#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 11:18:54 2023

@author: james
"""

# Python3 program to for tree traversals
 
# A class that represents an individual node in a
# Binary Tree
 
 
class Node:
    def __init__(self, key):
        self.left = None
        self.right = None
        self.val = key
 
 
# A function to do inorder tree traversal
def printInorder(root):
    
    # try:
    #     print(root.val)
    # except:
    #     pass

    res = []
    if root:
 
        # First recur on left child
        res = printInorder(root.left)
        
 
        # then print the data of node
        res.append(root.val)
        # res.append(root.val)
 
        # now recur on right child
        res = res + printInorder(root.right)
    return res
 
# Driver code
if __name__ == "__main__":
    root = Node(1)
    root.left = Node(2)
    root.right = Node(3)
    root.left.left = Node(4)
    root.left.right = Node(5)
 
    # Function call
    print('Inorder traversal of binary tree is')
    
    res = printInorder(root)