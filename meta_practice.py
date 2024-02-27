#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 16:35:08 2024

@author: james
"""

class TreeNode:
    def __init__(self, value=0, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

def is_complete_binary_tree(root):
    if not root:
        return True

    queue = [root]
    end = False  # Flag to mark the end of complete part

    while queue:
        current = queue.pop(0)
        print(current.value)

        if current.left:
            print(current.left.value)
            if end:
                # If we have seen a non-full node, and we see a node with children, it's not complete
                return False
            queue.append(current.left)
        else:
            # If this node doesn't have a left child, the next nodes must not have children
            end = True

        if current.right:
            print(current.right.value)
            if end:
                # If we have seen a non-full node, and we see a node with children, it's not complete
                return False
            queue.append(current.right)
        else:
            # If this node doesn't have a right child, the next nodes must not have children
            end = True

    return True

# Example usage
# Constructing a complete binary tree
#        1
#       / \
#      2   3
#     / \
#    4   5
# root = TreeNode(1, TreeNode(2, TreeNode(4), TreeNode(5)), TreeNode(3))


# resu = is_complete_binary_tree(root)


# print(is_complete_binary_tree(root))  # Output: True

# Constructing a non-complete binary tree
#        1
#       / \
#      2   3
#       \   \
#        5   6
# root = TreeNode(1, TreeNode(2, None, TreeNode(5)), TreeNode(3, None, TreeNode(6)))

# is_complete_binary_tree(root)

# print(is_complete_binary_tree(root))  # Output: False














def minAddToMakeValid(s):
    left = right = 0
    for symbol in s:
        right += 1 if symbol == '(' else -1
        if right == -1:
            right += 1
            left += 1
    return right + left



# s = "()"

# print(minAddToMakeValid(s))






def validWordAbbreviation(self, word: str, abbr: str) -> bool:
    p1 = p2 = 0
    while p1 < len(word) and p2 < len(abbr):
        if abbr[p2].isdigit():
            if abbr[p2] == '0': # leading zeros are invalid
                return False
            shift = 0
            while p2 < len(abbr) and abbr[p2].isdigit():
                shift = (shift*10)+int(abbr[p2])
                p2 += 1
            p1 += shift
        else:
            if word[p1] != abbr[p2]:
                return False
            p1 += 1
            p2 += 1
    return p1 == len(word) and p2 == len(abbr)





def abbrtest(word, abbr):
    p1 = p2 = 0
    
    while p1<len(word) and p2<len(abbr):
        if abbr[p2].isdigit():
            if abbr[p2] == '0':
                return False
            shift = 0
            while p2 < len(abbr) and abbr[p2].isdigit():
                shift = shift* 10 + int(abbr[p2])
                p2 += 1
            p1 += shift 
        
        else:
            if word[p1] != abbr[p2]:
                return False
            p1 += 1
            p2 += 1
    return p1 == len(word) and p2 == len(abbr)


# print(abbrtest(word, abbr))
            




# Given two sorted, non-overlapping interval lists, return a 3rd interval list that is the union of the input interval lists.
# For example:
# Input:
# {[1,2], [3,9]}
# {[4,6], [8,10], [11,12]}



# A = [[1,5],[10,14],[16,18]]
# B = [[2,6],[8,10],[11,20]]



def two_interval_list(A, B):
    i = j = 0
    resu = []
    
    while i < len(A) or j < len(B):
        if i == len(A):
            curr = B[j]
            j += 1
        elif j == len(B):
            curr = A[i]
            i += 1
        elif A[i][0] < B[j][0]:
            curr = A[i]
            i += 1
        else:
            curr = B[j]
            j+= 1
        
        
        if resu and resu[-1][-1] >= curr[0]:
            resu[-1][-1] = max(resu[-1][-1], curr[-1])
        
        else:
            resu.append(curr)
        
    return resu        
        
        
        

# follow up same list


# A = [[1,5],[2,6],[8,10]]

# i = 1

# resu = [A[0]]

# for i in range(1, len(A)):
#     if resu[-1][-1] >= A[i][0]:
#         resu[-1][-1] = max(resu[-1][-1], A[i][-1])
#     else:
#         resu.append(A[i])
    






def binaryexp(x, n):
    if n == 0:
        return 1
    
    if n < 0:
        return 1 / binaryexp(x, -n)
    
    if n % 2 == 1:
        return x * binaryexp(x * x, (n-1) // 2)
    
    else:
        return binaryexp(x * x, n // 2)
    

print(binaryexp(2, 3))



def power(x, n):
    r = 1
    for _ in range(n):
        r *= x
    return r


# iteration



def power2(x, n):
    res = 1
    while n > 0:
        if n % 2 == 1:
            res *= x
        x *= x
        n //= 2
        
        
        




class Solution:
    def validPalindrome(self, s: str) -> bool:
        def check_palin(s, i, j):
            while i < j:
                if s[i] != s[j]:
                    return False
                i += 1
                j -= 1
                
            return True
        
        i = 0
        j = len(s) - 1
        while i < j:
            if s[i] != s[j]:
                return check_palin(s, i, j- 1) or check_palin(s, i+1, j)
            i += 1
            j -= 1
        return True





import random

class CitySelector:
    def __init__(self, cities):
        self.cities = cities
        self.cumulative_weights = []
        total_population = 0
        for city, population in cities:
            total_population += population
            self.cumulative_weights.append(total_population)

    def select_city(self):
        rnd = random.uniform(0, self.cumulative_weights[-1])
        for i, weight in enumerate(self.cumulative_weights):
            if rnd < weight:
                return self.cities[i][0]

# Example usage:
# cities = [("NY", 7), ("SF", 5), ("LA", 8)]
# selector = CitySelector(cities)

# # Simulate multiple calls to select_city
# for _ in range(4):
#     print(selector.select_city())
    
    
    




class Solution:

    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """

        # Stack for tree traversal
        stack = [root]

        # Dictionary for parent pointers
        parent = {root: None}

        # Iterate until we find both the nodes p and q
        while p not in parent or q not in parent:

            node = stack.pop()

            # While traversing the tree, keep saving the parent pointers.
            if node.left:
                parent[node.left] = node
                stack.append(node.left)
            if node.right:
                parent[node.right] = node
                stack.append(node.right)

        # Ancestors set() for node p.
        ancestors = set()

        # Process all ancestors for node p using parent pointers.
        while p:
            ancestors.add(p)
            p = parent[p]

        # The first ancestor of q which appears in
        # p's ancestor set() is their lowest common ancestor.
        while q not in ancestors:
            q = parent[q]
        return q
    
    

    





#     So most straightforward solution is to use game board to keep track of positions that have been probed already. Also we need to keep track of a list of positions as the actual path taken.
# Board = [
# [0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 1, 1, 0], [0, 0, 1, 0, 1, 0, 1], [1, 1, 1, 0, 0, 0, 0]
# ]




def dfs(board, x, y ,path):
    path.append((x, y))
    if x == len(board) - 1 and y == len(board[0]) - 1:
        return True
    
    if x < 0 or x >= len(board) or y < 0 or y>=len(board[0]) or not board[x][y] == 0:
        del path[-1]
        return False
    
    board[x][y]=2 
    
    if dfs(board, x+1, y, path):
        return True 
    if dfs(board, x, y+1, path):
        return True 
    if dfs(board, x-1, y, path):
        return True
    if dfs(board, x, y-1, path):
        return True
    
    del path[-1]
    
    return False 



# Board = [
# [0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 1, 1, 0], [0, 0, 1, 0, 1, 0, 1], [1, 1, 1, 0, 0, 0, 0]
# ]
# path = []
# print(dfs(Board, 0, 0, path))
    

    

from collections import deque
    
def bfs(board):
    rows, cols = len(board), len(board[0])
    
    directions = [(0,1), (1, 0), (0, -1), (-1, 0)]
    
    # (path, (x, y))
    queue = deque([([(0, 0)], (0, 0))])

    visited = set([0, 0])
    
    while queue:
        path, (x, y) = queue.popleft()

        if (x, y) == (rows - 1, cols - 1):
            return path 
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy 
            if 0 <= nx < rows and 0 <= ny < cols and board[nx][ny] == 0 and (nx, ny) not in visited:
                queue.append((path + [(nx, ny)], (nx, ny)))
                visited.add((nx, ny))
    return False





# # Example board
# board = [
#     [0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 1, 0, 0, 1, 0],
#     [0, 0, 1, 0, 1, 1, 0],
#     [0, 0, 1, 0, 0, 0, 0],
#     [1, 1, 1, 0, 0, 0, 0]
# ]

# # Find the shortest path
# shortest_path = bfs(board)
# print(shortest_path)








# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def verticalTraversal(self, root: TreeNode):
        node_list = []

        def DFS(node, row, column):
            if node is not None:
                node_list.append((column, row, node.val))
                # preorder DFS
                DFS(node.left, row + 1, column - 1)
                DFS(node.right, row + 1, column + 1)

        # step 1). construct the node list, with the coordinates
        DFS(root, 0, 0)

        # step 2). sort the node list globally, according to the coordinates
        node_list.sort()

        # step 3). retrieve the sorted results grouped by the column index
        ret = []
        curr_column_index = node_list[0][0]
        curr_column = []
        for column, row, value in node_list:
            if column == curr_column_index:
                curr_column.append(value)
            else:
                # end of a column, and start the next column
                ret.append(curr_column)
                curr_column_index = column
                curr_column = [value]
        # add the last column
        ret.append(curr_column)

        return ret
    
    
    
    





def findLocalMinimum(arr):
    low, high = 0, len(arr) - 1

    # Handle edge cases for arrays of length 1 and 2
    if len(arr) == 1:
        return arr[0], 0
    if arr[0] <= arr[1]:
        return arr[0], 0
    if arr[-1] <= arr[-2]:
        return arr[-1], len(arr) - 1

    while low <= high:
        mid = (low + high) // 2

        # Check if the mid element is a local minimum
        if (mid == 0 or arr[mid] <= arr[mid - 1]) and (mid == len(arr) - 1 or arr[mid] <= arr[mid + 1]):
            return arr[mid], mid

        # If the left neighbor is less than the mid element, then there must be a local min on the left half
        if mid > 0 and arr[mid - 1] < arr[mid]:
            high = mid - 1
        else:
            # Otherwise, the local min must be on the right half
            low = mid + 1

    return "No local minimum found"

# # Example usage
# arr = [9, 6, 3, 14, 5, 7, 4]
# local_min, index = findLocalMinimum(arr)
# print(f"Local minimum: {local_min} found at index {index}")





def findLocalMin(arr):
    if not arr:
        return None 
    res = []
    n = len(arr)
    if n == 1 or arr[0] <= arr[1]:
        res.append(arr[0])
    if arr[n-1] <= arr[n-2]:
        res.append(arr[n-1])
    
    for i in range(1, n-1):
        if arr[i] <= arr[i -1] and arr[i] <= arr[i+1]:
            res.append(arr[i])
    
    return res


# arr = [9, 6, 3, 14, 5, 7, 4]


# print(findLocalMin(arr))




def findLocalMin2(arr):
    if not arr:
        return None 
    
    low, high = 0, len(arr)-1
    
    while low<=high:
        mid = low+(high-low) // 2
        
        if (mid == 0 or arr[mid-1]>=arr[mid]) and (mid == len(arr)-1 or arr[mid+1] >= arr[mid]):
            return arr[mid]
        
        if mid > 0 and arr[mid-1] < arr[mid]:
            high = mid - 1
        else:
            low = mid + 1
            
    return None 



# arr = [9, 6, 3, 14, 5, 7, 4]


# print(findLocalMin2(arr))






def moving_average(arr, k):
    res = []
    
    moving_sum = sum(arr[:k])
    
    res.append(moving_sum)
    
    for i in range(1, len(arr)-k+1):
        moving_sum = moving_sum - arr[i-1] + arr[i+k-1]
        res.append(moving_sum)

    return [num / k for num in res]


# arr = [1,2,3,4,5]
# k = 3

# print(moving_average(arr, k))





# 31. Next Permutation

def swap(nums, i, j):
    nums[i], nums[j] = nums[j], nums[i]

def reverse(nums, i):
    j = len(nums) - 1
    while i < j:
        swap(nums, i, j)
        i += 1
        j -= 1

def nextPermutation(nums):
    i = len(nums) - 2 
    
    while i >= 0 and nums[i+1] < nums[i]:
        i -= 1
    
    if i >= 0:
        j = len(nums) - 1
        while nums[j] < nums[i]:
            j -= 1
        swap(nums, i, j)
    reverse(nums, i + 1)
    
    return nums


nums = '34722641'

print(nextPermutation(list(nums)))
    
    
    
    
    
    
    
    # Sort Using Custom Alphabet


s = "abcd"
from collector import Counter

sCounter = Counter(s)
    
    
    
