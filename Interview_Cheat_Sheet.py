#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 21:48:37 2023

@author: james
"""

# =============================================================================
# # Binary search
# 
# 排列数组（30-40%是二分法）
# 当面试官让你找一个比O(n）更小的时间复杂度算法的时候（99%）
# 找到一个数组的分割位置， 是的左半部分满足某个条件， 右边部分不满足（100%）
# 找到一个最大、最小的值使得某个条件被满足（90%）
# =============================================================================


def binary_search(self, nums, target):
    # corner case
    # nums is None or len(nums) == 0
    if not nums:
        return -1
    start, end = 0, len(nums) - 1
    # 用 start + 1<end 而不是start<end 的目的是为了避免死循环
    #在 first position of target 的情况下不会出现死循环
    # 但是在last position of target 的情况下会出现死循环
    # 样例 nums=[1,1] target = 1
    # 为了统一模板，我们就都采用start+1<end, 就保证不会出现死循环
    while start + 1 < end:
        # Python 没有overflow的问题， 直接 //2 就可以
        # java 和 C++ 最好写成 mid = start + (end - start) / 2
        # 防止在 start = 2^31-1, end = 2^31-1 的情况下出现加法 overflow
        mid = (start + end) // 2
        # >, =, < 的逻辑先分开写， 然后再看看=的情况是否能合并到其他的分支里面
        if nums[mid] < target:
            start = mid
        elif nums[mid] == target:
            end = mid
        else:
            end=mid
    # 因为上面的循环退出条件是 start + 1 < end
    # 因此这里循环结束的时候， start 和 end 的关系是相邻关系 (1 和 2， 3 和 4 这种)
    # 因此需要再单独判断 start 和 end 这两个数谁是我们要的答案
    # 如果是找到first position of target 就先看 start， 否则就先看 end
    if nums[start] == target:
        return start
    if nums[end] == target:
        return end
    return -1






# =============================================================================
"""
search insert position 

[1,3,5,6], insert: 5: result: 2
[1,3,5,6], insert: 2: result: 1
[1,3,5,6], insert: 0: result: 0

"""



def searchInsert(nums, target):
    if not nums:
        return -1
    
    start, end = 0, len(nums)-1
    
    while start + 1 < end:
        mid = (start + end) // 2
        if nums[mid] < target:
            start = mid 
        else:
            end = mid 
    
    print(start, end)
    if nums[start] == target or nums[start] > target:
        return start 
    
    if nums[end] >= target and nums[start] < target:
        return end 
    
    if nums[end] < target:
        return end+1
    
nums = [1,3,5,6]
target = 8
result = searchInsert(nums, target)








# =============================================================================
"""
Binary search example:
search in a big sorted array 
the array is so big so that you can not get the length of the whole array 
directly, and you can only access the kth number by ArrayReader.get(k) function
Your algotithm should be in O(log k)
"""

def searchBigSortedArray(ArrayReader, target):
    # get the right boundary
    
    count = 1
    while ArrayReader.get(count-1) < target:
        count = count * 2
    
    start = count / 2 
    end = count - 1
    
    while start + 1 < end:
        mid = (start + end) // 2
        if ArrayReader.get(mid) < target:
            start = mid 
        else:
            end = mid
    
    if ArrayReader.get(start) == target:
        return start 
    
    if ArrayReader.get(end) == target:
        return end 
    
    return -1







# =============================================================================
"""
33. Search in Rotated Sorted Array
Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4

经常考
"""

def search(A, target):
    start, end = 0, len(A) - 1
    
    while start + 1 < end:
        mid = (start + end) // 2
        if A[mid] == target:
            return mid 
        
        if A[start] < A[mid]:
            if A[start] <= target and target <= A[mid]:
                end = mid
            else:
                start = mid
        
        else:
            if A[mid] <= target and target <= A[end]:
                start = mid
            else:
                end = mid
    
    if A[start] == target:
        return start 
    
    if A[end] == target:
        return end
    
    return -1







# =============================================================================

# 需要继续复习理解
# leetcode 4

"""
median of two sorted arrays
A = [1,2,3,4,5,6], B = [2,3,4,5]. the median is 3.5
A = [1,2,3], B = [4,5]. the median is 3

"""


# define helper function
# find kth number of two sorted array

def findKth(A, A_start, B, B_start, k):
    if A_start >= len(A):
        return B[B_start + k - 1]
    if B_start >= len(B):
        return A[A_start + k - 1]

    if k == 1:
        return min(A[A_start], B[B_start])

    A_key = A[A_start+k/2-1] if A_start + k / 2 -1 < len(A) else float('inf')
    
    B_key = B[B_start+k/2-1] if B_start + k / 2 -1 < len(B) else float('inf')
    

    if A_key < B_key:
        return findKth(A, A_start + k /2, B, B_start, k-k/2)
    else:
        return findKth(A, A_start, B, B_start + k / 2,  k-k/2)







# =============================================================================
"""
# recover rotated sorted array

[4,5,1,2,3] --> [1,2,3,4,5]


三步翻转法
4,5,1,2,3
4,5   1,2,3
5,4   3,2,1
1,2,3,4,5

"""


def recoverRotatedSortedArray(nums):
    split_position = find_split(nums)
    if split_position == len(nums) - 1:
        return 
    
    swap(nums, 0, split_position)
    swap(nums, split_position, len(nums))
    
    nums.reverse()
    return 


def find_split(nums):
    if nums is None or len(nums) < 2:
        return 0
    
    for i in range(1, len(nums)):
        if nums[i] < nums[i-1]:
            return i
    return i

def swap(nums, start, end):
    if start == end:
        return nums 
    
    left, right = start, end - 1
    while left < right:
        nums[left], nums[right] = nums[right], nums[left]
        left += 1
        right += 1
















# =============================================================================
# # 双指针
# 
# 滑动窗口（90%）
# 时间复杂度要求O(n)（80%是双指针）
# 要求原地操作， 只可以使用交换， 不能使用额外空间（80%）
# 有子数组 subarray、 子字符串 substring 的关键词（50%）
# 有回文 palindrome关键词 （50%）
# =============================================================================



# 1， 相向双指针 （ patition of quicksort)

def partition(A, start, end):
    if start > end:
        return 
    
    left, right = start, end
    # key point 1: pivot is the value, not the index 
    pivot = A[(start + end) // 2]
    # key point 2: every time you compare left & right, it should be 
    # left <= right not left < right
    
    while left <= right:
        while left <= right and A[left] < pivot:
            left += 1
        while left <= right and A[right] > pivot:
            right -= 1
        if left <= right:
            A[left], A[right] = A[right], A[left]
            left += 1
            right -= 1
            
    return left
            
            

# A = [12,9,7,15,10]
# start=0
# end = len(A)-1
# partition(A, start, end)



# 2, 背向双指针

def partition2(s, position, something):
    left = position
    right = position + 1
    
    while left >= 0 and right < len(s):
        if left == something and right == something: # 可以停下来:
            break
        left -= 1
        right += 1
        
        
        
# 3, 同向双指针
"""
def partition3(s):
    j = 0
    for i in range(n):
        # 不满足则循环到满足搭配为止
        while j < n and i 到 j 之间不满足条件:
            j += 1
        if i 到 j 之间满足条件:
            处理i 到j 这段区间

"""



# 4, 合并双指针

def merge(list1, list2):
    new_list = []
    i, j = 0, 0
    
    # 合并的过程只能操作i， j 的移动， 不要去用 list.pop(0) 之类的操作
    # 因为 pop（0） 是O(n) 的时间复杂度
    
    while i < len(list1) and j < len(list2):
        if list1[i] < list2[j]:
            new_list.append(list1[i])
            i += 1
        else:
            new_list.append(list2[j])
            j += 1
            
        # 合并剩下的数列到new_list 里面
        # 不要用new_list.extend(list1[i:])之类的方法
        # 因为list1[i:] 会产生额外空间耗费
        
    while i < len(list1):
        new_list.append(list1[i])
        i += 1
    
    while j < len(list2):
        new_list.append(list2[j])
        j += 1
    
    return new_list
        


# =============================================================================
#     
# =============================================================================



# =============================================================================
# # 排序算法 Sorting
# 
# 复杂度
# 
# 时间复杂度
# 快速排序 期望复杂度 O(nlogn)
# 归并排序 最坏复杂度 O(nlogn)
# 
# =============================================================================



# quick sort

class Solution:
    # @param {int[]} A an integer array
    # @return  nothing 

    def sortIntegers(self, A):
        
        self.quickSort(A, 0, len(A) - 1)
        
    
    def quickSort(self, A, start, end):
        if start >= end:
            return 
        
        left, right = start, end
        
        #key point 1: piovt is the value, not the index
        pivot = A[(start + end) // 2]
        
        #key point 2: every time you compare left & right, it should be
        # left <= right not left < right
        
        while left <= right:
            while left <= right and A[left] < pivot:
                left += 1
            
            while left <= right and A[right] > pivot:
                right -= 1
                
            if left <= right:
                A[left], A[right] = A[right], A[left]
                
                left += 1
                right -= 1
                
        self.quickSort(A, start, right)
        self.quickSort(A, left, end)
        
        return A
        
        
nums = [12, 9, 7, 10, 15]
solu = Solution()

test = solu.sortIntegers(nums)


# =============================================================================

# merge sort

class Solution:
    def sortIntegers(self, A):
        if not A:
            return A
        
        temp = [0] * len(A)
        self.merge_sort(A, 0, len(A) - 1, temp)
        
    
    def merge_sort(self, A, start, end, temp):
        if start >= end:
            return 
        
        # deal with left part
        self.merge_sort(A, start, (start + end) // 2, temp)
        # deal with right part
        self.merge_sort(A, (start + end) // 2 + 1, end, temp)
        # merge and sort array
        self.merge(A, start, end, temp)
        
    def merge(self, A, start, end, temp):
        middle = (start + end) // 2
        left_index = start 
        right_index  = middle + 1
        index = start 
        
        while left_index <= middle and right_index <= end:
            if A[left_index] < A[right_index]:
                temp[index] = A[left_index]
                index += 1
                left_index += 1
                
            else:
                temp[index] = A[right_index]
                index += 1
                right_index += 1
                
        while left_index <= middle:
            temp[index] = A[left_index]
            index += 1
            left_index += 1
            
        while right_index <= end:
            temp[index] = A[right_index]
            index += 1
            right_index += 1
            
        for i in range(start, end+1):
            A[i] = temp[i]



# =============================================================================
# =============================================================================
# 
# # Binary Tree Divide and Conquer
# 
# 二叉树分治
# 
# 二叉树相关的问题
# 可以一分为二去分别处理之后再合并结果
# 数组相关的问题
# 
# 时间复杂度 O(n)


# 碰到二叉树的问题， 就想想在整棵树在该问题上的结果
# 和左右儿子在该问题上的结果之间有什么联系

# =============================================================================

"""
# Preorder, Postorder, Inorder

# Version 1: Traverse
递归
1. 初始定义
2. 递归函数
"""




def preorderTraversal(root):
    result = []
    traverse(root, result)
    return result

# 把root为根的前序遍历放到result里面
def traverse(root, result):
    if not root:
        return result
    
    result.add(root.val)
    traverse(root.left, result)
    traverse(root.right, result)



# -------------------------------------

# Version 2: Divide and Conquer

def preorderTraversal2(root):
    result = []
    
    if not root:
        return result
    
    # Divide 
    left = preorderTraversal2(root.left)
    right = preorderTraversal2(root.right)
    
    # Conquer
    result.append(root.val)
    result.append(left)
    result.append(right)
    
    return result




# =============================================================================

"""
leetcode maximum depth of binary tree

root = [3,9,20,null,null,15,7]
maximum depth = 3
"""

# traverse

def maxDepth(root):
    max_depth = 0
    traverse(root, 1, max_depth)
    return result 

def traverse(root, depth, max_depth):
    if not root:
        return depth
    
    max_depth = max(max_depth, depth)
    traverse(root.left, depth + 1)
    traverse(root.right, depth + 1)
    
    

#-------------------------------------------

# divide and conquer

def maxDepth2(root):
    if not root:
        return 0
    
    leftDepth = maxDepth2(root.left)
    rightDepth = maxDepth2(root.right)
    
    return max(leftDepth, rightDepth) + 1





# =============================================================================

"""
leetcode balanced binary tree
110

Input: root = [3,9,20,null,null,15,7]
Output: true

"""









# =============================================================================

"""
leetcode
lowest common ancestor

在root为根的二叉树中找A， B的LCA
如果找到了 就返回这个LCA
如果只找到n1， 就返回n1
如果只找到n2， 就返回n2
如果都没有就返回null
"""

def getAncesstor(root, node1, node2):
    if not root:
        return None

    if root == node1 or root == node2:
        return root
    
    # divide
    left = getAncesstor(root.left, node1, node1)
    right = getAncesstor(root.right, node1, node2)
    
    # conquer
    if left != None and right != None:
        return root 
    
    if left != None:
        return left 
    
    if right != None:
        return right 
    
    return None 




# =============================================================================
# 二叉搜索树非递归 BST Iterator
# 使用条件
# 用非递归的方式 Non-recursion / iteration 实现二叉树的中序遍历
# 常用于 BST 但不仅仅可以用于 BST

# =============================================================================


def inorder_traversal(root,TreeNode):
    if not root:
        return []
    
    # create a dummy note, right pointer to root
    # put into stack, dummy at the top of the stack
    # is the iterator's position
    
    dummy = TreeNode(0)
    dummy.right = root
    stack = [dummy]
    
    inorder = []
    # move iterator to next point every time
    # which is adjest stack to make stack to next point
    
    while stack:
        node = stack.pop()
        
        if node.right:
            node = node.right 
            
            while node:
                stack.append(node)
                node = node.left 
        
        if stack:
            inorder.append(stack[-1])
    
    return inorder 






# =============================================================================
# 宽度优先搜索BFS
#
#
#
# 使用条件
# 拓扑排序
# 出现连通块的关键词
# 分层 遍历
# 简单图最短路径
# 给定一个变换规则，从初始状态变到终止状态最少几步
# =============================================================================
import collections

def bfs(start_node):
    """
    # BFS 必须要用队列 queue，别用栈 stack!
    
    # distance(dict) 有两个作用，一个是记录一个点是否被丢进过队列了，避免重复访问
    
    # 另外一个是记录 start_node 到其他所有节点的最短距离
    
    # 如果只求连通性的话，可以换成 set 就行
    
    # node 做 key 的时候比较的是内存地址
    """
    
    queue = collections.deque([start_node])
    distance = {start_node:0}
    
    # while 队列不空，不停的从队列里拿出一个点，拓展邻居节点放到队列中
    
    while queue:
        node = queue.popleft() 
        # 如果有明确的终点可以在这里加终点的判断
        
        if node: # is the end
            break  #or return something
        
        for neighbor in node.get_neighbors():
            if neighbor in distance:
                continue
            
            queue.append(neighbor)
            
            distance[neighbor] = distance[node] + 1
            
    # 如果需要返回所有点离起点的距离，就 return hashmap
    return distance

# # 如果需要返回所有连通的节点, 就 return HashMap 里的所有点
# return distance.keys()


# # 如果需要返回离终点的最短距离
# return distance[end_node]



            