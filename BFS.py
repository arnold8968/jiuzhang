#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 21:33:54 2023

@author: james
"""

def BFS(graph, start, end):
    queue = []
    visited = []
    queue.append([start])
    visited.add(start)
    
    while queue:
        node = queue.pop()
        visited.add(node)
        
        process(node)
        nodes = generate_related_nodes(node)
        queue.push(nodes)




visited = set()

def dfs(node, visited):
    visited.add(node)
    for next_node in node.children():
        if not next_node in visited:
            dfs(next_node, visited)




# =============================================================================
# create a queue Q 

# mark v as visited and put v into Q 

# while Q is non-empty 

#     remove the head u of Q 

#     mark and enqueue all (unvisited) neighbors of u
# =============================================================================

graph = {
  '5' : ['3','7'],
  '3' : ['2', '4'],
  '7' : ['8'],
  '2' : [],
  '4' : ['8'],
  '8' : []
}


visited = []
queue = []

def bfs(graph, visited, note):
    visited.append(note)
    queue.append(note)
    
    while queue:
        m = queue.pop()
        for i in graph[m]:
            if i not in visited:
                visited.append(i)
                queue.append(i)

res = bfs(graph, visited, '5')


























# visited = [] # List for visited nodes.
# queue = []     #Initialize a queue




# def bfs(visited, graph, node): #function for BFS
#   visited.append(node)
#   queue.append(node)

#   while queue:          # Creating loop to visit each node
#     m = queue.pop(0) 
#     print (m, end = " ") 

#     for neighbour in graph[m]:
#       if neighbour not in visited:
#         visited.append(neighbour)
#         queue.append(neighbour)


# bfs(visited, graph, '5')    # function calling