#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 16:35:08 2024

@author: james
"""





# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 09:17:05 2021

@author: usr006612
"""

#import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta

#%% PCP VISITS

# Query to extract note id from doctor visits

def patientVisitsFunction(list_of_patients, con):

    pat_id_string = "".join(
        map(lambda pat_id: str(pat_id)+", ", list_of_patients))[:-2]

    pat_visits_query = """     
                           select
                           A.patient_id,
                           CONCAT('visit_', CAST(A.visit_id AS CHAR)) AS event_id,
                           LEFT(A.visit_date, 10) as event_date,
                           A.note_id
                           from DASHBOARD_PROD.NB_NOTE_HDR A
                           WHERE A.patient_id in ({pat_id_string})
                       """.format(pat_id_string=pat_id_string)

    return pd.read_sql(pat_visits_query, con=con).sort_values(by=['patient_id',
                                                                   'event_date'], ascending=[1, 1])

#%% HOSPITAL ADMISSIONS

# Query to extract final diagnosis from hospital admissions

def patientAdmissionsFunction(list_of_patients, con):

    pat_id_string = "".join(
        map(lambda pat_id: str(pat_id)+", ", list_of_patients))[:-2]

    pat_admissions_query = """ 
                               select
                               A.patient_id as patient_id,
                               CONCAT('admis_', CAST(A.admission_id AS CHAR)) AS event_id,
                               LEFT(A.admission_date, 10) as event_date,
                               LEFT(A.discharge_date, 10) as discharge_date,
                               RIGHT(B.snomed_icd9_id, LENGTH(B.snomed_icd9_id)-LOCATE('_',B.snomed_icd9_id)) as diag_code
                               from BIDW.hits_admissions A
                                   inner join BIDW.hits_diagnosis B
                               ON A.admission_id = B.admission_id  
                               WHERE A.patient_id in ({pat_id_string}) 
                               AND (A.admission_type_id IN (1,4) OR A.observation = 1)
                               AND A.active = 1
                               AND B.active = 1
                               AND B.is_final=1
                               AND B.snomed_icd9_id is NOT NULL
                           """.format(pat_id_string=pat_id_string)
                           
    result = pd.read_sql(pat_admissions_query, con=con)

    result = result[result['diag_code'].str.match('[A-Z][0-9][0-9]')]

    result['diag_code'] = result['diag_code'].str.replace('.','')                    

    return result.sort_values(by=['patient_id',
                                  'event_date'], ascending=[1, 1])

#%% DOCTOR NOTES

# Query to extract diagnosis from doctor notes

def diag_codesFunction(list_of_notes, con):

    note_id_string = "".join(
        map(lambda note_id: str(note_id)+", ", list_of_notes))[:-2]
    
    diag_codes_query = """     
                           select
                           note_id,
                           REPLACE(diag_code,'.','') as diag_code
                           from DASHBOARD_PROD.NB_VISIT_ICD_AP_DTLS
                           WHERE NOTE_ID in ({note_id_string})                         
                           AND diag_code is NOT NULL
                           AND REGEXP_LIKE(LEFT(diag_code,3),'[A-Z][0-9][0-9]')   
                       """ .format(note_id_string=note_id_string)
                       
    return   pd.read_sql(diag_codes_query, con=con)



# %% Diagnosis & Event Type Filter


def test_cond(diag,
              diag_pttr_tuple):
    cond = False
    for pttr in diag_pttr_tuple:
        if diag.find(pttr) == 0:
            cond = True
            break
    return cond


def ExistEvent(event_id,
               event_date,
               event_diag_codes_set,
               event_type=None,
               present_time=None,
               duration=None,
               diag_pttr_tuple=None):

    type_cond = True
    time_cond = True
    dur_cond = True
    pttr_cond = True
    try:
        if event_type != None:
            type_cond = (event_id[:6] == event_type)

        if present_time != None:
            if duration != None:
                if duration.days > 0:
                    dur_cond = ((event_date > present_time) and
                                (event_date <= (date.fromisoformat(present_time) +
                                                duration).strftime('%Y-%m-%d')
                                  )
                                )
                elif duration.days < 0:
                    dur_cond = ((event_date > (date.fromisoformat(present_time) +
                                               duration).strftime('%Y-%m-%d')
                                  and (event_date <= present_time)
                                  )
                                )
            else:
                time_cond = (event_date <= present_time)

        if diag_pttr_tuple != None:
            pttr_cond = False
            try:
                for diag in  event_diag_codes_set:
                    pttr_cond = test_cond(diag, diag_pttr_tuple)
                    if pttr_cond:
                        break
            except:
                pass

        cond = type_cond and time_cond and dur_cond and pttr_cond

    except:
        cond = False

    return cond


def EventFilter(events, # tuple of event ids
                event_date_dict, # dictionary with event id keys and date values
                event_diag_codes_dict, # dictionary with event id keys and and diag code set values 
                event_type=None,
                present_time=None,
                duration=None,
                diag_pttr_tuple=None):
    try:
        return tuple(filter(lambda event_id: ExistEvent(event_id,
                                                        event_date_dict[event_id],
                                                        event_diag_codes_dict[event_id],
                                                        event_type,
                                                        present_time,
                                                        duration,
                                                        diag_pttr_tuple), 
                            events
                            )
                      )
    except:
        return None
# %% DIURETIC USE

# Query to extract patients using diuretic medication and their administration type


def diuretic_Function(list_of_patients, con):

    pat_id_string = "".join(
        map(lambda pat_id: str(pat_id)+", ", list_of_patients))[:-2]

    diuretic_query = """     
                       select
                       patient_id as patient_id,
                       ORDER_ID,
                       CREATED_DATE,
                       CASE WHEN REGEXP_LIKE(LOWER(CAT_DESCRIPTION),' iv') THEN 'IV'
                            WHEN REGEXP_LIKE(LOWER(CAT_DESCRIPTION),' im') THEN 'IM'
                            WHEN REGEXP_LIKE(LOWER(CAT_DESCRIPTION),' gtt') THEN 'IV'
                            ELSE 'OTHER' END AS admin_type
                       FROM DASHBOARD_PROD.PAT_ORDER_CATEGORY
                       WHERE patient_id in ({pat_id_string})
                       AND (REGEXP_LIKE(LOWER(CAT_DESCRIPTION),'lasix')
                            OR REGEXP_LIKE(LOWER(CAT_DESCRIPTION),'bumetanide')
                            OR REGEXP_LIKE(LOWER(CAT_DESCRIPTION),'bumex') 
                            OR REGEXP_LIKE(LOWER(CAT_DESCRIPTION),'furosemide'))
                       AND STATUS IN ('Performed','Pending Visit','Ordered')
                  """.format(pat_id_string=pat_id_string)

    return pd.read_sql(diuretic_query, con=con)


#%% EJECTION FRACTION 

# Query to extract patients echo results

def ejectionFractionFunction(list_of_patients, con):

    pat_id_string = "".join(
        map(lambda pat_id: str(pat_id)+", ", list_of_patients))[:-2]

    EF_measurements = pd.read_sql("""   SELECT
                                        patient_id as patient_id,
                                        study_dos,
                                        Left_Ventricle_EF_Lower, 
                                        Left_Ventricle_EF_Upper
                                        FROM BIDW.Study_Data_Report
                                        WHERE Study_type = 'Echocardiogram'
                                        AND patient_id in ({pat_id_string}) 
                                  """.format(pat_id_string=pat_id_string),con=con)
                           
    EF_measurements['patient_id'] = pd.to_numeric(EF_measurements['patient_id'], errors='coerce', ).fillna(-1).astype(int)
                           
    EF_measurements['study_dos'] = EF_measurements['study_dos'].astype(str)

    return EF_measurements






# =============================================================================
# Print matrix diagonals
# =============================================================================





















# =============================================================================
# Uniformly Pick Max Integer in Array
# =============================================================================
import random
def find_random_max(nums):
    max_num = max(nums)
    max_index = [i for i, num in enumerate(nums) if num == max_num ]
    return random.choice(max_index)

nums = [1, 3, 5, 7, 7, 6]
print(find_random_max(nums))
















# =============================================================================
# Select Kth Element
# =============================================================================

# =============================================================================
# Set Powerset
# =============================================================================



























# =============================================================================
# Sorted Iterator over K Sorted Lists
# =============================================================================

# Example usage
lists = [[1, 5, 7], [2, 3, 10], [4, 6, 9]]

import heapq

def sort_iterator(lists):
    heap = [] 
    res = []
    
    for index, lst in enumerate(lists):
        heapq.heappush(heap, (lst[0], index, 0)) # value, lst_index, ele_index
    
    while heap:
        val, lst_index, ele_index = heapq.heappop(heap)
        res.append(val)
        if ele_index < len(lists[lst_index]) - 1:
            heapq.heappush(heap, (lists[lst_index][ele_index+1], lst_index, ele_index+1))
    
    return res

print(sort_iterator(lists))




































import heapq

def sorted_iterator(lists):
    heap = []
    # Initialize the heap with the first element from each list
    for i, lst in enumerate(lists):
        if lst:  # Ensure the list is not empty
            heapq.heappush(heap, (lst[0], i, 0))
    
    # Continue until the heap is empty
    while heap:
        val, list_idx, elem_idx = heapq.heappop(heap)
        yield val
        # If there is a next element in the same list, push it onto the heap
        if elem_idx + 1 < len(lists[list_idx]):
            next_val = lists[list_idx][elem_idx + 1]
            heapq.heappush(heap, (next_val, list_idx, elem_idx + 1))


# Example usage
lists = [[1, 5, 7], [2, 3, 10], [4, 6, 9]]

heap = []
resu = []
# Initialize the heap with the first element from each list
for i, lst in enumerate(lists):
    if lst:  # Ensure the list is not empty
        heapq.heappush(heap, (lst[0], i, 0))

# Continue until the heap is empty
while heap:
    val, list_idx, elem_idx = heapq.heappop(heap)
    resu.append(val)
    # If there is a next element in the same list, push it onto the heap
    if elem_idx + 1 < len(lists[list_idx]):
        next_val = lists[list_idx][elem_idx + 1]
        heapq.heappush(heap, (next_val, list_idx, elem_idx + 1))



# Example usage
lists = [[1, 5, 7], [2, 3, 10], [4, 6, 9]]

import heapq

def sorted_iterator(lists):
    heap = []
    
    for index, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst[0], index, 0))
    
    while heap:
        val, list_idx, elem_idx = heapq.heappop(heap)
        if elem_idx + 1 < len(lists[list_idx]):
            next_val = lists[list_idx][elem_idx+1]
            heapq.heappush(heap, (next_val, list_idx, elem_idx + 1))

    return heap


lists = [[1, 5, 7], [2, 3, 10], [4, 6, 9]]

res = sorted_iterator(lists)

























# =============================================================================
# Merge Sorted Arrays
# We have two SORTED arrays of integers: A and B. A has empty slots at the end of
# it. It has exactly as many empty slots as there are elements in B. Your goal is to
# merge the elements from B into A so that array A contains all of the elements in
# sorted order.
# Optimize for speed and memory usage.
# Input:
# A = [1, 2, 3, _, _, _, _]
# B = [2, 4, 6, 100]
# Expected output:
# A = [1, 2, 2, 3, 4, 6, 100]
# =============================================================================

[4, 4, 6, 9, 100]

# two pointer

def merge_arrays2(A, B):
    lenA = len([x for x in A if x is not None])
    lenNone = len(A) - lenA
    lenB = len(B)
    
    indexA = lenA - 1
    indexB = lenB - 1
    mergeindex = len(A) - 1
    count = 0
    while count < lenNone:
        if indexA >= 0 and A[indexA] > B[indexB]:
            A[mergeindex] = A[indexA]
            indexA -= 1
        else:
            A[mergeindex] = B[indexB]
            indexB -= 1
        mergeindex -= 1
        count += 1
    return A

# Input
A = [1, 2, 9, None, None, None]
B = [2, 4, 6, 100]

# Expected Output: A = [1, 2, 2, 3, 4, 6, 100]
print(merge_arrays2(A, B))









# two pointer

def merge_arrays2(A, B):
    lenA = len([x for x in A if x is not None])
    lenA_None = len(A) - lenA
    lenB = len(B)
    
    indexA = lenA - 1
    indexB = lenB - 1
    mergeindex = len(A) - 1
    
    while indexB >= 0 and mergeindex >= 0:
        if indexA >= 0 and A[indexA] > B[indexB]:
            A[mergeindex] = A[indexA]
            indexA -= 1
        else:
            A[mergeindex] = B[indexB]
            indexB -= 1
        mergeindex -= 1
    return A


# Input
A = [1, 2, 9, None, None]
B = [2, 4, 6, 100]

# Expected Output: A = [1, 2, 2, 3, 4, 6, 100]
print(merge_arrays2(A, B))















def merge_arrays(A, B):
    i = j = 0
    res = []
    while i <= len(A) and j <= len(B):
        if i == len(A):
            return res + B[j:]
        elif j == len(B):
            return res + A[i:]
        elif A[i] <= B[j]:
            res.append(A[i])
            i += 1
        else:
            res.append(B[j])
            j += 1

A = [1, 2, 3]
B = [2, 4, 6, 100]

print(merge_arrays(A, B))
























# =============================================================================
# Dot Product Of Sparse Vectors
# =============================================================================



































# =============================================================================
# Add minimum amount of parentheses to make a string of parentheses
# balanced
# =============================================================================

# Input: ((( Output: 3
# Input: ()) Output: 1
# Input: (()) Output: 0
# Input: )( Output: 2


def parenthses_balance2(s):
    left = right = 0
    for char in s:
        if char == "(":
            left += 1 
        elif char == ")":
            if left > 0:
                left -= 1
            else:
                right += 1
    return left + right 


# Test cases
print(parenthses_balance2("((("))  # Output: 3
print(parenthses_balance2("())"))  # Output: 1
print(parenthses_balance2("(())"))  # Output: 0
print(parenthses_balance2(")("))    # Output: 2













def parenthses_balance(s):
    stack = []
    for char in s:
        if char == '(':
            stack.append(char)
        else:
            if stack:
                stack.pop()
            else:
                stack.append(char)
    return len(stack)

s= ')('
print(parenthses_balance(s))

































def removeMakevalide(s):
    stack = [] 
    remove_indices = set()
    
    for index, char in enumerate(s):
        if char == '(':
            stack.append(index)
        elif char == ')':
            if stack:
                stack.pop() 
            else:
                remove_indices.add(index)
    remove_indices = remove_indices.union(set(stack))
    
    result = ''.join(s[i] for i in range(len(s)) if i not in remove_indices)
    return result 
                
                
            



























def lca(root, p, q):
    if root is None:
        return None
    if root == p or root == q:
        return root
    
    left = lca(root.left, p, q)
    right = lca(root.right, p, q)
    
    if left is not None and right is not None:
        return root
    if left is not None:
        return left 
    if right is not None:
        return right 
    
    
    































import heapq

heap = []

numbers = [1, 3, 5, 7, 9, 2, 4, 6, 8, 0]

# top 2 num 
# k =4
# for num in numbers:
#     heapq.heappush(heap, num)
#     if len(heap) > k:
#         heapq.heappop(heap)

# print(heap[0])


# min 2 num 

k = 4

for num in numbers:
    heapq.heappush(heap, -num)
    if len(heap) > k:
        heapq.heappop(heap)

print(-heap[0])

























# =============================================================================
# For example,
# Given [1,1,1,2,2,3] and k = 2, return [1,2].
# Given [1,1,1,1] and k = 1 return [1].
# Given [1,2,3,2,3] and k = 2 return [2,3].
# Given [1,2,3,4] and k = 2 return [1,2].
# 
# 
# =============================================================================
from collections import Counter
import heapq

def op_k_frequent3(nums, k):
    nums_count = Counter(nums)
    heap = []
    
    for num, freq in nums_count.items():
        heapq.heappush(heap, (freq, num))
        if len(heap) > k:
            heapq.heappop(heap)
    
    return [heapq.heappop(heap)[1] for _ in range(k)][::-1]



nums = [1,1,1,2,2,3]
k = 2

# print(op_k_frequent3(nums, k))



nums_count = Counter(nums)
heap = []

for num, freq in nums_count.items():
    heapq.heappush(heap, (freq, num))
    if len(heap) > k:
        heapq.heappop(heap)




import heapq

heap = []
# Suppose we have numbers with their negated frequencies
heapq.heappush(heap, (-3, 10))  # Frequency of 3 for number 10
heapq.heappush(heap, (-1, 5))   # Frequency of 1 for number 5
heapq.heappush(heap, (-2, 8))   # Frequency of 2 for number 8
heapq.heappop(heap)
# The heap will organize these primarily by the first element of each tuple (the negated frequency)
print(heap) 



















# find  top k, can use heap to do that. heap time complexity is O(logk)


from collections import Counter
import heapq

def op_k_frequent2(nums, k):
    nums_count = Counter(nums)
    heap = []
    
    heap_count = 0
    for num, freq in nums_count.items():
        heapq.heappush(heap, (-freq, num))
        heap_count += 1
        if heap_count > k:
            heapq.heappop(heap)
    return heap



nums = [1,1,1,2,2,3]
k = 2

# res = op_k_frequent2(nums, k)

nums_count = Counter(nums)
heap = []

heap_count = 0
for num, freq in nums_count.items():
    heapq.heappush(heap, (freq, num))
    heap_count += 1
    if heap_count > k:
        heapq.heappop(heap)

[heapq.heappop(heap)[1] for _ in range(len(heap))][::-1]




import heapq
from collections import Counter

def top_k_frequent(nums, k):
    # Count the occurrences of each number in nums
    nums_count = Counter(nums)
    
    # Use a heap to keep the top k elements. The heap will store tuples of (-frequency, num)
    # to ensure that the heap is a min-heap based on frequencies (Python's heapq is a min-heap)
    heap = []
    for num, freq in nums_count.items():
        # Push the inverse of frequency to make it a max-heap by frequency
        heapq.heappush(heap, (-freq, num))
        # If the heap size exceeds k, pop the smallest element (which is the least frequent)
        if len(heap) > k:
            heapq.heappop(heap)
    
    # Extract the elements from the heap and return them
    return [heapq.heappop(heap)[1] for _ in range(len(heap))][::-1]

# Example usage
print(top_k_frequent([1,1,1,2,2,3], 2))  # [1,2]





import heapq

# Initialize an empty heap
heap = []

# List of numbers to add to the heap
numbers = [1, 3, 5, 7, 9, 2, 4, 6, 8, 0]

k =4
# Push numbers onto the heap as negatives to simulate a max heap
for number in numbers:
    heapq.heappush(heap, -number)
    if len(heap) > k:
        heapq.heappop(heap)


min_heap = []
k = 4
# Use a min heap to keep track of the top k largest elements
for number in numbers:
    heapq.heappush(min_heap, number)  # Push number directly into min_heap
    if len(min_heap) > k:  # If heap size exceeds k, remove the smallest element
        heapq.heappop(min_heap)
        
        
        
        
# Pop numbers from the heap and negate them to get the original values
res=  []
while heap:
    max_value = -heapq.heappop(heap)
    res.append(max_value)
print(res)




import heapq

# Initialize an empty heap
heap = []

# List of numbers to add to the heap
numbers = [1, 3, 5, 7, 9, 2, 4, 6, 8, 0]

# Push numbers onto the heap as negatives to simulate a max heap
for number in numbers:
    heapq.heappush(heap, number)

# Pop numbers from the heap and negate them to get the original values
res=  []
while heap:
    max_value = heapq.heappop(heap)
    res.append(max_value)
print(res)




# print(op_k_frequent2(nums, k))


# # Use a heap to keep the top k elements. The heap will store tuples of (-frequency, num)
# # to ensure that the heap is a min-heap based on frequencies (Python's heapq is a min-heap)
# heap = []
# for num, freq in nums_count.items():
#     # Push the inverse of frequency to make it a max-heap by frequency
#     heapq.heappush(heap, (-freq, num))
#     # If the heap size exceeds k, pop the smallest element (which is the least frequent)
#     if len(heap) > k:
#         heapq.heappop(heap)
        
            



# def op_k_frequent(nums, k):
#     nums_count = {}
    
#     for num in nums:
#         if num in nums_count:
#             nums_count[num] += 1
#         else:
#             nums_count[num] = 1
            
#     return list(dict(sorted(nums_count.items(), key=lambda x: x[1], reverse=True)).keys())[:2]



# nums = [1,2,3,4]
# k = 2
# print(op_k_frequent(nums, k))









# =============================================================================
# 
# =============================================================================



def customSort(input_str, order):
    char_count = {}
    for char in input_str:
        if char in char_count:
            char_count[char] += 1
        else:
            char_count[char] = 1

    output = ''
    
    for char in order:
        if char in char_count:
            output += char * char_count[char]
            del char_count[char]
    
    for char, count in char_count.items():
        output += char*count 
        
    return output



input_str =  'abcadcb555'

order = 'bac'

print(customSort(input_str, order))




# =============================================================================
# 
# =============================================================================
























# %% VITALS

# Query to extract patient monthly average vital value


def vitalsFunction(vital_desc, patients_and_range_date, con):

    name = vital_desc.replace(' ', '_')
    
    row = 'ROW({patient_id},"{present_date}","{earliest_month_date}")'

    values = ",".join(patients_and_range_date[['patient_id',
                                               'present_date',
                                               'earliest_month_date']].apply(lambda x: row.format(patient_id = x[0], 
                                                                                                  present_date = x[1],
                                                                                                  earliest_month_date = x[2]),
                                                                                                  axis=1).tolist())
                                                                       
    CTE = "(SELECT v.* FROM (VALUES {}) v(patient_id,present_date,earliest_month_date))".format(values)

    vital_query = """  WITH cte AS
                         {cte}
                       select
                       cte.patient_id as patient_id,
                       cte.present_date,
                       YEAR(A.Result_Date)*100 + MONTH(A.Result_Date) as yyyymm,
                       AVG(A.Test_Value) as {name}
                       from cte
                           LEFT JOIN BIDW.lab_and_vitals A
                       ON cte.patient_id = A.dash_patient_id
                           AND A.Result_Date BETWEEN cte.earliest_month_date AND cte.present_date
                       WHERE REGEXP_LIKE(LOWER(A.Test_Description),'{vital_desc}')
                       GROUP BY
                       cte.patient_id,
                       cte.present_date,
                       yyyymm
                  """.format(cte=CTE,
                             vital_desc=vital_desc,
                             name=name)

    return pd.read_sql(vital_query, con=con).sort_values(by=['patient_id',
                                                             'present_date',
                                                             'yyyymm'], ascending=[1, 1, 0])


def bpFunction(patients_and_range_date, con):

    row = 'ROW({patient_id},"{present_date}","{earliest_month_date}")'
    
    values = ",".join(patients_and_range_date[['patient_id',
                                               'present_date',
                                               'earliest_month_date']].apply(lambda x: row.format(patient_id = x[0], 
                                                                                                  present_date = x[1],
                                                                                                  earliest_month_date = x[2]),
                                                                                                  axis=1).tolist())
                                                                       
    CTE = "(SELECT v.* FROM (VALUES {}) v(patient_id,present_date,earliest_month_date))".format(values)

    bp_query = """  WITH cte AS
                         {cte}
                    select
                    cte.patient_id,
                    cte.present_date,
                    YEAR(A.Result_Date)*100 + MONTH(A.Result_Date) as yyyymm,
                    AVG(CAST(LEFT(A.Test_Value, LOCATE('/',A.Test_Value)-1) AS float)) as bp_sys,
                    AVG(CAST(RIGHT(A.Test_Value, LENGTH(A.Test_Value)-LOCATE('/',A.Test_Value)) AS float)) as bp_dia
                    from cte
                        LEFT JOIN BIDW.lab_and_vitals A
                    ON cte.patient_id = A.dash_patient_id
                        AND A.Result_Date BETWEEN cte.earliest_month_date AND cte.present_date
                    and REGEXP_LIKE(LOWER(A.Test_Description),'bp')
                    GROUP BY
                    cte.patient_id,
                    cte.present_date,
                    yyyymm
                """.format(cte=CTE)

    return pd.read_sql(bp_query, con=con).sort_values(by=['patient_id',
                                                          'present_date',
                                                          'yyyymm'], ascending=[1, 1, 0])
#%% GENDER AND DOB

def dob_and_genderFunction(pat_and_present_date , con):
    
    row = 'ROW({patient_id},"{present_date}")'
    
    pat_and_present_date['rows'] = pat_and_present_date[['patient_id',
                                                         'present_date']].apply(lambda x: 
                                                                                row.format(patient_id = x[0],
                                                                                           present_date=x[1]), 
                                                                                axis=1)                                                     

    values = ",".join(pat_and_present_date['rows'].tolist())
                                                                       
    CTE = "(SELECT v.* FROM (VALUES {}) v(patient_id,present_date))".format(values)

    gender_query = """
                   WITH cte AS
                       {}
                   SELECT 
                   cte.patient_id,
                   cte.present_date,
                   A.gender,
                   A.DATE_OF_BIRTH as dob
                   FROM cte
                       LEFT JOIN DASHBOARD_PROD.PATIENT A 
                   ON cte.patient_id = A.patient_id   
                """.format(CTE)

    gender = pd.read_sql(gender_query, con=con).groupby(['patient_id',
                                                         'present_date']).first().reset_index()
    
    gender['dob'] = gender['dob'].astype(str)
    
    gender['dob'] = gender['dob'].apply(lambda dt: dt[:10])
    
    return gender                 



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
    
    
    
    
    
    
