import numpy as np
import time

#Returns an admissable packing from an overweight one
def fix_overweight_fast(items_info, packing, knapsack_capacity):

    # generate sorted indices for every item according to profit/weight
    # only calculated once and stored in global variable
    if 'pw_sorted' not in globals():
        start = time.time()
        pw_unsorted = items_info[:, 0] / items_info[:, 1]
        global pw_sorted
        pw_sorted = np.argsort(pw_unsorted, kind='quicksort')

    # Calculate current weight of packing
    current_weight = np.sum(np.multiply(items_info[:, 1], packing))

    # Remove items in order according to profit/weight ratio until packing is valid
    step = 0
    while current_weight > knapsack_capacity:
        if packing[pw_sorted[step]]:
            packing[pw_sorted[step]] = 0
            current_weight -= items_info[pw_sorted[step]][1]
        step += 1

    # Try to refill the knapsack again according to highest profit/value
    difference = knapsack_capacity - current_weight
    for i in range(0, min(1000, len(pw_sorted))):
        if difference < 75:
            break
        if packing[pw_sorted[len(pw_sorted) - 1 - i]] == 0 and difference > items_info[pw_sorted[len(pw_sorted) - 1 - i]][1]:

            packing[pw_sorted[len(pw_sorted) - 1 - i]] = 1
            difference -= items_info[pw_sorted[len(pw_sorted) - 1 - i]][1]

    return packing

#generates a packing for a given route
# takes into account profit over weight ratio
def ranked_packing(items_info, packing, percentage, knapsack_capacity):
    # generate sorted indices according to profit/weight
    # store it in global variable
    if 'pw_sorted' not in globals():
        pw_unsorted = items_info[:, 0] / items_info[:, 1]
        global pw_sorted
        pw_sorted = np.argsort(pw_unsorted, kind='quicksort')

    # fill knapsack to certain fill percentage
    current_weight=0
    iter=0
    while current_weight< knapsack_capacity*percentage:
        packing[pw_sorted[len(items_info) - 1 - iter]] = 1
        current_weight += items_info[pw_sorted[len(items_info) - 1 - iter]][1]
        iter += 1

    # if the fill percentage is 100 fix the overweight packing
    if percentage==1:
        packing=fix_overweight_fast(items_info, packing, knapsack_capacity)

    return packing
