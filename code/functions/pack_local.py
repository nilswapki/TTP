# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 11:55:24 2023

@author: dell
"""

def initialize_solution(item_weights, item_values, max_weight):
    """Generate an initial solution based on value-to-weight ratio."""
    n = len(item_weights)
    item_ratios = [(item_values[i] / item_weights[i], i) for i in range(n)]
    item_ratios.sort(reverse=True)

    solution = [0] * n
    total_weight = 0
    for _, i in item_ratios:
        if total_weight + item_weights[i] <= max_weight:
            solution[i] = 1
            total_weight += item_weights[i]

    return solution

def calculate_weight_and_value(solution, item_weights, item_values):
    """Calculate the total weight and value of the current solution."""
    total_weight = sum(w * s for w, s in zip(item_weights, solution))
    total_value = sum(v * s for v, s in zip(item_values, solution))
    return total_weight, total_value

def local_search_knapsack(item_weights, item_values, max_weight, max_iterations=1000):
    """Perform a local search to optimize the knapsack solution."""
    n = len(item_weights)
    current_solution = initialize_solution(item_weights, item_values, max_weight)
    current_weight, current_value = calculate_weight_and_value(current_solution, item_weights, item_values)

    for _ in range(max_iterations):
        # Find the item with the lowest value-to-weight ratio in the solution
        lowest_ratio_item = None
        lowest_ratio = float('inf')
        for i in range(n):
            if current_solution[i] == 1:
                ratio = item_values[i] / item_weights[i]
                if ratio < lowest_ratio:
                    lowest_ratio = ratio
                    lowest_ratio_item = i

        # Find the item with the highest value-to-weight ratio not in the solution
        highest_ratio_item = None
        highest_ratio = 0
        for i in range(n):
            if current_solution[i] == 0:
                ratio = item_values[i] / item_weights[i]
                if ratio > highest_ratio:
                    highest_ratio = ratio
                    highest_ratio_item = i

        # Try to swap these items if it improves the solution
        if lowest_ratio_item is not None and highest_ratio_item is not None:
            new_solution = current_solution.copy()
            new_solution[lowest_ratio_item] = 0
            new_solution[highest_ratio_item] = 1
            new_weight, new_value = calculate_weight_and_value(new_solution, item_weights, item_values)

            if new_weight <= max_weight and new_value > current_value:
                current_solution = new_solution
                current_weight, current_value = new_weight, new_value

    return current_solution, current_value

# Example items
item_weights = [10, 20, 30, 40, 50]
item_values = [60, 100, 120, 140, 160]
max_weight = 100

# Run the local search algorithm
knapsack_solution, knapsack_value = local_search_knapsack(item_weights, item_values, max_weight)
knapsack_solution, knapsack_value

