#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import numpy as np
import time


def fitness_function_new(route, packing, assigned_city,  distance_matrix, items_info, knapsack_capacity, max_speed, min_speed):

        # Calculates the profit
        profit = np.sum(np.multiply(items_info[:, 0], packing))

        # Calculate speeds at each city based on the route, packing, and knapsack capacity
        speeds = calculate_speed_new2(route, packing, items_info, assigned_city, knapsack_capacity, max_speed, min_speed)

        travel_time = 0

        # calculate total travel time
        for i in range(len(route)):
            travel_time+=distance_matrix[route[i]-1][route[(i+1) % len(route)]-1]/speeds[i]
        fitness=np.array([-profit, travel_time])

        return fitness


def calculate_speed_new2(route, packing, items_info, assigned_city, knapsack_capacity, max_speed, min_speed):

    # pre-calculates weights per city outside the loop
    weights_per_city=dict(zip(route, np.zeros(len(route))))

    # only keep items which are being packed
    temp=np.multiply(assigned_city, packing)

    # adds item weights to corresponding city
    for i, item in enumerate(temp):
        if item > 0:
            weights_per_city[item] += items_info[i][1]

    # return the speed with the formula given in the instructions
    return np.subtract(max_speed, np.multiply(np.divide(np.cumsum(np.fromiter(weights_per_city.values(), dtype=float)), knapsack_capacity), (max_speed - min_speed)))
