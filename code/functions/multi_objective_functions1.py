import math
import random

import numpy as np
from scipy.spatial.distance import euclidean


# tests if no element of A is worse than B
def weaklyDominates(vectorA, vectorB):
    for i in range(len(vectorA)):
        if vectorB[i] < vectorA[i]:
            return False
    return True


# tests if A weakly dominates B and A and B are not equal
# this one will be mainly used
def dominates(vectorA, vectorB):
    if not (weaklyDominates(vectorA, vectorB)):
        return False

    if np.array_equal(vectorA, vectorB):
        return False
    else:
        return True


# tests if A dominates B and every element of A is better than that of B
# not in use right now
def strictlyDominates(vectorA, vectorB):
    if not (dominates(vectorA, vectorB)):
        return False

    for i in range(len(vectorA)):
        if vectorB[i] <= vectorA[i]:
            return False

    return True


# tests if a point does not get dominated by any other point in a set
def non_dominated(point, data_points):
    non_dominated = False
    for p in data_points:
        if dominates(p, point):
            break
    else:
        non_dominated = True

    return non_dominated


#calcultes if a point is calculated by any other point in the given set
# and returns a boolean value
def non_dominated_new(point, data_points):
    equal_ind = np.not_equal(point, data_points)
    if np.any(np.less_equal(data_points[:, 0], point[0]) * np.less_equal(data_points[:, 1], point[1]) * (
    equal_ind[:, 0], equal_ind[:, 1])):
        return False
    return True


# sorts the multidimensional entries and returns either the n best entries or the first pareto front
def non_dominated_sorting(data_points, n, first_front_only):  # array of shape [[-profit, time], [-profit, time], ...]
    # assures n is not out of bounds
    if n > len(data_points):
        n = len(data_points)

    # so that full first front is returned, no matter which n was specified
    if first_front_only:
        n = len(data_points)

    # the points from potentially multiple fronts that will be returned
    fronts = np.zeros(shape=(1, 2))

    while len(fronts) < n + 1:  # as long as there are not enough points gathered

        # remove already calculated fronts from set
        data_points = data_points[np.all(np.any((data_points - fronts[:, None]), axis=2), axis=0)]

        # checks for every remaining point if it belongs to new pareto front
        for point in data_points:

            if non_dominated_new(point, data_points):
                fronts = np.row_stack([fronts, point])

        # if only first pareto front is wanted, terminate loop after one run
        if first_front_only:
            break
    fronts = fronts[1:]
    # returns the n best points, first all from the first front, then second, and so on, until n is reached
    if len(fronts) <= n:
        return fronts
    # returns random subset of found fronts (in case first_front_only == False)
    # could potentially pick e.g. pick a point from 2rd PF instead of 1st PF (good for exploration?)
    else:
        random_indices = np.random.randint(len(fronts), size=n)
        return fronts[random_indices, :]


def tournament(fitness_values, n):
    # returns non-dominated point out of set of randomly chosen points
    random_index = np.random.randint(len(fitness_values), size=n)
    point = non_dominated_sorting(fitness_values[random_index, :], 1, False)

    # checks the index of given point
    for i in range(len(fitness_values)):
        if (fitness_values[i].reshape(1, 2) == point).all():
            return i  # returns index of individual that won the tournament
    return -1  # in case something went wrong


# select most evenly spaced n points from given pareto set
def getBestPareto(pf, n):
    pf = pf[pf[:, 0].argsort()]  # sort pareto front (does not matter for which objective)

    if len(pf) <= n:  # if given pareto front is smaller already
        return pf

    # as long is the pareto front does not have desired size, delete points from it
    while len(pf) > n:
        min_distance = np.inf
        min_index = -1

        # finds the two neighboring points with closest distance
        for i in range(len(pf) - 1):
            dist = euclidean(pf[i].flatten(), pf[i + 1].flatten())
            if dist < min_distance:
                min_distance = dist
                min_index = i

        # one of the two closest points gets deleted
        # point containing part of the nadir/ideal (first or last point in the front) should stay
        if min_index == 0:
            pf = np.delete(pf, min_index + 1, 0)
        else:
            pf = np.delete(pf, min_index, 0)

    return pf

# returns the worst individual from the population
#returns individual from last front with the smallest crowding distance
def get_worst(fitness_values):
    # Generates the last front
    last_front = np.array([])
    current_fit = fitness_values
    while len(current_fit) > 0:
        i = 0
        non_dom_ind = np.zeros(len(current_fit))

        for ind in current_fit:
            if non_dominated_new(ind, current_fit):
                non_dom_ind[i] = 1
            i += 1
        last_front = current_fit
        current_fit = current_fit[non_dom_ind == 0]


    # Chooses the worst individual from the last front according to the crowding distance
    if len(last_front) == 1:
        return last_front.flatten()

    if len(last_front) ==2:
        return last_front[np.random.randint(0, 2)]

    distance = np.ones(len(last_front)) * math.inf

    iter = -1
    for ind in last_front:
        iter += 1

        # Determine distances to the two nearest neighbors
        temp = np.copy(last_front)
        temp[:, 0] -= ind[0]
        temp[:, 1] -= ind[1]

        temp1 = np.clip(np.copy(temp)[:, 0], a_min=0, a_max=None)
        temp2 = np.nonzero(temp1)
        if len(temp2[0])==0:
            distance[iter] = math.inf
            continue
        x1 = np.min(temp1[temp2])

        temp1 = np.clip(np.copy(temp)[:, 0], a_min=None, a_max=0)
        temp2 = np.nonzero(temp1)
        if len(temp2[0])==0:
            distance[iter] = math.inf
            continue
        x2 = np.max(temp1[temp2])

        temp1 = np.clip(np.copy(temp)[:, 1], a_min=0, a_max=None)
        temp2 = np.nonzero(temp1)
        if len(temp2[0])==0:
            distance[iter] = math.inf
            continue
        y1 = np.min(temp1[temp2])

        temp1 = np.clip(np.copy(temp)[:, 1], a_min=None, a_max=0)
        temp2 = np.nonzero(temp1)
        if len(temp2[0])==0:
            distance[iter] = math.inf
            continue
        y2 = np.max(temp1[temp2])

        # Calculate manhatten distance to the two closest points
        distance[iter] = math.sqrt(x1 ** 2 + y1 ** 2) + math.sqrt(x2 ** 2 + y2 ** 2)

    # Return fitness value of worst individual
    return last_front[np.argmin(distance)]


if __name__ == '__main__':
    data_points = np.array([[1, 1], [0.5, 1.1], [0.2, 2], [1.3, 0.9], [2, 0.7],
                            [5, 3], [3, 5], [10, 100], [100, 10], [200, 200]])  # contains four fronts
    data_points1 = np.array([[1, 1], [0.5, 1.1], [0.2, 2], [1.3, 0.9], [2, 0.7]])
    # check that it works as intended
    # print(non_dominated_sorting(data_points, 2, True))
    # print(getBestPareto(non_dominated_sorting(data_points, 10, True), 3))
    # print(tournament(data_points, 10))  # for n=10 it should only return indices between 0-4 --> works

#print(get_worst(data_points))
