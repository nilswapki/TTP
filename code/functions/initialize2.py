import numpy as np
from functions.fix_overweight import fix_overweight_fast as fix_overweight
from functions.fix_overweight import ranked_packing
from functions.local_search_tsp import two_opt

def initialize_population(num_cities, num_individuals, distance_matrix, items_info, knapsack_capacity, assigned_city, parameters):
    routes = []
    packings = []

    tsp_nn = int(num_individuals * parameters['percentage_NN'])
    num_routes = num_individuals - tsp_nn
    better_packing=0

    if parameters['ranked_packing']:
        better_packing = parameters['num_improved_packing']


    routes = tsp_nearest_neighbor(num_cities, tsp_nn, distance_matrix)

    for _ in range(num_routes):
        city_order = np.array(np.insert(np.random.permutation(range(2,num_cities+1)),0,1))
        if len(routes) != 0:
            routes = np.vstack([routes, city_order])
        else:
            routes.append(city_order)
    if parameters['local_search']:
        iter=0
        for route in routes:
            routes[iter]=two_opt(route, distance_matrix, parameters['local_iterations'])
            iter+=1

    iter=0
    for perc in np.linspace(0, 1, better_packing):
        packings.append(packing_sorted(routes[iter], distance_matrix, items_info, knapsack_capacity, parameters['distance_factor'], assigned_city, perc))
        iter+=1

    counter = 0
    for _ in range(num_individuals-parameters['zero_packing']-better_packing):
        # Generate a random permutation of num_cities, excluding city 0 as the starting point

        valid = False

        average_item_weight=np.average(items_info[:,1])
        while not valid:

            # Generate a random item selection for the given city order
            item_selection=np.random.random([len(items_info)]) < (knapsack_capacity/(average_item_weight*len(items_info)))


            total_weight = np.sum(np.multiply(items_info[:, 1], item_selection))

            # If the total weight exceeds the knapsack capacity, fix the item selection
            if total_weight > knapsack_capacity:
                # Notice that the fix_overweight function is called
                counter += 1
                item_selection = fix_overweight(np.array(items_info), item_selection, knapsack_capacity)
                total_weight = np.sum(np.multiply(items_info[:, 1], item_selection))

            # Check whether the total weight is within the knapsack capacity
            if total_weight <= knapsack_capacity:
                packings.append(item_selection)
                valid = True  # it is a valid item selection

    # Add zero packing
    if parameters['zero_packing']:
        packings.append(np.zeros(len(items_info)))

    packings = np.stack(packings).astype('int8')
    return routes, packings


# Generates Routes according to the nearest neighbor heuristic
def tsp_nearest_neighbor(num_cities, tsp_nn, distance_matrix):
    routes = []
    temp_pop = []
    visited = np.zeros(num_cities)
    if tsp_nn == 0:
        return routes
    routes.append(1)
    visited[routes[0] - 1] = True
    for i in range(num_cities - 1):
        nearest_distance = 9999
        nearest_city = -1
        for city in range(1, num_cities + 1):
            if not visited[city - 1]:
                distance = distance_matrix[routes[i] - 1][city - 1]
                if distance < nearest_distance:
                    nearest_city = city
                    nearest_distance = distance
        routes.append(nearest_city)
        visited[nearest_city - 1] = True

    for i in range(tsp_nn - 1):
        new_start = np.random.randint(0, 2)
        if new_start==0:
            temp_pop.append(routes)
        else:
            temp_pop.append(np.insert(np.flip(routes)[0:num_cities-1], 0, 1))

    routes = np.array(routes)
    if tsp_nn - 1>0:
        routes = np.vstack([routes, temp_pop])
    return routes

def packing_sorted(route, distance_matrix, items_info, knapsack_capacity, location_factor, assigned_city, fill_percentage):

    if location_factor==0:
        return ranked_packing(items_info, np.zeros(len(items_info)), fill_percentage, knapsack_capacity)


    distances=np.array([])
    for i in range(len(route)):
        distances = np.append(distances, distance_matrix[route[i]-1][route[(i+1)%len(route)] - 1])

    distances_cum=np.cumsum(distances)
    percentage_left=np.divide(distances_cum, distances_cum[len(route)-1])

    pw=items_info[:, 0] / items_info[:, 1]
    pw_unsorted=[]
    iter=0
    for item in assigned_city:
        pw_unsorted.append(((1-location_factor)*pw[iter]+location_factor*pw[iter]*percentage_left[np.where(route==(item))])[0])
        b=percentage_left[np.where(route==(item))]
        iter+=1
    pw_sorted = np.argsort(pw_unsorted, kind='quicksort')

    limit_weight=fill_percentage*knapsack_capacity

    current_weight=0
    packing=np.zeros(len(items_info))
    iter=0
    while current_weight<limit_weight:
        packing[pw_sorted[len(items_info)-1-iter]]=1
        current_weight+=items_info[pw_sorted[len(items_info)- 1-iter]][1]
        iter+=1

    if fill_percentage==1:
        fix_overweight(items_info, packing, knapsack_capacity)

    return packing





