import numpy as np


def two_opt(city_path, distance_matrix, iterations):
    current_city_path = city_path.flatten()

    for _ in range(iterations):
        # obtain random indices where i < j
        i = -1
        j = -1
        while i >= j:
            i = np.random.randint(0, current_city_path.size)
            j = np.random.randint(0, current_city_path.size)

        # Calculate the distance between connected segments before and after the exchange
        old_distance = distance_matrix[current_city_path[i - 1]-1][current_city_path[i]-1] + \
                       distance_matrix[current_city_path[j-1]-1][current_city_path[j]-1]
        new_distance = distance_matrix[current_city_path[i - 1]-1][current_city_path[j-1]-1] + distance_matrix[current_city_path[i]-1][current_city_path[j]-1]

        # if there is an improvement, flip
        if new_distance < old_distance:
            current_city_path[i:j] = np.flip(current_city_path[i:j])

    return current_city_path

