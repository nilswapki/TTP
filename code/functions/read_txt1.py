from scipy.spatial.distance import cdist
import numpy as np
def process_txt_file(file_path):
    ttp_dateset = {}
    coordinates = []
    items_info = []
    items_index_info = {}
    assigned_city = []
    city_items = {}
    process_cities_title = False
    process_items_title = False

    # reads the file and decides which kind of variable to save it to
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith("PROBLEM NAME"):
                points = line.split(':')
                ttp_dateset["problem_name"] = points[1].strip()
            elif line.startswith("KNAPSACK DATA TYPE"):
                points = line.split(':')
                ttp_dateset["knapsack_data_type"] = points[1].strip()
            elif line.startswith("DIMENSION"):
                points = line.split(':')
                ttp_dateset["num_of_cities"] = points[1].strip()
            elif line.startswith("NUMBER OF ITEMS"):
                points = line.split(':')
                ttp_dateset["num_of_items"] = points[1].strip()
            elif line.startswith("CAPACITY OF KNAPSACK"):
                points = line.split(':')
                ttp_dateset["knapsack_capacity"] = points[1].strip()
            elif line.startswith("MIN SPEED"):
                points = line.split(':')
                ttp_dateset["min_speed"] = points[1].strip()
            elif line.startswith("MAX SPEED"):
                points = line.split(':')
                ttp_dateset["max_speed"] = points[1].strip()
            elif line.startswith("RENTING RATIO"):
                points = line.split(':')
                ttp_dateset["renting_ratio"] = points[1].strip()
            elif line.startswith("EDGE_WEIGHT_TYPE"):
                points = line.split(':')
                ttp_dateset["EDGE_WEIGHT_TYPE"] = points[1].strip()
            elif line.startswith("NODE_COORD_SECTION"):
                if "NODE_COORD_SECTION" in line:
                    process_cities_title = True
                    process_items_title = False
                    continue
            elif line.startswith("ITEMS SECTION"):
                if "ITEMS SECTION" in line:
                    process_items_title = True
                    process_cities_title = False
                    continue
            else:
                if process_cities_title:
                    points = line.split()
                    x1 = float(points[1])
                    y1 = float(points[2])
                    coordinates.append((x1, y1))
                if process_items_title:
                    points = line.split()
                    x0 = int(points[0])
                    x2 = float(points[1])
                    y2 = float(points[2])
                    items_info.append([x2, y2])
                    if x0 not in items_index_info:
                        items_index_info[x0] = []
                    items_index_info[x0] = (x2, y2)
                    z = int(points[3])
                    assigned_city.append(z)
                    if z not in city_items:
                        city_items[z] = []
                    city_items[z].append((x0))

    distance_matrix = cdist(coordinates, coordinates, metric='euclidean')
    distance_matrix[distance_matrix==0]=9999
    num_of_cities = int(ttp_dateset['num_of_cities'])
    min_speed = float(ttp_dateset['min_speed'])
    max_speed = float(ttp_dateset['max_speed'])
    knapsack_capacity = int(ttp_dateset['knapsack_capacity'])
    renting_ratio = float(ttp_dateset['renting_ratio'])

    # Speed up data processing and change to numpy format
    distance_matrix = np.array(distance_matrix)
    items_info = np.array(items_info)
    assigned_city = np.array(assigned_city)
    city_items = np.array(city_items)

    return items_info, assigned_city, city_items, distance_matrix, num_of_cities, min_speed, max_speed, knapsack_capacity, renting_ratio








