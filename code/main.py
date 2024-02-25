import numpy as np
import matplotlib.pyplot as plt
import functions.read_txt1 as read_txt
import functions.initialize2 as initialize
import functions.TSPandKP_Crossover as crossover
from functions.fix_overweight import fix_overweight_fast
import functions.fitness2 as fitness
import functions.multi_objective_functions1 as multi
from pymoo.indicators.hv import HV
from pymoo.visualization.scatter import Scatter
import functions.local_search_tsp as local_tsp
import time
import functions.mutation as mutation
from datetime import datetime


class Solver:
    def __init__(self, population_size, iterations, file_path, local_search_parameters, initial_parameters,
                 mutation_prob, reference_point, tournament_size=5):

        self.population_size = population_size
        self.iterations = iterations
        self.routes = []
        self.packings = []
        self.fitness_population = []
        self.reference_point = reference_point
        self.tournament_size = tournament_size
        self.local_search_parameters = local_search_parameters
        self.initial_parameters = initial_parameters
        self.mutation_prob = mutation_prob

        # Load Data
        self.items_info, self.assigned_city, self.city_items, self.distance_matrix, self.num_of_cities, self.min_speed, self.max_speed, \
            self.knapsack_capacity, self.renting_ratio = read_txt.process_txt_file(file_path)

    def optimizer(self):
        # Generate Population
        self.routes, self.packings = initialize.initialize_population(self.num_of_cities, self.population_size,
                                                                      self.distance_matrix, self.items_info,
                                                                      self.knapsack_capacity, self.assigned_city,
                                                                      self.initial_parameters)
        # Fixes the route if a route does not start with city 1
        for route in self.routes:
            route[route == 1] = route[0]
            route[0] = 1

        # Evaluate Fitness for the initial population
        self.fitness_population = fitness.fitness_function_new(self.routes[0], self.packings[0], self.assigned_city,
                                                               self.distance_matrix, self.items_info,
                                                               self.knapsack_capacity, self.max_speed, self.min_speed)
        for i in range(1, self.population_size):
            fitness_ind = fitness.fitness_function_new(self.routes[i], self.packings[i], self.assigned_city,
                                                       self.distance_matrix, self.items_info, self.knapsack_capacity,
                                                       self.max_speed, self.min_speed)
            self.fitness_population = np.vstack((self.fitness_population, fitness_ind))

        # Calculate and plot the initial pareto front
        pareto_front = multi.non_dominated_sorting(self.fitness_population, 1, True)
        plot_pareto1 = Scatter(title="Initial Pareto Front")
        plot_pareto1.add(pareto_front)
        plot_pareto1.show()

        # Plot the fitness values of the initial population
        plot_pareto4 = Scatter(title="Fitness Values")
        plot_pareto4.add(self.fitness_population)
        plot_pareto4.show()

        # List of hypervolumes for each iteration to plot convergence in the end
        hyper = []

        # Running the iterations of the EA
        iterations = 0
        while iterations < self.iterations:
            iterations += 1
            print('Iteration: ', iterations)

            # Tournament Selection
            p1_index = multi.tournament(self.fitness_population, self.tournament_size)
            p2_index = multi.tournament(self.fitness_population, self.tournament_size)

            # Routes crossover
            p1_route = self.routes[p1_index]
            p2_route = self.routes[p2_index]
            c1_route, c2_route = crossover.PMXCrossover(p1_route, p2_route)

            # Packing crossover
            p1_packing = self.packings[p1_index]
            p2_packing = self.packings[p2_index]
            c1_packing, c2_packing = crossover.kp_single(p1_packing.tolist(), p2_packing.tolist())

            # Mutations
            if np.random.random() < self.mutation_prob:
                c1_packing = mutation.bitflip(c1_packing)
                c2_packing = mutation.bitflip(c2_packing)

                c1_route = mutation.reverse_sequence_mutation(c1_route)
                c2_route = mutation.reverse_sequence_mutation(c2_route)

                for route in [c1_route, c2_route]:
                    route[route == 1] = route[0]
                    route[0] = 1

            # Fixing the new packings
            np.sum(np.multiply(self.items_info[:, 1], c1_packing))
            if np.sum(np.multiply(self.items_info[:, 1], c1_packing)) > self.knapsack_capacity:
                c1_packing = fix_overweight_fast(self.items_info, np.array(c1_packing), self.knapsack_capacity)

            if np.sum(np.multiply(self.items_info[:, 1], c2_packing)) > self.knapsack_capacity:
                c2_packing = fix_overweight_fast(self.items_info, np.array(c2_packing), self.knapsack_capacity)

            # Replacement
            c1_fitness = fitness.fitness_function_new(c1_route, c1_packing,
                                                      self.assigned_city,
                                                      self.distance_matrix,
                                                      self.items_info,
                                                      self.knapsack_capacity,
                                                      self.max_speed,
                                                      self.min_speed)
            c2_fitness = fitness.fitness_function_new(c2_route, c2_packing,
                                                      self.assigned_city,
                                                      self.distance_matrix,
                                                      self.items_info,
                                                      self.knapsack_capacity,
                                                      self.max_speed,
                                                      self.min_speed)

            # obtains the worst fitness, i.e. the fitness value from the last front with least crowding distance
            worst_fitness = multi.get_worst(np.vstack([self.fitness_population, np.transpose(c1_fitness)]))

            # replace for first offspring
            # avoids adding an individual that would be the worst for the population and avoids adding duplicates
            if not np.array_equal(worst_fitness, c1_fitness) and not c1_fitness[1] in self.fitness_population[:, 1]:
                # find index of worst individual
                index = np.where(
                    self.fitness_population[:, 0] - worst_fitness[0] + self.fitness_population[:, 1] - worst_fitness[
                        1] == 0)[0][0]

                # replacement
                self.routes[index] = c1_route
                self.packings[index] = c1_packing
                self.fitness_population[index] = c1_fitness

            # replacement for second offspring, works similar as above
            worst_fitness = multi.get_worst(np.vstack([self.fitness_population, np.transpose(c2_fitness)]))
            if not np.array_equal(worst_fitness, c2_fitness) and not c2_fitness[1] in self.fitness_population[:, 1]:
                index = np.where(
                    self.fitness_population[:, 0] - worst_fitness[0] + self.fitness_population[:, 1] - worst_fitness[
                        1] == 0)[0][0]

                self.routes[index] = c2_route
                self.packings[index] = c2_packing
                self.fitness_population[index] = c2_fitness

            # Calculate the current hypervolume
            pareto_front = multi.non_dominated_sorting(self.fitness_population, 1, True)
            ind = HV(ref_point=self.reference_point)
            volume = ind(pareto_front)
            hyper.append(volume)
            # print('HV', volume)

        # Conversion Plot of hypervolume
        print("Final Hypervolume: ", hyper[-1])
        plt.plot(range(len(hyper)), hyper)
        plt.show()

        # obtain final Pareto front
        pareto_front = multi.non_dominated_sorting(self.fitness_population, 1, True)

        # chooses n best point from current Pareto front
        reduced_pareto = multi.getBestPareto(pareto_front, 100)

        # length of Pareto front before picking points
        print('Final Size', len(pareto_front))

        # Plot of the final Pareto front
        plot_pareto2 = Scatter(title="Final Pareto Front")
        plot_pareto2.add(reduced_pareto)
        #plot_pareto2.add(self.reference_point, color="red")  # adds reference point to plot if wished for
        plot_pareto2.show()


        # saves the resulting data in the right format
        c = datetime.now()
        current_time = c.strftime('%H-%M-%S')
        with open(f'results/output/obj_values_{current_time}', 'w+') as f:
            for p in reduced_pareto:
                p[0] = p[0]*(-1)
                np.savetxt(f, p.reshape(1, -1), fmt='%1.3f', delimiter=" ")

        with open(f'results/output/tour_packing_{current_time}', 'w+') as f2:
            for p in reduced_pareto:
                tour = list(self.routes[np.where(self.fitness_population == p)[0], :])
                packing = list(self.packings[np.where(self.fitness_population == p)[0], :])
                np.savetxt(f2, tour, fmt='%i' ,delimiter=" ")
                np.savetxt(f2, packing, fmt='%i', delimiter=" ")
                f2.write("\n")

# Data Sets
file_path0 = "dataset/test-example-n4.txt"
file_path1 = "dataset/a280-n279.txt"
file_path2 = "dataset/a280-n1395.txt"
file_path3 = "dataset/a280-n2790.txt"
file_path4 = "dataset/fnl4461-n4460.txt"
file_path5 = "dataset/fnl4461-n22300.txt"
file_path6 = "dataset/fnl4461-n44600.txt"
file_path7 = "dataset/pla33810-n33809.txt"
file_path8 = "dataset/pla33810-n169045.txt"
file_path9 = "dataset/pla33810-n338090.txt"

# Reference point for each data set
reference_points = np.array([[1e-3, 1e5], [1e-3, 1e5], [1e-3, 1e5], [1e-3, 1e7], [1e-3, 1e7], [1e-3, 1e7], [1e-3, 5e10], [1e-3, 5e10], [1e-3, 5e10]])

# Parameters
iterations = 100

population_size = 150
tournament_size = 10

local_search_parameters = {'n_iterations': 5, 'searches': 0, 'num_ind': 4}
ranked_packing = 1
distance_factor = 1

initial_parameters = {
    'percentage_NN': 0.2,  # percentage of routes initialized by nearest neighbor
    'ranked_packing': 1,  # activate and deactivate packing by heuristic
    'num_improved_packing': 10,  # number of packings with heuristic
    'distance_factor': 0.0,  # between 0 and one how much the distance to end of route is taking into account for the improved packings (if 0 runs a lot faster )
    'local_search': 1,  # 0 or 1 activates the local search for every route
    'local_iterations': 1000,  # number of iterations of local search per route
    'zero_packing': 1  # 0 or 1 adds a packing with all zeros (to include edge case)
}
mutation_prob = 0.4

# Initialization
# Adjust file_path AND reference point here!
# Adjust how many points the Pareto front should consist of inside the code
# Adjust types of crossover inside the code
s = Solver(population_size, iterations, file_path1, local_search_parameters,
           initial_parameters, mutation_prob, reference_points[3], tournament_size)
s.optimizer()
