import os
import csv
import random
import sys
import time
import json

import numpy as np

from scipy.spatial import KDTree
from scipy.spatial.distance import euclidean

from cuts import Cuts
from landing import Landing
from landings import Landings

from point_manager import PointManager

from solution import Solution
from heuristic import SimulatedAnnealing, RecordToRecord, ThresholdAccepting   

class Solver():
    def __init__(self, heuristic, solution):
        self.heuristic = heuristic
        self.solution = solution
            
    def solve(self):
        iterations = 0
        start_time = time.time()

        iteration_fitnesses = {}
        iteration_fitnesses["current_value"] = {}
        iteration_fitnesses["best_value"] = {}
        while self.heuristic.continue_solving(iterations):
            iterations += 1

            #print("{}\t{}\t{}".format(self.solution, self.solution.compute_value(), self.heuristic.best_value))
            self.solution.forward()
            #print("{}\t{}\t{}".format(self.solution, self.solution.compute_value(), self.heuristic.best_value))

                
            if not self.heuristic.accept_solution(self.solution):
                #print("{}\t{}\t{} Rejecting".format(self.solution, self.solution.compute_value(), self.heuristic.best_value))
                self.solution.reverse()
            else:
                #print("{}\t{}\t{} Accepting".format(self.solution, self.solution.compute_value(), self.heuristic.best_value))
                self.heuristic.set_base_solution(self.solution)

            #print("{}\t{}\t{}".format(self.solution, self.solution.compute_value(), self.heuristic.best_value))
            #print()
            
            if iterations % 1000 == 0:
                print("{} Curr Value {} Base Value {} Best Value {} seconds {}".format(iterations, self.solution.compute_value(), self.heuristic.base_value, self.heuristic.best_value, time.time() - start_time))
                print("Curr {}\tFinal {}".format(self.solution, self.heuristic.final_solution))
                iteration_fitnesses["current_value"][iterations] = self.heuristic.base_value
                iteration_fitnesses["best_value"][iterations] = self.heuristic.best_value
                start_time = time.time()
            

        return (self.heuristic.final_solution, iteration_fitnesses)
             

tree_points_path = os.path.join(".", "44120_g2_trees.txt")
road_points_path = os.path.join(".", "44120_g2_roads.txt")
 
tree_points = []
tree_weights = []
tree_basins = []

#top_left_x, top_left_y = (1377133.37353, 1118720.7104)
#min_basin = 15184

top_left_x, top_left_y = (0, 0)
min_basin = 0

with open(tree_points_path) as tree_points_file:
    tree_points_reader = csv.reader(tree_points_file)
    tree_points_header = next(tree_points_reader)
    
    for tree_point_line in tree_points_reader:
        _, _, _, _, elev_str, basin_str, x_str, y_str, height_str = tree_point_line
        #x = float(x_str) - top_left_x
        #y = top_left_y - float(y_str)
        x = float(x_str) - top_left_x
        y = float(y_str) - top_left_y
        
        
        height = float(height_str)
        weight = height * 0.82 * 49.91 * 0.000454
        
        elevation = float(elev_str)
        
        basin = (int(basin_str) - min_basin) * 500

        tree_points.append((x, y))
        tree_basins.append(basin)
        tree_weights.append(weight)

possible_tree_points_arr = np.array(tree_points)
possible_tree_weights_arr = np.array(tree_weights)
possible_tree_basin_arr = np.array(tree_basins)
possible_tree_points_kdtree = KDTree(possible_tree_points_arr)
        

road_points = []
road_coordinates = []
min_x = 100000000
min_y = 100000000
with open(road_points_path) as road_points_file:
    road_points_reader = csv.reader(road_points_file)
    road_points_header = next(road_points_reader)
    
    for road_point_line in road_points_reader:
        _, _, _, x_str, y_str, elev_str, basin_str = road_point_line
    
        #x = float(x_str) - top_left_x
        #y = top_left_y - float(y_str)
        x = float(x_str) - top_left_x
        y = float(y_str) - top_left_y

        if x < min_x:
            min_x = x

        if y < min_y: 
            min_y = y

        elevation = float(elev_str)
        basin = (int(basin_str) - min_basin) * 500

        road_coordinates.append((x, y))
        road_points.append((x, y, basin))

road_points_arr = np.array(road_coordinates)
road_points_kdtree = KDTree(road_coordinates)

feasible_tree_indeces = [index for index, l in enumerate(possible_tree_points_kdtree.query_ball_tree(road_points_kdtree, 400)) if l]

tree_points_arr = possible_tree_points_arr[feasible_tree_indeces]
tree_basins_arr = possible_tree_basin_arr[feasible_tree_indeces]
tree_weights_arr = possible_tree_weights_arr[feasible_tree_indeces]
tree_points_kdtree = KDTree(tree_points_arr) 

num_trials = 100

heuristic_type = "SimulatedAnnealing"
    
if not os.path.exists(heuristic_type):
    os.makedirs(heuristic_type)

for j in range(num_trials):
    landing_point_manager = PointManager(set(()), set(road_points))        
  
    initial_cuts = Cuts(tree_points_arr, tree_points_kdtree, tree_weights_arr, tree_basins_arr, landing_point_manager)
    initial_landings = Landings(landing_point_manager)

    landing_point_manager.subscribe_active_changes(initial_cuts.update_landings)

    for i in range(40):
        initial_landings.add_random_landing()

    initial_solution = Solution()
    initial_solution.add_component(initial_landings)
    initial_solution.add_component(initial_cuts)

    if heuristic_type == "RecordToRecord":
        heuristic = RecordToRecord()
    elif heuristic_type == "SimulatedAnnealing":
        heuristic = SimulatedAnnealing()

    heuristic.configure()
    
    solver = Solver(heuristic, initial_solution)

    final_solution, iteration_fitnesses = solver.solve()

    solution_dir = os.path.join(".", heuristic_type, "{}_{}".format(j, int(final_solution.compute_value())))
    os.makedirs(solution_dir)

    final_solution.export(solution_dir)
    json.dump(iteration_fitnesses, open(os.path.join(solution_dir, "iteration_fitnesses.json"), "w"), indent=2)
        

    