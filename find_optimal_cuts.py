import os
import csv
import random
import sys
import time

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
        while self.heuristic.continue_solving(iterations):
            iterations += 1
            
            #print("{:03.2f} \t {}".format(self.solution.compute_value(), self.solution))
            #print("{:03.2f} \t {:03.2f} \t {:03.2f}".format(self.solution.compute_value(), self.heuristic.base_value, self.heuristic.best_value))

            self.solution.forward()
            
            #print("{:03.2f} \t {}".format(self.solution.compute_value(), self.solution))
            #print("{:03.2f} \t {:03.2f} \t {:03.2f}".format(self.solution.compute_value(), self.heuristic.base_value, self.heuristic.best_value))

                
            if not self.heuristic.accept_solution(self.solution):
                self.solution.reverse()
            else:
                self.heuristic.set_base_solution(self.solution)
            
            #print("{:03.2f} \t {}".format(self.solution.compute_value(), self.solution))
            #print("{:03.2f} \t {:03.2f} \t {:03.2f}".format(self.solution.compute_value(), self.heuristic.base_value, self.heuristic.best_value))
            #print()
            
            if iterations % 10000 == 0:
                print("{} Curr Value {} Base Value {} Best Value {} seconds {}".format(iterations, self.solution.compute_value(), self.heuristic.base_value, self.heuristic.best_value, time.time() - start_time))
                print("Curr {}\tFinal {}".format(self.solution, self.heuristic.final_solution))
                start_time = time.time()
            
            #if iterations % 10000 == 0:
            #    if not os.path.exists(str(iterations)):
            #        os.makedirs(str(iterations))
            #    
            #    self.heuristic.final_solution.export(str(iterations))
                

        #self.heuristic.final_solution.export("final")
        return self.heuristic.final_solution
             

top_left_y = 1141311.44275
top_left_x = 1374717.3778

basin_min = 15184

tree_points_path = os.path.join(".", "44120_g2_trees.txt")
road_points_path = os.path.join(".", "44120_g2_roads.txt")
 
tree_points = []
tree_weights = []
tree_basins = []

with open(tree_points_path) as tree_points_file:
    tree_points_reader = csv.reader(tree_points_file)
    tree_points_header = next(tree_points_reader)
    
    for tree_point_line in tree_points_reader:
        _, _, _, _, elev_str, basin_str, x_str, y_str, height_str = tree_point_line
        #x = float(x_str) - top_left_x
        #y = top_left_y - float(y_str)
        x = float(x_str)
        y = float(y_str)
        
        
        height = float(height_str)
        weight = height * 0.82 * 49.91 * 0.000454
        
        elevation = float(elev_str)
        
        basin = int(basin_str) * 500

        tree_points.append((x, y))
        tree_basins.append(basin)
        tree_weights.append(weight)

tree_points_arr = np.array(tree_points)
tree_weights_arr = np.array(tree_weights)
tree_basin_arr = np.array(tree_basins)
tree_points_kdtree = KDTree(tree_points_arr)
        
road_points = []

with open(road_points_path) as road_points_file:
    road_points_reader = csv.reader(road_points_file)
    road_points_header = next(road_points_reader)
    
    for road_point_line in road_points_reader:
        _, _, _, x_str, y_str, elev_str, basin_str = road_point_line
    
        #x = float(x_str) - top_left_x
        #y = top_left_y - float(y_str)
        x = float(x_str)
        y = float(y_str)

        elevation = float(elev_str)
        basin = int(basin_str) * 500
    
        road_points.append((x, y, basin))

num_trials = 100
    
if not os.path.exists("RecordToRecord"):
    os.makedirs("RecordToRecord")

for j in range(num_trials):
    landing_point_manager = PointManager(set(()), set(road_points))        
  
    initial_cuts = Cuts(tree_points_arr, tree_points_kdtree, tree_weights_arr, tree_basin_arr, landing_point_manager)
    initial_landings = Landings(landing_point_manager)

    landing_point_manager.subscribe_active_changes(initial_cuts.update_landings)

    for i in range(20):
        initial_landings.add_random_landing()

    initial_solution = Solution()
    initial_solution.add_component(initial_landings)
    initial_solution.add_component(initial_cuts)

    heuristic = RecordToRecord()
    heuristic.configure()
    
    solver = Solver(heuristic, initial_solution)

    final_solution = solver.solve()

    solution_dir = os.path.join(".", "RecordToRecord", "{}_{}".format(j, int(final_solution.compute_value())))
    
    os.makedirs(solution_dir)
    final_solution.export(solution_dir)    
        

    