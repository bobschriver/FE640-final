import os
import csv
import random
import sys
import time

import numpy as np

from scipy.spatial import KDTree
from scipy.spatial import ConvexHull
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
            
            self.solution.forward()
                
            if not self.heuristic.accept_solution(self.solution):
                #print("Not accepting \t {:03.2f} \t {:03.2f} \t {:03.2f}".format(self.solution.compute_value(), self.heuristic.base_value, self.heuristic.best_value))
                self.solution.reverse()
            else:
                #print("Accepting \t {:03.2f} \t {:03.2f} \t {:03.2f}".format(self.solution.compute_value(), self.heuristic.base_value, self.heuristic.best_value))
                self.heuristic.set_base_solution(self.solution)
            
            if iterations % 1000 == 0:
                print("{} Curr Value {} Base Value {} Best Value {} seconds {}".format(iterations, self.solution.compute_value(), self.heuristic.base_value, self.heuristic.best_value, time.time() - start_time))
                start_time = time.time()
            
            if iterations % 10000 == 0:
                if not os.path.exists(str(iterations)):
                    os.makedirs(str(iterations))
                
                self.heuristic.best_solution.export(str(iterations))
                

        return self.solution
             
            
tree_points_path = os.path.join("44120-G2_tree_points.csv")
road_points_path = os.path.join("44120_G2_road_points.csv")

tree_points = []
tree_weights = []

with open(tree_points_path) as tree_points_file:
    tree_points_reader = csv.reader(tree_points_file)
    tree_points_header = next(tree_points_reader)
    
    for tree_point_line in tree_points_reader:
        _, _, x_str, y_str, height_str = tree_point_line
        x = float(x_str)
        y = float(y_str)
        
        height = float(height_str)
        weight = height * 0.82 * 49.91 * 0.000454
        
        tree_points.append((x, y))
        tree_weights.append(weight)

tree_points_arr = np.array(tree_points)
tree_weights_arr = np.array(tree_weights)
tree_points_kdtree = KDTree(tree_points_arr)
        
road_points = []

with open(road_points_path) as road_points_file:
    road_points_reader = csv.reader(road_points_file)
    road_points_header = next(road_points_reader)
    
    for road_point_line in road_points_reader:
        _, _, x_str, y_str = road_point_line
    
        x = float(x_str)
        y = float(y_str)
    
        road_points.append((x, y))


landing_point_manager = PointManager(set(()), set(road_points))        
  
initial_cuts = Cuts(tree_points_arr, tree_points_kdtree, tree_weights_arr, landing_point_manager)
initial_landings = Landings(landing_point_manager)

landing_point_manager.subscribe_active_changes(initial_cuts.update_landings)

#initial_landings.add_landing(Landing((1389150.86068, 1137820.3201)))
#initial_landings.add_landing(Landing((1393250.01155, 1138337.5164)))
#initial_landings.add_landing(Landing((1386765.88973, 1137579.16137)))
#initial_landings.add_landing(Landing((1381831.45325, 1132434.50497)))
#initial_landings.add_landing(Landing((1378475.21019, 1132576.38934)))

for i in range(20):
    initial_landings.add_random_landing()

initial_solution = Solution()
initial_solution.add_component(initial_landings)
initial_solution.add_component(initial_cuts)

heuristic = RecordToRecord()
heuristic.configure()

solver = Solver(heuristic, initial_solution)

best_solution = solver.solve()

        
        

    