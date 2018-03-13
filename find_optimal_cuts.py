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
from landings import Landings

from solution import Solution
from objective import Objective        

class Solver():
    def __init__(self, objective, solution):
        self.objective = objective
        self.solution = solution
            
    def solve(self):
        while not self.objective.done_solving(self.solution):
            self.solution.forward()
                
            if not self.objective.accept_solution(self.solution):
                self.solution.reverse()
                    
        return self.solution
             
            
tree_points_path = os.path.join("44120-G2_tree_points.csv")
road_points_path = os.path.join("44120_G2_road_points.csv")

tree_points = []
tree_point_heights = {}

with open(tree_points_path) as tree_points_file:
    tree_points_reader = csv.reader(tree_points_file)
    tree_points_header = next(tree_points_reader)
    
    for tree_point_line in tree_points_reader:
        _, _, x_str, y_str, height_str = tree_point_line
        x = float(x_str)
        y = float(y_str)
        
        height = int(height_str)
        
        tree_points.append((x, y))
        tree_point_heights[(x, y)] = height

     

road_points = []

with open(road_points_path) as road_points_file:
    road_points_reader = csv.reader(road_points_file)
    road_points_header = next(road_points_reader)
    
    for road_point_line in road_points_reader:
        _, _, x_str, y_str = road_point_line
    
        x = float(x_str)
        y = float(y_str)
    
        road_points.append((x, y))


tree_kdtree = KDTree(tree_points)  

initial_cuts = Cuts(tree_points, tree_point_heights)
initial_landings = Landings(road_points)

initial_cuts.landings = initial_landings
initial_landings.cuts = initial_cuts

for i in range(3):
    initial_landings.add_random_landing()

for i in range(5):
    initial_cuts.add_random_cut()
    
initial_solution = Solution()

initial_solution.add_component(initial_cuts)
initial_solution.add_component(initial_landings)

objective = Objective()

solver = Solver(objective, initial_solution)

solver.solve()

        
        

    