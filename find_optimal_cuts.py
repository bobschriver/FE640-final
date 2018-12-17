from cuts import Cuts
from landing import Landing
from landings import Landings

from point_manager import PointManager

from solution import Solution
from heuristic import SimulatedAnnealing, RecordToRecord  

import os
import csv
import random
import sys
import time
import json
import argparse
import uuid

import numpy as np

from scipy.spatial import KDTree
from scipy.spatial.distance import euclidean

import boto3

s3 = boto3.resource('s3')

class Solver():
    def __init__(self, heuristic, solution, output_dirname):
        self.heuristic = heuristic
        self.solution = solution
        self.output_dirname = output_dirname
            
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
                
            if iterations % 1000 == 0:
                iteration_fitnesses["current_value"][iterations] = self.heuristic.base_value
                iteration_fitnesses["best_value"][iterations] = self.heuristic.best_value
                start_time = time.time()

            if iterations % 10000 == 0:
                current_solution_json = self.solution.to_json()
                current_solution_path = os.path.join(bucket_dirname, "{}.json".format(int(current_solution_json["fitness"])))
                current_solution_object = s3.Object("optimal-cuts", current_solution_path)
                print("Writing current solution to {}".format(current_solution_path))
                current_solution_object.put(Body=json.dumps(current_solution_json, indent=2))

        return (self.heuristic.final_solution, iteration_fitnesses)

parser = argparse.ArgumentParser()
parser.add_argument("--tile_name", dest="tile_name")      
parser.add_argument("--heuristic", dest="heuristic")
parser.add_argument("--num_trials", dest="num_trials", default=100, type=int)
parser.add_argument("--top_left", dest="top_left", default=(0, 0), type=tuple)

args = parser.parse_args()

tile_name = args.tile_name
tree_points_path = "{}-trees.txt".format(tile_name)
landing_points_path = "{}-landings.txt".format(tile_name)
heuristic_type = args.heuristic
num_trials = args.num_trials
top_left_x, top_left_y = args.top_left
min_basin = 0
algorithm_name = "polygon"

#top_left_x, top_left_y = (1377133.37353, 1118720.7104)
#min_basin = 15184

tree_points = []
tree_weights = []
tree_basins = []

with open(tree_points_path) as tree_points_file:
    tree_points_reader = csv.reader(tree_points_file)
    header = next(tree_points_reader)

    x_index = header.index("x")
    y_index = header.index("y")
    elevation_index = header.index("elevation")
    basin_index = header.index("basin")
    height_index = header.index("height")

    for tree_point_line in tree_points_reader:
        x = float(tree_point_line[x_index])
        y = float(tree_point_line[y_index])

        elevation = float(tree_point_line[elevation_index])
        basin = int(tree_point_line[basin_index]) * 1000

        height = float(tree_point_line[height_index])
        weight = height * 0.82 * 49.91 * 0.000454

        tree_points.append((x, y))
        tree_basins.append(basin)
        tree_weights.append(weight)

possible_tree_points_arr = np.array(tree_points, dtype=np.float)
possible_tree_weights_arr = np.array(tree_weights, dtype=np.float)
possible_tree_basin_arr = np.array(tree_basins, dtype=np.int)
possible_tree_points_kdtree = KDTree(possible_tree_points_arr)
        
road_points = []
road_coordinates = []
with open(landing_points_path) as landing_points_file:
    landing_points_reader = csv.reader(landing_points_file)
    header = next(landing_points_reader)

    x_index = header.index("x")
    y_index = header.index("y")
    elevation_index = header.index("elevation")
    basin_index = header.index("basin")

    for landing_point_line in landing_points_reader:
        x = float(landing_point_line[x_index])
        y = float(landing_point_line[y_index])

        elevation = float(landing_point_line[elevation_index])
        basin = int(landing_point_line[basin_index]) * 1000

        road_coordinates.append((x, y))
        road_points.append((x, y, basin))

road_points_arr = np.array(road_coordinates)
road_points_kdtree = KDTree(road_coordinates)

feasible_tree_indeces = [index for index, l in enumerate(possible_tree_points_kdtree.query_ball_tree(road_points_kdtree, 400)) if l]

tree_points_arr = possible_tree_points_arr[feasible_tree_indeces]
tree_basins_arr = possible_tree_basin_arr[feasible_tree_indeces]
tree_weights_arr = possible_tree_weights_arr[feasible_tree_indeces]
tree_points_kdtree = KDTree(tree_points_arr) 

for j in range(num_trials):
    print("TRIAL {}".format(j))
    trial_uuid = str(uuid.uuid4())

    bucket_dirname = os.path.join("output", algorithm_name, heuristic_type, tile_name, trial_uuid)
    os.makedirs(bucket_dirname)

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
    
    solver = Solver(heuristic, initial_solution, bucket_dirname)

    final_solution, iteration_fitnesses = solver.solve()
    final_solution_json = final_solution.to_json()

    final_solution_path = os.path.join(bucket_dirname, "{}_final.json".format(int(final_solution_json["fitness"])))
    final_solution_object = s3.Object("optimal-cuts", final_solution_path)
    print("Writing final solution to {}".format(final_solution_path))
    final_solution_object.put(Body=json.dumps(final_solution_json, indent=2))

    final_solution_fitnesses_object = s3.Object("optimal-cuts", os.path.join(bucket_dirname, "iteration_fitnesses.json"))
    final_solution_fitnesses_object.put(Body=json.dumps(iteration_fitnesses, indent=2))


    