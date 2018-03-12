import os
import csv
import random
import sys
import time

import numpy as np

from scipy.spatial import KDTree
from scipy.spatial import ConvexHull
from scipy.spatial.distance import euclidean

class Cut():
    def __init__(self, availible_tree_points, tree_point_heights, tree_kdtree, landing_points_kdtree):
        self.availible_tree_points = availible_tree_points
        self.tree_point_heights = tree_point_heights
        self.tree_distances = tree_kdtree
        self.landing_points_kdtree = landing_points_kdtree
        
        self.cut_tree_points = set()
        
        self.closest_landing_point = (sys.maxsize, sys.maxsize)
        self.centroid = None
        
        self.cost = sys.maxsize
        self.total_felling_cost = 0
        self.total_processing_cost = 0
        self.total_skidding_cost = 0
        
        self.total_felling_value = 0
        self.total_harvest_value = 0
        
        self.add_cluster(self.availible_tree_points.pop())
          
    def update_centroid(self, points):
        if len(points) == 0:
            return
    
        new_points_arr = np.array(points)
        
        new_points_len = new_points_arr.shape[0]
        all_points_len = len(self.cut_tree_points)    
        old_points_len = all_points_len - new_points_len
    
        sum_x = np.sum(new_points_arr[:, 0])
        sum_y = np.sum(new_points_arr[:, 1])
    
        if self.centroid is None:
            self.centroid = (sum_x / float(new_points_len), sum_y / float(new_points_len))
        else:
            new_x = self.centroid[0] * (old_points_len / all_points_len) +  sum_x / all_points_len
            new_y = self.centroid[1] * (old_points_len / all_points_len) +  sum_y / all_points_len
            
            self.centroid = (new_x, new_y)
        
        self.update_closest_landing_point()
            
    
    def add_cluster(self, seed_point, radius=150.0):
        start = time.time()
        tree_cluster_indeces = self.tree_distances.query_ball_point(x=seed_point, r=radius)
        print("Query Ball Point {} seconds".format(time.time() - start))
        
        possible_tree_points = set([tuple(self.tree_distances.data[index]) for index in tree_cluster_indeces])
        added_tree_points = possible_tree_points & self.availible_tree_points 
        self.availible_tree_points -=  added_tree_points
        self.cut_tree_points |= added_tree_points
        print("Finding availible points {} seconds".format(time.time() - start))
        
        felling_cost, processing_cost, skidding_cost, felling_value, harvest_value = self.compute_cost_value_for_points(list(added_tree_points))
        
        self.total_felling_cost += felling_cost
        self.total_processing_cost += processing_cost
        self.total_skidding_cost += skidding_cost
        self.total_felling_value += felling_value
        self.total_harvest_value += harvest_value
        
        self.cost = self.sum_cost()
        print("Updating cost {} seconds".format(time.time() - start))
        
        self.update_centroid(list(added_tree_points))
        print("Updating centroid {} seconds".format(time.time() - start))
    
    def sum_cost(self):
        return (self.total_felling_value + self.total_harvest_value) - (self.total_felling_cost + self.total_processing_cost + self.total_skidding_cost)
    
    def add_neighbor_cluster(self):
        source_tree = self.cut_tree_points.pop()
        self.availible_tree_points.add(source_tree)
        
        self.add_cluster(source_tree)
    
    def update_landing_points_kdtree(self, landing_points_kdtree):
        self.landing_points_kdtree = landing_points_kdtree
        self.update_closest_landing_point()
    
    def update_closest_landing_point(self):
        closest_landing_point_distance, closest_landing_point_index = self.landing_points_kdtree.query([self.centroid])
    
        if closest_landing_point_distance < euclidean(self.centroid, self.closest_landing_point):
            self.closest_landing_point = self.landing_points_kdtree.data[closest_landing_point_index]
            _, _, skidding_cost, _, _ = self.compute_cost_value_for_points(
                self.cut_tree_points, 
                compute_felling_cost=False,
                compute_processing_cost=False,
                compute_skidding_cost=True,
                compute_felling_value=False,
                compute_harvest_value=False)
                
            self.total_skidding_cost = skidding_cost
            
            self.cost = self.sum_cost()
         
    # Feet to Cubic Feet
    def height_to_merchantable_volume(self, height):
        return height * 0.60 - 9.87
    
    # Feet to Cubic Feet
    def height_to_volume(self, height):
        return height * 0.87
    
    # Cubic Feet to tonne
    # Water pounds per cubic foot = 62.4
    # Juniper pounds per sq foot @ 12% water = 31
    # 0.12 * 62.4 + 0.88 * x = 31
    # x = 26.71
    # Green Weight = 65% water
    # 0.65 * 64.4 + 0.35 * 26.71 = 49.91
    def volume_to_green_weight(self, volume):
        return (volume * 49.91) * 0.000454
    
    # Assume a DBH of 20 for all trees with no merchantable timber
    # tonne to dollars
    def felling_cost_no_merchantable(self, weight):
        return weight * 12
    
    # tonne to dollars
    def felling_cost(self, weight):
        return weight * 10
    
    # tonne to dollars
    def processing_cost(self, weight):
        return weight * 15
        
    # tonne/feer to dollars
    def skidding_cost(self, weight, distance):
        return weight * distance * 0.061 + weight * 20
    
    # dollars
    def felling_value(self):
        return 3
    
    # cubic feet to dollars
    def harvest_value(self, merchantable_volume):
        # convert cubic feet to board feet
        return (merchantable_volume * 12) * 1.25
    
    def compute_cost_value_for_points(
        self, 
        tree_points, 
        compute_felling_cost=True, 
        compute_processing_cost=True, 
        compute_skidding_cost=True,
        compute_felling_value=True,
        compute_harvest_value=True
    ):
        total_felling_cost = 0
        total_processing_cost = 0
        total_skidding_cost = 0
        
        total_felling_value = 0
        total_harvest_value = 0
        
        for tree_point in tree_points:
            tree_height = tree_point_heights[tree_point]
            tree_merchantable_volume = self.height_to_merchantable_volume(tree_height)
                
            if compute_felling_cost or compute_processing_cost or compute_skidding_cost:
                tree_volume = self.height_to_volume(tree_height)
                tree_green_weight = self.volume_to_green_weight(tree_volume)
                
            if compute_skidding_cost:
                tree_distance = euclidean(tree_point, self.closest_landing_point)
                
            if compute_felling_cost:
                if tree_merchantable_volume < 0:
                    total_felling_cost += self.felling_cost_no_merchantable(tree_green_weight)
                else:
                    total_felling_cost += self.felling_cost(tree_green_weight)
            
            if compute_processing_cost and tree_merchantable_volume > 0:
                total_processing_cost += self.processing_cost(tree_green_weight)
            
            if compute_skidding_cost and tree_merchantable_volume > 0:
                total_skidding_cost += self.skidding_cost(tree_green_weight, tree_distance)
                
            if compute_felling_value:
                total_felling_value += self.felling_value()
                
            if compute_harvest_value:
                total_harvest_value += self.harvest_value(tree_merchantable_volume)
                    
        
        return (total_felling_cost, total_processing_cost, total_skidding_cost, total_felling_value, total_harvest_value)

    def free_tree_points(self):
        self.availible_tree_points |= self.cut_tree_points
  
class Landings():
    def __init__(self, availible_landing_points):
        self.availible_landing_points = availible_landing_points
        
        self.landing_points = []
        self.landing_points_kdtree = None
        self.cost = 0
    
    def add_landing(self):
        point = self.availible_landing_points.pop()
        self.landing_points.append(point)
        self.landing_points_kdtree = KDTree(self.landing_points)
        
        self.cost -= 10000
    
    def remove_landing(self):
        point = random.choice(self.landing_points)
        self.landing_points.remove(point)
        self.availible_landing_points.add(point)
        

        self.landing_points_kdtree = KDTree(self.landing_points)
        
        self.cost += 10000
        
class Solution():
    def __init__(self, availible_tree_points, tree_point_heights, tree_kdtree, availible_landing_points):
        self.availible_tree_points = availible_tree_points
        self.tree_point_heights = tree_point_heights
        self.tree_kdtree = tree_kdtree
        
        self.availible_landing_points = availible_landing_points
    
        self.cuts = []
        self.landings = Landings(availible_landing_points)
    
    def add_tree_cluster_to_random_cut(self):
        random.choice(self.cuts).add_neighbor_cluster()
    
    def add_cut(self):
        self.cuts.append(
            Cut(self.availible_tree_points, 
                self.tree_point_heights, 
                self.tree_kdtree,
                self.landings.landing_points_kdtree)
            )
    
    def remove_cut(self):
        if len(self.cuts) == 1:
            return
        
        cut = random.choice(self.cuts)
        cut.free_tree_points()
        self.cuts.remove(cut)
    
    def add_landing(self):
        self.landings.add_landing()
        for cut in self.cuts:
            cut.update_landing_points_kdtree(self.landings.landing_points_kdtree)
    
    def remove_landing(self):
        if len(self.landings.landing_points) == 1:
            return
    
        self.landings.remove_landing()
        for cut in self.cuts:
            cut.update_landing_points_kdtree(self.landings.landing_points_kdtree)
    
    def compute_cost(self):
        cost = 0
        
        for i,cut in enumerate(self.cuts):
            print("Cut {} cost {}".format(i, cut.cost))
            cost += cut.cost
            
        cost += self.landings.cost
        
        return cost
        
        

            
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

tree_kdtree = KDTree(tree_points)  
availible_tree_points = set(tree_points)      

road_points = []

with open(road_points_path) as road_points_file:
    road_points_reader = csv.reader(road_points_file)
    road_points_header = next(road_points_reader)
    
    for road_point_line in road_points_reader:
        _, _, x_str, y_str = road_point_line
    
        x = float(x_str)
        y = float(y_str)
    
        road_points.append((x, y))
    
availible_road_points = set(road_points)
    
initial_solution = Solution(availible_tree_points, tree_point_heights, tree_kdtree, availible_road_points)

initial_solution.add_landing()
initial_solution.add_cut()


for i in range(100):
    choice = random.random()
    start = time.time()
    if choice < 0.2:
        print("Removing Cut")
        initial_solution.remove_cut()
        print("Time {}".format(time.time() - start))
    elif choice < 0.33:
        print("Adding Cut")
        initial_solution.add_cut()
        print("Time {}".format(time.time() - start))
    elif choice < 0.66:
        print("Adding Cluster")
        initial_solution.add_tree_cluster_to_random_cut()
        print("Time {}".format(time.time() - start))
    elif choice < 0.7:
        print("Adding Landing")
        initial_solution.add_landing()
        print("Time {}".format(time.time() - start))
    elif choice < 1.0:
        print("Removing Landing")
        initial_solution.remove_landing()
        print("Time {}".format(time.time() - start))
        


    #road_point = random.choice(road_points)
    #initial_solution.add_landing_point(road_point)
    #cut = Cut(availible_tree_points, tree_point_heights, tree_kdtree)
    #initial_solution.add_cut(cut)

    print("Solution cost {}".format(initial_solution.compute_cost()))
    print("")
        
        

    