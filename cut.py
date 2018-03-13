import sys
import time

import numpy as np

from scipy.spatial import KDTree
from scipy.spatial.distance import euclidean

class Cut():
    def __init__(self, init_point, tree_kdtree, tree_point_heights):
        self.tree_distances = tree_kdtree     
        self.tree_point_heights = tree_point_heights
        
        self.init_point = init_point
        self.centroid = init_point
        
        self.cut_tree_points = set()
        self.cut_tree_points.add(init_point)
        
        self.closest_landing_point = (sys.maxsize, sys.maxsize)
        self.closest_landing_distance = euclidean(init_point, self.closest_landing_point)
        self.closest_landing_distance_margin = 0
        
        self.total_felling_cost = 0
        self.total_processing_cost = 0
        self.total_skidding_cost = 0
        
        self.total_felling_value = 0
        self.total_harvest_value = 0
          
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
            
            new_centroid = (new_x, new_y)
            
            self.closest_landing_distance_margin = euclidean(self.centroid, new_centroid)
            self.centroid = new_centroid
            
    
    def add_neighbor_cluster(
        self, 
        active_tree_points, 
        inactive_tree_points, 
        active_landing_distances
    ):
        source_tree_point = self.cut_tree_points.pop()
        
        active_tree_points.remove(source_tree_point)
        inactive_tree_points.add(source_tree_point)

        self.add_cluster(
            source_tree_point, 
            active_tree_points, 
            inactive_tree_points, 
            active_landing_distances
        )
    
    def add_cluster(
        self,
        source_tree_point, 
        active_tree_points, 
        inactive_tree_points,
        active_landing_distances,
        radius=10.0
    ):
        start = time.time()
        tree_cluster_indeces = self.tree_distances.query_ball_point(x=source_tree_point, r=radius)
        possible_tree_points = set([tuple(self.tree_distances.data[index]) for index in tree_cluster_indeces])
        
        added_tree_points = possible_tree_points & inactive_tree_points 
        
        inactive_tree_points -=  added_tree_points
        active_tree_points |= added_tree_points
        self.cut_tree_points |= added_tree_points
        
        felling_cost, processing_cost, skidding_cost, felling_value, harvest_value = self.compute_cost_value_for_points(list(added_tree_points))
        
        self.total_felling_cost += felling_cost
        self.total_processing_cost += processing_cost
        self.total_skidding_cost += skidding_cost
        self.total_felling_value += felling_value
        self.total_harvest_value += harvest_value
        
        self.update_centroid(list(added_tree_points))
        self.update_closest_landing_point(active_landing_distances)
    
    def update_closest_landing_point(self, active_landing_distances):
        # could maybe do a ball query here
        closest_landing_point_distance, closest_landing_point_index = active_landing_distances.query([self.centroid])
    
        if closest_landing_point_distance < euclidean(self.centroid, self.closest_landing_point):
            self.closest_landing_point = active_landing_distances.data[closest_landing_point_index]
            _, _, skidding_cost, _, _ = self.compute_cost_value_for_points(
                self.cut_tree_points, 
                compute_felling_cost=False,
                compute_processing_cost=False,
                compute_skidding_cost=True,
                compute_felling_value=False,
                compute_harvest_value=False)
                
            self.total_skidding_cost = skidding_cost
            
    # Feet to Cubic Feet
    def height_to_merchantable_volume(self, height):
        return height * 0.60 - 9.87
    
    # Feet to Cubic Feet
    def height_to_volume(self, height):
        return height * 0.82
    
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
        return 4
    
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
            tree_height = self.tree_point_heights[tree_point]
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

    def compute_value(self):
        return (self.total_felling_value + self.total_harvest_value) - (self.total_felling_cost + self.total_processing_cost + self.total_skidding_cost)    
        
    def free_tree_points(self, active_tree_points, inactive_tree_points):
        active_tree_points -= self.cut_tree_points
        inactive_tree_points |= self.cut_tree_points
        
    def __str__(self):
        return self.init_point.__str__()