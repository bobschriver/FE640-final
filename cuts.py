import random
import sys
import os
import time
import copy
import json

import numpy as np

from scipy.spatial import KDTree
from scipy.spatial.distance import euclidean
from scipy.spatial import ConvexHull

from landings import Landings

class Cuts():
    def __init__(self, tree_points, tree_points_kdtree, tree_weights, landing_point_manager):
        self.tree_points = tree_points
        self.tree_points_kdtree = tree_points_kdtree
        self.tree_weights = tree_weights
        self.landing_point_manager = landing_point_manager
        
        self.inactive = np.ones_like(tree_weights, dtype=bool)
        
        self.cuts = []
        self.values = []
        self.update_cached = []
        
        self.forward_options = [
            self.remove_random_cut,
            self.add_random_cluster,
            self.add_random_cut,
            self.split_random_cut,
            self.join_random_cuts
        ]
        
        self.forward_probabilities = [
            0.4,
            0.5,
            0.8,
            0.05,
            0.0
        ]
        
        self.reverse_map = {
            self.add_random_cut: self.remove_cut,
            self.remove_random_cut: self.init_cut,
            self.add_random_cluster: self.remove_cluster
        }

    def split_random_cut(self):
        if len(self.cuts) <= 1:
            return
    
        source_cut_index = random.randint(0, len(self.cuts) - 1)
        cut = self.cuts[source_cut_index]
        
        # Three is the minimum size cut to split
        if cut.size < 4:
            return
        
        chord_first_coordinate, chord_second_coordinate = np.random.choice(cut, size=(2, 1), replace=False)
        
        x1, y1 = self.tree_points[chord_first_coordinate[0]]
        x2, y2 = self.tree_points[chord_second_coordinate[0]]
        
        #print("X1 {} Y1 {} X2 {} Y2 {}".format(x1, y1, x2, y2))
        
        cut_side_one = []
        cut_side_two = []
        for point_index in cut:
            x, y = self.tree_points[point_index]
            
            #print("X {} Y {}".format(x, y))
            
            d = (x - x1)*(y2 - y1) - (y - y1)*(x2 - x1)
            
            if d <= 0:
                cut_side_one.append(point_index)
            else:
                cut_side_two.append(point_index)
        
        #print(cut_side_one)
        #print(cut_side_two)
        
        if len(cut_side_one) == 0 or len(cut_side_two) == 0:
            return
        
        self.cuts[source_cut_index] = np.array(cut_side_one, dtype=np.int32)
        self.update_cached[source_cut_index] = True
        
        self.init_cut(np.array(cut_side_two, dtype=np.int32))
    
    def join_random_cuts(self):
        # I think I should reimagine this, potentially join closest cuts?
        if len(self.cuts) <= 2:
            return
    
        first_cut_index, second_cut_index = random.sample(range(len(self.cuts) - 1), 2)

        second_cut = self.cuts.pop(second_cut_index)
        self.values.pop(second_cut_index)
        self.update_cached.pop(second_cut_index)
        
        first_cut = self.cuts[first_cut_index]
        
        self.cuts[first_cut_index] = np.concatenate([first_cut, second_cut])
        self.update_cached[first_cut_index] = True
        
    
    def add_random_cut(self):
        inactive_tree_indeces = np.nonzero(self.inactive)[0]
        source_index = np.random.choice(inactive_tree_indeces)
        source_point = self.tree_points[source_index]
        
        cut = np.array([], dtype=np.int32)
        self.init_cut(cut)

        cut_index = len(self.cuts) - 1        
        self.add_cluster(cut_index, source_point, random.randint(100, 300))
        
        return cut_index

    def init_cut(self, cut):
        #print("Init Cut Inactive Length {}".format(np.nonzero(self.inactive)[0].shape))
        self.cuts.append(cut)
        self.values.append(0)
        self.update_cached.append(True)
        
        self.inactive[cut] = False
        #print("Init Cut Inactive Length {}".format(np.nonzero(self.inactive)[0].shape))
    
    def add_cluster(self, cut_index, source_point, radius=100):
        #print("Add Cluster Cut Length {} Inactive Length {}".format(self.cuts[cut_index].shape, np.nonzero(self.inactive)[0].shape))
    
        start = time.time()
        tree_cluster_indeces = self.tree_points_kdtree.query_ball_point(x=source_point, r=radius, eps=1.0)
        
        inactive_tree_cluster_indeces = [tree_cluster_indeces[index] for index in np.nonzero(self.inactive[tree_cluster_indeces])[0]]
        
        self.cuts[cut_index] = np.concatenate([self.cuts[cut_index], np.array(inactive_tree_cluster_indeces, dtype=np.int32)])
        self.update_cached[cut_index] = True
        
        self.inactive[inactive_tree_cluster_indeces] = False
        
        #print("Add Cluster Cut Length {} Inactive Length {}".format(self.cuts[cut_index].shape, np.nonzero(self.inactive)[0].shape))

        
    def add_random_cluster(self):
        if len(self.cuts) <= 0:
            return None
            
        start = time.time()    

        cut_index = random.randint(0, len(self.cuts) - 1)
        source_index = np.random.choice(self.cuts[cut_index])
        
        current_cut_length = len(self.cuts[cut_index]) - 1
        
        self.add_cluster(cut_index, self.tree_points[source_index], random.randint(25, 75))
        
        return (cut_index, current_cut_length)

    def remove_cluster(self, cluster_index): 
        cut_index, cluster_start_index = cluster_index
        #print("Remove Cluster Cut Length {} Inactive Length {}".format(self.cuts[cut_index].shape, np.nonzero(self.inactive)[0].shape))
        if cluster_start_index == 0:
            return

        cut = self.cuts[cut_index]
        cluster = cut[cluster_start_index:]
        
        self.inactive[cluster] = True
        self.update_cached[cut_index] = True
        
        self.cuts[cut_index] = cut[:cluster_start_index]
            
        #print("Remove Cluster Cut Length {} Inactive Length {}".format(self.cuts[cut_index].shape, np.nonzero(self.inactive)[0].shape))
        
    def remove_cut(self, cut_index):
        #print("Remove Cut Cuts Length {} Inactive Length {}".format(len(self.cuts), np.nonzero(self.inactive)[0].shape))
            
        cut = self.cuts.pop(cut_index)
        self.values.pop(cut_index)
        self.update_cached.pop(cut_index)
        
        self.inactive[cut] = True
        
        #print("Remove Cut Cuts Length {} Inactive Length {}".format(len(self.cuts), np.nonzero(self.inactive)[0].shape))
        return cut

    def remove_random_cut(self):
        if len(self.cuts) <= 0:
            return None

        cut_index = random.randint(0, len(self.cuts) - 1)
        cut = self.remove_cut(cut_index)
        
        return cut
    
    def get_cut_value(self, cut_index):
        if self.update_cached[cut_index]:
            cut = self.cuts[cut_index]
            
            cut_tree_points = self.tree_points[cut]
            cut_tree_weights = self.tree_weights[cut]
            
            cut_tree_points_length = float(len(cut_tree_points))
            centroid = (np.sum(cut_tree_points[:, 0]) / cut_tree_points_length, np.sum(cut_tree_points[:, 1]) / cut_tree_points_length)
        
            closest_landing_point_distance, closest_landing_point_index = self.landing_point_manager.active_points_kdtree.query([centroid], eps=1.0)
            
            closest_landing_point = self.landing_point_manager.active_points_kdtree.data[closest_landing_point_index][0]
            
            cut_tree_distances = np.linalg.norm(cut_tree_points - closest_landing_point, axis=1)
            
            cut_tree_weights_gt = cut_tree_weights[cut_tree_weights >= 0.3]
            cut_tree_weights_lt = cut_tree_weights[cut_tree_weights < 0.3]
            
            cut_felling_cost = np.sum(cut_tree_weights_lt * 12) + np.sum(cut_tree_weights_gt * 10)
            cut_processing_cost = np.sum(cut_tree_weights_gt * 15)
            cut_skidding_cost = np.sum(np.select([cut_tree_weights >= 0.3], [cut_tree_weights * (cut_tree_distances * 0.061 + 20)]))
            
            cut_felling_value = 3.0 * cut_tree_weights.shape[0]
            cut_harvest_value = np.sum(cut_tree_weights_gt * 71.65)

            self.values[cut_index] = float((cut_felling_value + cut_harvest_value) - (cut_felling_cost + cut_processing_cost + cut_skidding_cost))
        
        return self.values[cut_index]
    
    def compute_value(self):
        total_value = 0
        for i, cut in enumerate(self.cuts):
            total_value += self.get_cut_value(i)
        
        return total_value

    def update_landings(self):
        self.update_cached = [True for i in range(len(self.update_cached))]
        
    def update_state(self):
        pass
        
    def copy_writable(self):
        start_time = time.time()
        writable =  Cuts(self.tree_points, None, self.tree_weights, None)
        writable.cuts = copy.deepcopy(self.cuts)
        writable.values = copy.copy(self.values)
        
        return writable
        
    def export(self, output_dir):
        cuts_output_dir = os.path.join(output_dir, "cuts")
        if not os.path.exists(cuts_output_dir):
            os.makedirs(cuts_output_dir)
            
        for cut_index, cut in enumerate(self.cuts):
            output_dict = {}
            
            output_dict["fitness"] = self.values[cut_index]

            cut_tree_points_list = list(self.tree_points[cut])

            if len(cut_tree_points_list) >= 3:
                hull = ConvexHull(cut_tree_points_list)
                hull_points = [tuple(cut_tree_points_list[index]) for index in hull.vertices]
        
                output_dict["hull_points"] = hull_points

            cut_tree_weights = self.tree_weights[cut]
            output_dict["harvest_weight"] = np.sum(cut_tree_weights[cut_tree_weights >= 0.3])
            output_dict["non_harvest_weight"] = np.sum(cut_tree_weights[cut_tree_weights < 0.3])
                
            output_dict["num_trees"] = cut.size
            
            filename = "{}.json".format(cut_index)
       
            with open(os.path.join(cuts_output_dir, filename), 'w') as fp:
                json.dump(output_dict, fp)
