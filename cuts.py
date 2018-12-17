import random
import sys
import os
import time
import copy
import json

from enum import Enum

import numpy as np

from scipy.spatial import KDTree
from scipy.spatial.distance import euclidean
from scipy.spatial import ConvexHull

from landings import Landings

class ClosestLandingState(Enum):
    KNOWN = 1 # Closest landing point is known and active
    UNKNOWN = 2 # Closest landing point is unknown
    ORPHANED = 3 # Closest landing point is inactive

class Cuts():
    def __init__(self, tree_points, tree_points_kdtree, tree_weights, tree_basins, landing_point_manager):
        self.tree_points = tree_points
        self.tree_points_kdtree = tree_points_kdtree
        
        self.tree_weights = tree_weights
        self.tree_basins = tree_basins
        
        self.landing_point_manager = landing_point_manager
        
        self.inactive = np.ones_like(tree_weights, dtype=bool)
        
        self.cuts = []
        self.values = []

        self.update_cached = []

        self.closest_landings = []
        self.landing_distances = []
        self.closest_landing_states = []
        
        self.felling_values = []
        self.harvest_values = []
        self.equipment_moving_costs = []
        self.felling_costs = []
        self.processing_costs = []
        self.skidding_costs = []
        
        self.forward_options = [
            self.add_random_cut,
            self.remove_random_cut,
            self.add_random_cluster,
            self.split_random_cut,
            self.remove_orphaned_cuts,
        ]
        
        self.forward_probabilities = [
            1.0,
            0.5,
            0.25,
            0.0,
            0.0001,
        ]
        
        self.reverse_map = {
            self.add_random_cut: self.remove_cut,
            self.remove_random_cut: self.init_cut,
            self.add_random_cluster: self.remove_cluster,
            self.split_random_cut: self.join_cuts,
        }
        
        self.max_iterations = 100000.0
        
        self.starting_forward_probabilities = [
            1.0,
            0.5,
            0.25,
            0.0,
            0.0001,
        ]

        self.ending_forward_probabilites = [
            0.7,
            1.0,
            0.25,
            0.0,
            0.0001,
        ]

    def step(self):
        for i in range(len(self.forward_options)):
            self.forward_probabilities[i] = self.forward_probabilities[i] + \
                (self.ending_forward_probabilites[i] - self.starting_forward_probabilities[i]) * \
                (1 / self.max_iterations)
            #print("{} {}".format(i, self.forward_probabilities[i]))

    def split_random_cut(self):
        #print("Split Random Cut")
        if len(self.cuts) <= 1:
            return
    
        source_cut_index = random.randint(0, len(self.cuts) - 1)
        if self.closest_landing_states[source_cut_index] == ClosestLandingState.ORPHANED:
            #print("Cannot split orphaned cut")
            return

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
            
            d = (x-x1) * (y2-y1) - (y-y1) * (x2-x1)
            
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
        self.closest_landing_states[source_cut_index] = ClosestLandingState.UNKNOWN
        
        split_cut_index = self.init_cut(np.array(cut_side_two, dtype=np.int32))

        return (source_cut_index, split_cut_index)
    
    def join_cuts(self, cut_indeces):
        first_cut_index, second_cut_index = cut_indeces

        first_cut = self.cuts[first_cut_index]
        second_cut = self.cuts[second_cut_index]

        self.remove_cut(second_cut_index)
        self.remove_cut(first_cut_index)
        
        joined_cut = np.concatenate([first_cut, second_cut])

        self.init_cut(joined_cut)

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
        #print("Add Random Cut")
        inactive_tree_indeces = np.nonzero(self.inactive)[0]
        if inactive_tree_indeces.size == 0:
            return None

        source_index = np.random.choice(inactive_tree_indeces)
        
        cut = np.array([], dtype=np.int32)
        self.init_cut(cut)

        cut_index = len(self.cuts) - 1        
        self.add_cluster(cut_index, source_index, random.randint(100, 300))
        
        return cut_index

    def init_cut(self, cut):
        #print("Init Cut Inactive Length {}".format(np.nonzero(self.inactive)[0].shape))
        cut_index = len(self.cuts)
        self.cuts.append(cut)
        
        self.values.append(0)
        self.update_cached.append(True)

        self.closest_landings.append((sys.maxsize, sys.maxsize, sys.maxsize))
        self.landing_distances.append(sys.maxsize)
        self.closest_landing_states.append(ClosestLandingState.UNKNOWN)

        self.felling_values.append(0)
        self.harvest_values.append(0)
        self.equipment_moving_costs.append(0)
        self.felling_costs.append(0)
        self.processing_costs.append(0)
        self.skidding_costs.append(0)
        
        self.inactive[cut] = False

        return cut_index

    
    def add_cluster(self, cut_index, source_index, radius=100):
        #print("Add Cluster Cut Length {} Inactive Length {}".format(self.cuts[cut_index].shape, np.nonzero(self.inactive)[0].shape))
        source_point = self.tree_points[source_index]
        source_basin = self.tree_basins[source_index]

        tree_cluster_indeces = self.tree_points_kdtree.query_ball_point(x=source_point, r=radius, eps=1.0)
        #print(tree_cluster_indeces)

        inactive_tree_cluster_indeces = [
            tree_cluster_indeces[index] for index in np.where(self.inactive[tree_cluster_indeces])[0]
        ]
        
        #print(inactive_tree_cluster_indeces)
        #print(self.tree_basins[inactive_tree_cluster_indeces])
        #print(np.where(self.tree_basins[inactive_tree_cluster_indeces] == source_basin)[0])

        valid_tree_cluster_indeces = [
            inactive_tree_cluster_indeces[index] for index in np.where(self.tree_basins[inactive_tree_cluster_indeces] == source_basin)[0]
        ]
        #print(valid_tree_cluster_indeces)

        self.cuts[cut_index] = np.concatenate([self.cuts[cut_index], np.array(valid_tree_cluster_indeces, dtype=np.int32)])
        self.update_cached[cut_index] = True
        
        self.inactive[inactive_tree_cluster_indeces] = False
        
        #print("Add Cluster Cut Length {} Inactive Length {}".format(self.cuts[cut_index].shape, np.nonzero(self.inactive)[0].shape))

        
    def add_random_cluster(self):
        #print("Add Random Cluster")
        if len(self.cuts) <= 0:
            return None
            
        start = time.time()    

        cut_index = random.randint(0, len(self.cuts) - 1)
        source_index = np.random.choice(self.cuts[cut_index])
        
        current_cut_length = len(self.cuts[cut_index])
        
        self.add_cluster(cut_index, source_index, random.randint(25, 75))
        
        return (cut_index, current_cut_length)

    def remove_cluster(self, cluster_index): 
        cut_index, cluster_start_index = cluster_index

        #print("Remove Cluster Cut Length {} Inactive Length {}".format(self.cuts[cut_index].shape, np.nonzero(self.inactive)[0].shape))
        if cluster_start_index == 0:
            return None
        
        cut = self.cuts[cut_index]
        cluster = cut[cluster_start_index:]
        
        self.inactive[cluster] = True
        self.update_cached[cut_index] = True
        
        self.cuts[cut_index] = cut[:cluster_start_index]
            
        #print("Remove Cluster Cut Length {} Inactive Length {}".format(self.cuts[cut_index].shape, np.nonzero(self.inactive)[0].shape))
        
    def remove_cut(self, cut_index):
        #print("Remove Cut Cuts Length {} Inactive Length {}".format(len(self.cuts), np.nonzero(self.inactive)[0].shape)) 
        
        self.values.pop(cut_index)
        self.update_cached.pop(cut_index)
        
        self.closest_landings.pop(cut_index)
        self.landing_distances.pop(cut_index)
        self.closest_landing_states.pop(cut_index)
                
        self.felling_values.pop(cut_index)
        self.harvest_values.pop(cut_index)
        self.equipment_moving_costs.pop(cut_index)
        self.felling_costs.pop(cut_index)
        self.processing_costs.pop(cut_index)
        self.skidding_costs.pop(cut_index)

        cut = self.cuts.pop(cut_index)
        self.inactive[cut] = True
        
        #print("Remove Cut Cuts Length {} Inactive Length {}".format(len(self.cuts), np.nonzero(self.inactive)[0].shape))
        return cut

    def remove_random_cut(self):
        #print("Remove Random Cut")
        if len(self.cuts) <= 0:
            return None

        cut_index = random.randint(0, len(self.cuts) - 1)
        cut = self.remove_cut(cut_index)
        
        return cut

    def remove_orphaned_cuts(self):
        #print(self.closest_landing_states)
        orphaned_cut_indeces = np.where(self.closest_landing_states == ClosestLandingState.ORPHANED)[0]
        #print(orphaned_cut_indeces)

        for orphaned_cut_index in orphaned_cut_indeces:
            self.remove_cut(orphaned_cut_index)
    
    #TODO: Fix this
    def compute_landing_distance(self, tree_point, tree_basin, landing):
        landing_point_x, landing_point_y, landing_basin = landing
        landing_point = (landing_point_x, landing_point_y)

        landing_distance = 0
        if tree_basin != landing_basin:
            landing_distance += 1000

        return landing_distance + euclidean(tree_point, landing_point)

    def compute_landing_distances(self, tree_points, tree_basins, landing):
        landing_point_x, landing_point_y, landing_basin = landing
        landing_point = (landing_point_x, landing_point_y)

        return np.linalg.norm(tree_points - landing_point, axis=1) + \
            np.not_equal(tree_basins, landing_basin) * 1000

    def get_cut_centroid(self, cut_index):
        cut = self.cuts[cut_index]
        cut_tree_points = self.tree_points[cut]

        cut_tree_points_length = float(len(cut_tree_points))
        x_centroid = np.sum(cut_tree_points[:, 0]) / cut_tree_points_length
        y_centroid = np.sum(cut_tree_points[:, 1]) / cut_tree_points_length

        return (x_centroid, y_centroid)

    def get_cut_basin(self, cut_index):
        cut = self.cuts[cut_index]
        cut_tree_basins = self.tree_basins[cut]

        return cut_tree_basins[0]

    def update_closest_landing(self, cut_index, cut_tree_points, cut_tree_basins): 
        x_centroid, y_centroid = self.get_cut_centroid(cut_index)
        cut_basin = self.get_cut_basin(cut_index)
        
        center_point = (x_centroid, y_centroid)
        center = (x_centroid, y_centroid, cut_basin)
        
        _, closest_landing_index = self.landing_point_manager.active_points_kdtree.query([center], eps=1.0)
        closest_landing = self.landing_point_manager.active_points_kdtree.data[closest_landing_index][0]   
        closest_landing = tuple(closest_landing)

        current_landing = self.closest_landings[cut_index]
        if closest_landing != current_landing:
            # Closest landing has changed, check if we are now orphaned
            new_landing_distance = self.compute_landing_distance(
                center_point,
                cut_basin,
                closest_landing)
            
            if self.closest_landing_states[cut_index] != ClosestLandingState.UNKNOWN:
                if new_landing_distance > 1000:
                    # Our new landing is in a new basin, we've been orphaned!
                    if self.landing_distances[cut_index] < 1000 and self.closest_landing_states[cut_index] == ClosestLandingState.KNOWN:
                        #print("Orphaning cut {}".format(cut_index))
                        self.closest_landing_states[cut_index] = ClosestLandingState.ORPHANED
                else:
                    # Our new landing is within our basin
                    if self.closest_landing_states[cut_index] == ClosestLandingState.ORPHANED:
                        #print("Resolving orphaned cut {}".format(cut_index))
                        self.closest_landing_states[cut_index] = ClosestLandingState.KNOWN
            else:
                self.closest_landing_states[cut_index] = ClosestLandingState.KNOWN

            self.closest_landings[cut_index] = closest_landing
            self.landing_distances[cut_index] = new_landing_distance

    def get_cut_value(self, cut_index):
        if self.update_cached[cut_index]:
            cut = self.cuts[cut_index]
            
            cut_tree_points = self.tree_points[cut]
            cut_tree_weights = self.tree_weights[cut]
            cut_tree_basins = self.tree_basins[cut]
            
            self.update_closest_landing(cut_index, cut_tree_points, cut_tree_basins)

            cut_tree_distances = self.compute_landing_distances(cut_tree_points, cut_tree_basins, np.array(self.closest_landings[cut_index]))

            cut_tree_weights_gt = cut_tree_weights[cut_tree_weights >= 0.3]
            cut_tree_weights_lt = cut_tree_weights[cut_tree_weights < 0.3]
            
            equipment_moving_cost = self.landing_distances[cut_index] * 0.01
            cut_felling_cost = np.sum(cut_tree_weights_lt * 12) + np.sum(cut_tree_weights_gt * 10)
            cut_processing_cost = np.sum(cut_tree_weights_gt * 15)
            cut_skidding_cost = np.sum(np.select([cut_tree_weights >= 0.3], [cut_tree_weights * (cut_tree_distances * 0.061 + 20)]))
            
            cut_felling_value = 2.0 * cut_tree_weights.shape[0]
            cut_harvest_value = np.sum(cut_tree_weights_gt * 71.65)

            self.felling_values[cut_index] = float(cut_felling_value)
            self.harvest_values[cut_index] = float(cut_harvest_value)
            
            self.equipment_moving_costs[cut_index] = float(equipment_moving_cost)
            self.felling_costs[cut_index] = float(cut_felling_cost)
            self.processing_costs[cut_index] = float(cut_processing_cost)
            self.skidding_costs[cut_index] = float(cut_skidding_cost)
            
            self.values[cut_index] = float((cut_felling_value + cut_harvest_value) - (equipment_moving_cost + cut_felling_cost + cut_processing_cost + cut_skidding_cost))
        
            self.update_cached[cut_index] = False
            
        #print(self.values[cut_index])
        if self.closest_landing_states[cut_index] == ClosestLandingState.ORPHANED:
            return 0.0
        else:
            return self.values[cut_index]
    
    def compute_value(self):
        total_value = 0
        for i, cut in enumerate(self.cuts):
            total_value += self.get_cut_value(i)
        
        return total_value

    def update_landings(self, active_landing_points):
        for cut_index, cut in enumerate(self.cuts):
            self.update_cached[cut_index] = True
        
    def update_state(self):
        pass
        
    def copy_writable(self):
        start_time = time.time()
        writable =  Cuts(self.tree_points, None, self.tree_weights, self.tree_basins, None)
        writable.cuts = copy.deepcopy(self.cuts)
        
        writable.values = copy.copy(self.values)
        writable.felling_values = copy.copy(self.felling_values)
        writable.harvest_values = copy.copy(self.harvest_values)
            
        writable.equipment_moving_costs = copy.copy(self.equipment_moving_costs)
        writable.felling_costs = copy.copy(self.felling_costs)
        writable.processing_costs = copy.copy(self.processing_costs)
        writable.skidding_costs = copy.copy(self.skidding_costs)

        writable.closest_landings = copy.deepcopy(self.closest_landings)
        writable.landing_distances = copy.copy(self.landing_distances)
        writable.closest_landing_states = copy.deepcopy(self.closest_landing_states)
        
        writable.update_cached = copy.copy(self.update_cached)
        
        writable.forward_probabilities = copy.copy(self.forward_probabilities)
        
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
                try:
                    hull = ConvexHull(cut_tree_points_list)
                    hull_points = [tuple(cut_tree_points_list[index]) for index in hull.vertices]
        
                    output_dict["hull_points"] = hull_points
                except:
                    print("Invalid Hull")

            cut_tree_weights = self.tree_weights[cut]
            output_dict["harvest_weight"] = np.sum(cut_tree_weights[cut_tree_weights >= 0.3])
            output_dict["non_harvest_weight"] = np.sum(cut_tree_weights[cut_tree_weights < 0.3])
                
            output_dict["num_trees"] = cut.size
            
            output_dict["closest_landing"] = self.closest_landings[cut_index]
            output_dict["closest_landing_distance"] = self.landing_distances[cut_index]
            
            output_dict["felling_value"] = self.felling_values[cut_index]
            output_dict["harvest_value"] = self.harvest_values[cut_index]
            
            output_dict["equipment_moving_cost"] = self.equipment_moving_costs[cut_index]
            output_dict["felling_cost"] = self.felling_costs[cut_index]
            output_dict["processing_cost"] = self.processing_costs[cut_index]
            output_dict["skidding_cost"] = self.skidding_costs[cut_index]
            
            filename = "{}.json".format(cut_index)
       
            with open(os.path.join(cuts_output_dir, filename), 'w') as fp:
                json.dump(output_dict, fp)

    def to_json(self):
        cuts_json = {}
        cuts_json["component_type"] = "cuts"
        cuts_json["active_cuts"] = []

        for cut_index, cut in enumerate(self.cuts):
            cut_json = {}
            
            x, y = self.get_cut_centroid(cut_index)
            cut_json["x"] = float(x)
            cut_json["y"] = float(y)
            cut_json["basin"] = int(self.get_cut_basin(cut_index))

            cut_json["fitness"] = float(self.values[cut_index])

            cut_tree_points_list = list(self.tree_points[cut])

            if len(cut_tree_points_list) >= 3:
                try:
                    hull = ConvexHull(cut_tree_points_list)
                    hull_points = [tuple(cut_tree_points_list[index]) for index in hull.vertices]
        
                    cut_json["hull_points"] = hull_points
                except:
                    print("Invalid Hull")

            cut_tree_weights = self.tree_weights[cut]
            cut_json["harvest_weight"] = np.sum(cut_tree_weights[cut_tree_weights >= 0.3])
            cut_json["non_harvest_weight"] = np.sum(cut_tree_weights[cut_tree_weights < 0.3])
                
            cut_json["num_trees"] = cut.size
            
            cut_json["closest_landing"] = self.closest_landings[cut_index]
            cut_json["closest_landing_distance"] = self.landing_distances[cut_index]
            
            cut_json["felling_value"] = self.felling_values[cut_index]
            cut_json["harvest_value"] = self.harvest_values[cut_index]
            
            cut_json["equipment_moving_cost"] = self.equipment_moving_costs[cut_index]
            cut_json["felling_cost"] = self.felling_costs[cut_index]
            cut_json["processing_cost"] = self.processing_costs[cut_index]
            cut_json["skidding_cost"] = self.skidding_costs[cut_index]

            cut_json["orphaned"] = self.closest_landing_states[cut_index] == ClosestLandingState.ORPHANED

            cuts_json["active_cuts"].append(cut_json)
        
        return cuts_json

    def __str__(self):
        return "{} Active Cuts".format(len(self.cuts))

    def __repr__(self):
        return self.__str__()