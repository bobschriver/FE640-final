import random
import os
import json
import copy

from scipy.spatial import KDTree
from scipy.spatial.distance import euclidean

from landing import Landing

class Landings():
    def __init__(self, landing_point_manager):        
        self.landing_point_manager = landing_point_manager
        self.landings = []

        self.forward_options = [
            self.add_random_landing,
            self.remove_random_landing
        ]
        
        self.forward_probabilities = [
            0.05,
            0.02
        ]
        
        self.reverse_map = {
            self.add_random_landing: self.remove_landing,
            self.remove_random_landing: self.add_landing
        }

    def add_landing(self, landing):
        #print("Adding Landing {} {}".format(landing, len(self.landings)))
        self.landings.append(landing)
        self.landing_point_manager.activate_points(set([landing.point]))
        
 
    def add_random_landing(self):
        landing_point = self.landing_point_manager.get_random_inactive()
        
        landing = Landing(landing_point)
        
        self.add_landing(landing)
        return landing
    
    def remove_landing(self, landing):
        #print("Removing Landing {} {}".format(landing, len(self.landings)))
        self.landings.remove(landing)
        
        self.landing_point_manager.deactivate_points(set([landing.point]))
    
    def remove_random_landing(self):
        if len(self.landings) == 1:
            return None
    
        landing = random.choice(self.landings)
        self.remove_landing(landing)
        return landing
    
    def compute_value(self):
        value = 0
        for landing in self.landings:
            value += landing.compute_value()
            
        return value
    
    def copy_writable(self):
        writable = Landings(None)
        writable.landings = copy.deepcopy(self.landings)   
        
        return writable
    
    def export(self, output_dir):
        landings_output_dir = os.path.join(output_dir, "landings")
        if not os.path.exists(landings_output_dir):
            os.makedirs(landings_output_dir)
        
        landings_dict = {}
        
        landing_points = [landing.point for landing in self.landings]
        landings_dict["landing_points"] = landing_points
        
        with open(os.path.join(landings_output_dir, "landings.json"), 'w') as fp:
            json.dump(landings_dict, fp)
