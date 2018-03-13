import random

from scipy.spatial import KDTree
from scipy.spatial.distance import euclidean

from landing import Landing

class Landings():
    def __init__(self, landing_points):        
        self.active_landing_points = set()
        self.inactive_landing_points = set(landing_points)
        self.landings = []
        
        self.active_landing_distances = None
        
        self.forward_options = [
            self.add_random_landing,
            self.remove_random_landing
        ]
        
        self.reverse_map = {
            self.add_random_landing: self.remove_landing,
            self.remove_random_landing: self.add_landing
        }
    
    
    def add_landing(self, landing):
        print("Adding landing {}".format(landing))
        self.landings.append(landing)
        
        self.inactive_landing_points.remove(landing.point)
        self.active_landing_points.add(landing.point)
        
        self.active_landing_distances = KDTree(list(self.active_landing_points))
        self.cuts.update_landings(self.active_landing_distances)
 
    def add_random_landing(self):
        #Yet I know this is dumb
        landing_point = self.inactive_landing_points.pop()
        self.inactive_landing_points.add(landing_point)
        
        landing = Landing(landing_point)
        
        self.add_landing(landing)
        return landing
    
    def remove_landing(self, landing):
        print("Removing landing {}".format(landing))
        self.landings.remove(landing)
        
        self.inactive_landing_points.add(landing.point)
        self.active_landing_points.remove(landing.point)
        
        self.active_landing_distances = KDTree(list(self.active_landing_points))
        self.cuts.update_landings(self.active_landing_distances)
    
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