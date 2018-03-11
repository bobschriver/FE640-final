import os
import csv
import random
import sys
import time

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
        self.convex_hull = None
        
        self.closest_landing_point = (sys.maxsize, sys.maxsize)
        self.cost = sys.maxsize
        
        self.add_cluster(self.availible_tree_points.pop())
          

    def update_convex_hull(self, points):
        if len(points) == 0:
            return
    
        if self.convex_hull is None:
            self.convex_hull = ConvexHull(points, incremental=True)
        else:
            self.convex_hull.add_points(points)
            
        self.update_closest_landing_point()
            
    
    def add_cluster(self, seed_point, radius=150.0):
        start = time.time()
        tree_cluster_indeces = self.tree_distances.query_ball_point(x=seed_point, r=radius)
        #print("Query Ball Point {} seconds".format(time.time() - start))
        
        possible_tree_points = set([tuple(self.tree_distances.data[index]) for index in tree_cluster_indeces])
        added_tree_points = possible_tree_points & self.availible_tree_points 
        self.availible_tree_points -=  added_tree_points
        self.cut_tree_points |= added_tree_points
        #print("Finding availible points {} seconds".format(time.time() - start))
        
        #Its possible to update the cost twice here
        self.update_convex_hull(list(added_tree_points))
        #print("Updating convex hull {} seconds".format(time.time() - start))
        
        self.update_cost()
    
    def add_neighbor_cluster(self):
        source_tree = self.cut_tree_points.pop()
        self.availible_tree_points.add(source_tree)
        
        self.add_cluster(source_tree)
    
    def update_landing_points_kdtree(self, landing_points_kdtree):
        self.landing_points_kdtree = landing_points_kdtree
        self.update_closest_landing_point()
    
    def update_closest_landing_point(self):
        start = time.time()
        min_landing_distances, min_landing_indeces = self.landing_points_kdtree.query(self.convex_hull.points)
        min_landing_distances_indeces = zip(min_landing_distances, min_landing_indeces)     
        _, min_landing_index = min(min_landing_distances_indeces, key=lambda distance_index: distance_index[0])
        
        new_landing_point = tuple(self.landing_points_kdtree.data[min_landing_index])  
        print("Finding new landing {}".format(time.time() - start))
        
        
        if new_landing_point != self.closest_landing_point:
            self.closest_landing_point = new_landing_point
            self.update_cost()
        print("Updating Cost {}".format(time.time() - start))
            
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
    
    
    # Should split this into a update processing cost, and update skidding cost
    def update_cost(self):
        cut_value = 0
        
        for tree_point in self.cut_tree_points:
            tree_height = tree_point_heights[tree_point]        
            tree_merchantable_volume = self.height_to_merchantable_volume(tree_height)
            tree_volume = self.height_to_volume(tree_height)
            tree_green_weight = self.volume_to_green_weight(tree_volume)
            tree_distance = euclidean(tree_point, self.closest_landing_point)
            
            tree_value = self.felling_value()
            #print("Felling Value {}".format(tree_value))
            
            if tree_merchantable_volume < 0:
                tree_value -= self.felling_cost_no_merchantable(tree_green_weight)
                #print("Felling Cost (no merchantable) {}".format(tree_value))
            else:
                tree_value -= self.felling_cost(tree_green_weight)
                #print("Felling Cost {}".format(tree_value))
                tree_value -= self.processing_cost(tree_green_weight)
                #print("Processing Cost {}".format(tree_value))
                tree_value -= self.skidding_cost(tree_green_weight, tree_distance)
                #print("Skidding Cost {}".format(tree_value))
                
                tree_value += self.harvest_value(tree_merchantable_volume)
                #print("Harvest Value {}".format(tree_value))
            
            cut_value += tree_value
            #print("Cut Value {}\n".format(cut_value))
            
        self.cost = cut_value

class Landings():
    def __init__(self):
        self.landing_points = []
        self.landing_points_kdtree = None
        self.cost = 0
    
    def add_landing(self, point):
        self.landing_points.append(point)
        self.landing_points_kdtree = KDTree(self.landing_points)
        
        self.cost -= 1000
    
        
 
class Solution():
    def __init__(self, availible_tree_points, tree_point_heights, tree_kdtree):
        self.availible_tree_points = availible_tree_points
        self.tree_point_heights = tree_point_heights
        self.tree_kdtree = tree_kdtree
    
        self.cuts = []
        self.landings = Landings()
    
    def add_tree_cluster_to_random_cut(self):
        random.choice(self.cuts).add_neighbor_cluster()
    
    def add_cut(self):
        self.cuts.append(
            Cut(self.availible_tree_points, 
                self.tree_point_heights, 
                self.tree_kdtree, 
                self.landings.landing_points_kdtree)
            )
    
    def add_landing_point(self, landing_point):
        self.landings.add_landing(landing_point)
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
    
initial_solution = Solution(availible_tree_points, tree_point_heights, tree_kdtree)

inital_landing = availible_road_points.pop()
initial_solution.add_landing_point(inital_landing)

initial_solution.add_cut()


for i in range(100):
    choice = random.random()
    start = time.time()
    if choice < 0.33:
        print("Adding Cut")
        initial_solution.add_cut()
        print("Time {}".format(time.time() - start))
    elif choice < 0.66:
        print("Adding Cluster")
        initial_solution.add_tree_cluster_to_random_cut()
        print("Time {}".format(time.time() - start))
    elif choice < 1.0:
        print("Adding Landing")
        road_point = random.choice(road_points)
        initial_solution.add_landing_point(road_point)
        print("Time {}".format(time.time() - start))


    #road_point = random.choice(road_points)
    #initial_solution.add_landing_point(road_point)
    #cut = Cut(availible_tree_points, tree_point_heights, tree_kdtree)
    #initial_solution.add_cut(cut)

    print("Solution cost {}".format(initial_solution.compute_cost()))
    print("")
        
        

    