import os
import csv
import random
import sys
import time

from scipy.spatial import KDTree
from scipy.spatial import ConvexHull
from scipy.spatial.distance import euclidean

class Cut():
    def __init__(self, availible_tree_points, tree_point_heights, tree_distances):
        self.availible_tree_points = availible_tree_points
        self.tree_point_heights = tree_point_heights
        self.tree_distances = tree_distances
        
        self.cut_tree_points = set()
        self.convex_hull = None
        self.convex_hull_kdtree = None
        
        self.cut_landing_distance = sys.maxsize
        
        self.add_cluster(self.availible_tree_points.pop())
          

    def update_convex_hull(self, points):
        if len(points) == 0:
            return
    
        if self.convex_hull is None:
            self.convex_hull = ConvexHull(points, incremental=True)
        else:
            self.convex_hull.add_points(points)
            
        self.convex_hull_kdtree = KDTree(self.convex_hull.points)
    
    def add_cluster(self, seed_point, radius=150.0):
        start = time.time()
        tree_cluster_indeces = self.tree_distances.query_ball_point(x=seed_point, r=radius)
        #print("Query Ball Point {} seconds".format(time.time() - start))
        
        possible_tree_points = set([tuple(self.tree_distances.data[index]) for index in tree_cluster_indeces])
        added_tree_points = possible_tree_points & self.availible_tree_points 
        self.availible_tree_points -=  added_tree_points
        self.cut_tree_points |= added_tree_points
        #print("Finding availible points {} seconds".format(time.time() - start))
        
        self.update_convex_hull(list(added_tree_points))
        #print("Updating convex hull {} seconds".format(time.time() - start))
    
    def add_neighbor_cluster(self):
        source_tree = self.cut_tree_points.pop()
        self.availible_tree_points.add(source_tree)
        
        self.add_cluster(source_tree)
    
    def compute_hull_distance(self, landing_points):
        #Don't have any landing points yet
        if landing_points_kdtree is None:
            return self.cut_landing_distance
        
        start = time.time()

        min_hull_landing_distances, _ = self.convex_hull_kdtree.query(landing_points_kdtree.data)
        self.cut_landing_distance = min(min_hull_landing_distances)
        
        #print("Compute hull distance {} seconds".format(time.time() - start))
        
        return self.cut_landing_distance
    
    #67$ per processed ton?
    
    # Feet to Cubic Feet
    def height_to_merchantable_volume(height):
        return height * 0.37
    
    def height_to_dbh(height):
        return height * 0.375 + 2.25
    
    # Feet to Cubic Feet
    def height_to_volume(height)
        return height * 0.87
    
    # Cubic Feet to Pounds
    def volume_to_green_weight(volume):
        return volume * 50
           
    def compute_cost(self, landing_points):
        tree_merchantable_value = 0
        for tree_point in self.cut_tree_points:
            tree_merchantable_value += tree_point_heights[tree_point]
            
        num_trees = len(self.cut_tree_points)
        
        cut_area = self.convex_hull.volume
        
        hull_distance = self.compute_hull_distance(landing_points)    
        
        return (tree_merchantable_value, num_trees, cut_area, hull_distance)                

class Landings():
    def __init__(self):
        self.landing_points = []
    
    def add_landing(self, point):
        self.landing_points.append(point)
    
    def compute_cost(self):
        cost = 0
        for landing_point in self.landing_points:
            cost += 1000
        return cost
        
 
class Solution():
    def __init__(self):
        self.cuts = []
        self.landings = Landings()
    
    def add_tree_cluster_to_random_cut(self):
        random.choice(self.cuts).add_neighbor_cluster()
    
    def add_cut(self, cut):
        self.cuts.append(cut)
    
    def add_landing_point(self, landing_point):
        self.landings.add_landing(landing_point)
    
    def compute_cost(self):
        cost = 0
        
        for cut in self.cuts:
            print("Cuts cost {}".format(cut.compute_cost(self.landings.landing_points)))
            
        print("Landing Cost {}".format(self.landings.compute_cost()))

            
tree_points_path = os.path.join("D:\\", "crown_location_height", "Pine Creek", "44120-G2", "tree_points", "44120-G2-tree_points.csv")
road_points_path = os.path.join("D:\\", "roads", "44120_G2", "44120_G2_road_points.csv")

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
    

initial_solution = Solution()
cut = Cut(availible_tree_points, tree_point_heights, tree_kdtree)
initial_solution.add_cut(cut)

for i in range(100):
    choice = random.random()
    
    if choice < 0.33:
        print("Adding Cut")
        cut = Cut(availible_tree_points, tree_point_heights, tree_kdtree)
        initial_solution.add_cut(cut)
    elif choice < 0.66:
        print("Adding Cluster")
        initial_solution.add_tree_cluster_to_random_cut()
    elif choice < 1.0:
        print("Adding Landing")
        road_point = random.choice(road_points)
        initial_solution.add_landing_point(road_point)

    #road_point = random.choice(road_points)
    #initial_solution.add_landing_point(road_point)
    #road_point = random.choice(road_points)
    #initial_solution.add_landing_point(road_point)
    #cut = Cut(availible_tree_points, tree_point_heights, tree_kdtree)
    #initial_solution.add_cut(cut)

    initial_solution.compute_cost()
    print("")
        
        

    