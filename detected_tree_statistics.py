import os 
import csv

tree_points_path = os.path.join("44120-G2_tree_points.csv")

total_tree_heigt = 0
total_tree_height_gt12 = 0
total_tree_height_lt12 = 0

total_trees = 0
total_trees_gt12 = 0
total_trees_lt12 = 0

with open(tree_points_path) as tree_points_file:
    tree_points_reader = csv.reader(tree_points_file)
    tree_points_header = next(tree_points_reader)
    
    for tree_point_line in tree_points_reader:
        _, _, x_str, y_str, height_str = tree_point_line

        height = int(height_str)
        
        if height >= 17:
            total_tree_height_gt12 += height
            total_trees_gt12 += 1
        else:
            total_tree_height_lt12 += height
            total_trees_lt12 += 1
            
        total_tree_heigt += height
        total_trees += 1
        
print("Avg tree height {}".format(total_tree_heigt / total_trees))
print("Avg tree height gt12 {}".format(total_tree_height_gt12 / total_trees_gt12))
print("Avg tree height lt12 {}".format(total_tree_height_lt12 / total_trees_lt12))