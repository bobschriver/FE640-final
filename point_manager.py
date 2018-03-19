import copy

from scipy.spatial import KDTree


class PointManager():
    def __init__(self, active_points, inactive_points):
        self.active_points = active_points
        self.inactive_points = inactive_points
        
        self.points_kdtree = KDTree(list(active_points | inactive_points))
        
        self.active_change_callbacks = [self.update_active_points_kdtree]
        self.inactive_change_callbacks = []
    
    def subscribe_active_changes(self, callback):
        self.active_change_callbacks.append(callback)

    def active_intersection(self, points):
        return self.active_points & points

    def get_random_inactive(self):
        point = self.inactive_points.pop()
        self.inactive_points.add(point)
        
        return point
    
    def activate_points(self, points):
        self.active_points |= points
        self.inactive_points -= points
        
        for callback in self.active_change_callbacks:
            callback()
    
    def deactivate_points(self, points):
        self.active_points -= points
        self.inactive_points |= points
        
        for callback in self.active_change_callbacks:
            callback()
        
    def update_active_points_kdtree(self):
        self.active_points_kdtree = KDTree(list(self.active_points))
        
    def __copy__(self):
        new_point_manager = PointManager(copy.copy(self.active_points), copy.copy(self.inactive_points))
        return new_point_manager