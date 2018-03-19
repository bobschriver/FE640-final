class Landing():
    def __init__(self, point):
        self.point = point
        self.value = -2000
    
    def compute_value(self):
        return self.value
        
    def __str__(self):
        return self.point.__str__()