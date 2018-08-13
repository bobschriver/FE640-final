class Landing():
    def __init__(self, point):
        self.point = point
        self.value = -500
    
    def compute_value(self):
        return self.value
        
    def to_json(self):
        landing_json = {}
        landing_json["point"] = self.point
        landing_json["fitness"] = self.value

        return landing_json

    def __str__(self):
        return self.point.__str__()