import random

class Solution():
    def __init__(self):
        self.forward_options = []        
        self.reverse_map = {}

        self.components = []
        
        self.reverse_function = None

        self.iterations = 0
        
    def add_component(self, component):
        self.forward_options += component.forward_options
        self.reverse_map.update(component.reverse_map)
        
        self.components.append(component)
    
    def compute_value(self):
        value = 0
        
        for component in self.components:
            value +=  component.compute_value()
        
        return value
        
    def forward(self):
        choice = random.choice(self.forward_options)      
        result = choice()
        
        if choice in self.reverse_map:
            reverse_choice = self.reverse_map[choice]
            self.reverse_function = lambda: reverse_choice(result)
        else:
            self.reverse_function = None
    
    def reverse(self):
        if self.reverse_function is not None:
            self.reverse_function()