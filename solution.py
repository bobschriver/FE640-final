import random
import functools

from collections import deque

class Solution():
    def __init__(self):
        self.forward_options = []  
        self.forward_probabilities = []
        self.reverse_map = {}

        self.components = []
        
        self.reverse_queue = deque()

    def add_component(self, component):
        self.forward_options += component.forward_options
        self.forward_probabilities += component.forward_probabilities
        self.reverse_map.update(component.reverse_map)
        
        self.components.append(component)
    
    def compute_value(self):
        value = 0.0
        
        for component in self.components:
            value += component.compute_value()
        
        return value
        
    def forward(self):
        # This means we haven't reversed since last moving forward aka a solution has been accepted
        if self.reverse_queue:
            self.reverse_queue.clear()
        
        for component in self.components:
            component.step()
        
        for i in range(1):
            chose_option = False
            while not chose_option:
                forward_index = random.randint(0, len(self.forward_options) - 1)
                if random.random() < self.forward_probabilities[forward_index]:
                    forward_function = self.forward_options[forward_index]
                    result = forward_function()
                    
                    if result is not None and forward_function in self.reverse_map:
                        reverse_choice = self.reverse_map[forward_function]
                        self.reverse_queue.append(functools.partial(reverse_choice, result))
                    
                    chose_option = True
            
    def reverse(self):
        while self.reverse_queue:
            #print("Reversing")
            reverse_function = self.reverse_queue.pop()
            reverse_function()
            
    def export(self, output_dir):
        for component in self.components:
            component.export(output_dir)
            
    def copy_writable(self):
        writeable = Solution()
        
        for component in self.components:
            writeable.add_component(component.copy_writable())
        
        return writeable
        
    def __str__(self):
        string = ""

        for component in self.components:
            string += str(component) + "\t"
        
        return string

    def __repr__(self):
        return self.__str__()