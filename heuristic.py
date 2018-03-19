import sys
import math
        
class SimulatedAnnealing():
    def __init__(self):
        self.base_value = sys.maxsize
        self.best_value = sys.maxsize
        
        self.iterations = 0
        
    def configure(self, temperature=1.0, min_temperature=0.0001, alpha=0.99, repetitions=1000):
        self.temperature = temperature
        self.repetitions = repetitions
        self.alpha = alpha
        self.min_temperature = min_temperature    
        
    def set_base_value(self, value):
        self.base_value = value
        
        if value > self.best_value:
            self.best_value = value     
        
    def continue_solving(self, iterations):
        if iterations % self.repetitions == 0:
            self.temperature = self.temperature * self.alpha
        
        return self.temperature > self.min_temperature
    
    def accept_solution(self, neighbor_solution):
        neighbor_solution_value = neighbor_solution.compute_value()
        
        value_delta = neighbor_solution_value - self.base_value
       
        try:
            accept_probability =  1 / math.exp((value_delta / self.temperature))
        except OverflowError:
            accept_probability = 0
            
        accept = random.random() < accept_probability
        
        if accept:
            set_base_value(neighbor_solution_value)
            
        return accept
        
class RecordToRecord():
    def __init__(self):
        self.base_value = -1000000.0
        
        self.best_value = -1000000.0

    def configure(self, deviation=0.05, max_iterations=200000):           
        self.deviation = deviation
        self.max_iterations = max_iterations        
    
    def set_base_solution(self, solution):
        solution_value = solution.compute_value()
        self.base_value = solution_value
        
        if solution_value > self.best_value:
            self.best_value = solution_value
            self.best_solution = solution.copy_writable()

    def continue_solving(self, iterations):
        return iterations < self.max_iterations
        
    def accept_solution(self, solution):
        solution_value = solution.compute_value()

        return solution_value > self.best_value - abs(self.best_value * self.deviation)

        
class ThresholdAccepting():
    def __init__(self):
        self.base_value = 1
        self.best_value = 1

    
    def compute_normalized_cost_delta(self, base_value, compare_value):
        if compare_value < 0:
            return 1 - base_value / compare_value
        else:
            return 1 - compare_value / base_value
    
    def configure(self, threshold=0.05, min_threshold=-0.025, threshold_step=0.001, repetitions=500):
        self.threshold = threshold
        self.min_threshold = min_threshold
        self.threshold_step = threshold_step
        self.repetitions = repetitions
    
    def set_base_value(self, value):
        self.base_value = value
        
        if value > self.best_value:
            self.best_value = value  
    
    def continue_solving(self, iterations):
        if iterations % self.repetitions == 0:
            self.threshold -= self.threshold_step    
        
        return self.threshold > self.min_threshold
        
    def accept_solution(self, neighbor_solution):

        neighbor_solution_value = neighbor_solution.compute_value()
        normalized_delta = self.compute_normalized_cost_delta(self.base_value, neighbor_solution.compute_value())
      
        accept = normalized_delta < self.threshold
        
        if accept:
            self.set_base_value(neighbor_solution_value)
            
        return accept