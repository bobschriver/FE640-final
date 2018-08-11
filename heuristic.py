import sys
import math
import random
        
class SimulatedAnnealing():
    def __init__(self):
        self.base_value = -100000.0
        self.best_value = -100000.0

        self.iterations_since_improvement = 0
        
    def configure(self, temperature=0.25, min_temperature=0.0001, alpha=0.99, repetitions=200):
        self.temperature = temperature
        self.repetitions = repetitions
        self.alpha = alpha
        self.min_temperature = min_temperature    
        
    def set_base_solution(self, solution):
        solution_value = solution.compute_value()
        
        self.base_solution = solution
        self.base_value = solution_value

        self.final_solution = self.base_solution.copy_writable()

        if self.base_value > self.best_value:
            self.best_value = self.base_value
            self.iterations_since_improvement = 0
        else:
            self.iterations_since_improvement += 1

    def continue_solving(self, iterations):
        if iterations % self.repetitions == 0:
            self.temperature = self.temperature * self.alpha
        
        continue_solving = True

        if self.temperature < self.min_temperature:
            print("Reached minimum temperature {}".format(iterations))
            continue_solving = False
        elif self.iterations_since_improvement >= 10000:
            print("Have not improved in 10000 iterations {}".format(iterations))
            continue_solving = False

        return continue_solving
    
    def accept_solution(self, neighbor_solution):
        neighbor_solution_value = neighbor_solution.compute_value()
        
        value_delta = (self.base_value - neighbor_solution_value) / (abs(self.base_value))

        try:
            accept_probability =  1.0 / math.exp((value_delta / self.temperature))
        except OverflowError:
            accept_probability = 0
        except ZeroDivisionError:
            accept_probability = 1
            

        return random.random() < accept_probability
        
class RecordToRecord():
    def __init__(self):
        self.base_value = -1000000.0
        self.best_value = -1000000.0

    def configure(self, deviation=0.05, max_iterations=100000):           
        self.deviation = deviation
        self.max_iterations = max_iterations        
    
    def set_base_solution(self, solution):
        solution_value = solution.compute_value()
        self.base_value = solution_value
        
        if solution_value > self.best_value:
            self.best_value = solution_value
            self.best_solution = solution.copy_writable()

            self.final_solution = self.best_solution

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