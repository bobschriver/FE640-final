class Objective():
    def __init__(self):
        self.iterations = 0
        
    def done_solving(self, solution): 
        self.iterations += 1
        return self.iterations > 100
    
    def accept_solution(self, solution):
        solution_value = solution.compute_value()
        
        print("Iteration {} value {}".format(self.iterations, solution_value))
        print()
        
        return True