from fitness.regression import regression
from . import supervised_learning


class regression_bloat_control(regression):
    """
    Fitness function for regression using a multiobjective approach
    to control bloat. 
    We slightly specialise the function for supervised_learning.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, ind, dist="training"):
        objective_1 = super().__call__(ind, dist)
        objective_1 = float(objective_1)
        objective_2 = ind.nodes
        return [objective_1, objective_2]

    @staticmethod
    def value(fitness_vector, objective_index):
        if not isinstance(fitness_vector, list):
            return float("inf")
        return fitness_vector[objective_index]

    @staticmethod
    def num_objectives():
        return 2
