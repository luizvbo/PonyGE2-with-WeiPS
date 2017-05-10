import math

from math import sqrt

from fitness import moo_fitness
from fitness.moo_fitness import phenotype_binary_decoder


class moo_zdt1(moo_fitness.MooFitness):

    """
    Fitness function for the first problem (T_1) presented in 
    [Zitzler2000].
    
    .. Zitzler, Eckart, Kalyanmoy Deb, and Lothar Thiele. Comparison 
    of multiobjective evolutionary algorithms: Empirical results.
    Evolutionary computation 8.2 (2000): 173-195.
    """

    def moo_eval(self, phen):
        min_value = [0] * 30
        max_value = [1] * 30
        chromo = phenotype_binary_decoder(phen, 30, min_value, max_value)

        summation = 0
        for i in range (1, len(chromo)):
            summation += chromo[i]

        g = 1 + 9 * summation / (len(chromo) - 1.0)
        h = 1 - math.sqrt(chromo[0] / g)
        # Two objectives list
        objectives = [chromo[0], (g * h)]
        return objectives

    def min_pareto_front(self):
        return [0, 0]

    def max_pareto_front(self):
        return [1, 1]

    def p_star_size(self):
        return 500

    def pareto_front_point(self, objectives):
        return [objectives[0], 1-sqrt(objectives[0])]
