import math

from math import sqrt

from fitness import moo_fitness
from fitness.moo_fitness import binary_phen_to_float


class moo_zdt2(moo_fitness.MooFitness):
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
        real_chromosome = binary_phen_to_float(phen, 30, min_value, max_value)

        summation = 0
        for i in range(1, len(real_chromosome)):
            summation += real_chromosome[i]

        g = 1 + 9 * summation / (len(real_chromosome) - 1.0)
        h = 1 - (real_chromosome[0] / g)**2
        # Two objectives list
        objectives = [real_chromosome[0], (g * h)]
        return objectives

    def num_objectives(self):
        return 2