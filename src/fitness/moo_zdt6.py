import math

from math import sqrt, sin, pi, cos, exp

from fitness import moo_fitness
from fitness.moo_fitness import binary_phen_to_float


class moo_zdt6(moo_fitness.MooFitness):
    """
    Fitness function for the first problem (T_1) presented in
    [Zitzler2000].

    .. Zitzler, Eckart, Kalyanmoy Deb, and Lothar Thiele. Comparison
    of multiobjective evolutionary algorithms: Empirical results.
    Evolutionary computation 8.2 (2000): 173-195.
    """

    def moo_eval(self, phen):
        min_value = [0] * 10
        max_value = [1] * 10
        chromosome = binary_phen_to_float(phen, 30, min_value, max_value)

        f1 = 1 - exp(-4 * chromosome[0]) * \
            sin(6 * pi * chromosome[0]) ** 6
        summation = 0
        for i in range(1, len(chromosome)):
            summation += chromosome[i]

        g = 1 + 9 * (summation / (len(chromosome)-1)) ** 0.25
        h = 1 - (f1 / g) ** 2
        # Two objectives list
        objectives = [f1, (g * h)]
        return objectives

    def num_objectives(self):
        return 2
