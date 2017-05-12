import math

from math import sqrt, sin, pi, cos

from fitness import moo_fitness
from fitness.moo_fitness import binary_phen_to_string


class moo_zdt5(moo_fitness.MooFitness):
    """
    Fitness function for the first problem (T_1) presented in 
    [Zitzler2000].

    .. Zitzler, Eckart, Kalyanmoy Deb, and Lothar Thiele. Comparison 
    of multiobjective evolutionary algorithms: Empirical results.
    Evolutionary computation 8.2 (2000): 173-195.
    """

    def moo_eval(self, phen):
        n_codon = [30]
        chromosome = binary_phen_to_string(phen, n_codon.extend([5] * 10))

        f1 = 1 + chromosome[0].count('1')
        g = 0
        for i in range(1, len(real_chromosome)):
            u = chromosome[i].count('1')
            if u < 5:
                g += 2 + u
            else:
                g += 1

        h = 1 / float(f1)
        # Two objectives list
        objectives = [f1, (g * h)]
        return objectives

    @staticmethod
    def v(self, x):
        return x.count('1')
