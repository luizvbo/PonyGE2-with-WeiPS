from math import isnan, isinf

import numpy as np
import abc

from operators.moo_replacement import first_pareto_front

np.seterr(all="raise")


class MooFitness:

    """
    Fitness function for multi-objective optimization problems. 
    The objective functions are defined in implementation of this  
    class. The control parameters regarding the problem, as number
    of input variables and their range are defined implicitly by 
    the grammar.

    This is an abstract class which exists just to be subclassed:
    should not be instantiated.
    """

    default_fitness = []
    # default_fitness = np.NaN

    def __call__(self, ind):
        """
        Note that math functions used in the solutions are imported from either
        utilities.fitness.math_functions or called from numpy.

        :param ind: An individual to be evaluated.
        :return: The fitness of the evaluated individual.
        """

        phen = ind.phenotype

        try:
            # the multi-objective fitness is defined as a list of
            # values, each one representing the output of one
            # objective function. The computation is made by the
            # function multi_objc_eval, implemented by a subclass,
            # according to the problem.
            fitness = self.moo_eval(phen)

        except (FloatingPointError, ZeroDivisionError, OverflowError,
                MemoryError):
            # FP err can happen through eg overflow (lots of pow/exp calls)
            # ZeroDiv can happen when using unprotected operators
            fitness = self.default_fitness
        except Exception as err:
            # other errors should not usually happen (unless we have
            # an unprotected operator) so user would prefer to see them
            print(err)
            raise

        # don't use "not fitness" here, because what if fitness = 0.0?!
        # if isnan(fitness):
        #     fitness = self.default_fitness

        return fitness

    @abc.abstractmethod
    def moo_eval(self, phen):
        """
        This method implements the fitness functions defined by the
        optimization problem being solved.
        
        :param phen: The phenotype defined by an individual
        :return: The resulting fitness
        """
        return

    @staticmethod
    def value(fitness_vector, objective_index):
        if not isinstance(fitness_vector, list):
            return float("inf")
        if isnan(fitness_vector[objective_index]):
            return float("inf")
        return fitness_vector[objective_index]

    @abc.abstractmethod
    def num_objectives(self):
        return


def binary_phen_to_float(phen, n_codon, min_value, max_value):
    """
    This method converts a phenotype, defined by a
    string of bits in a list of float values

    :param phen: Phenotype defined by a bit string 
    :param n_codon: Number of codons per gene, defined in the grammar
    :param min_value: Minimum value for a gene
    :param max_value: Maximum value for a gene
    :return: A list os float values, representing the chromosome
    """
    i = 0
    count = 0
    chromosome = []
    while i < len(phen):
        gene = phen[i:(i + n_codon)]
        # Convert the bit string in gene to an float/int
        gene_i = int(gene, 2)
        gene_f = float(gene_i) / (2 ** n_codon - 1)
        # Define the variation for the gene
        delta = max_value[count] - min_value[count]
        # Append the float value to the chromossome list
        chromosome.append(gene_f * delta + min_value[count])
        i = i + n_codon
        count += 1
    return chromosome


def binary_phen_to_string(phen, n_codon):
    """
    This method converts a phenotype, defined by a
    string of bits in a list of (string) bits.

    :param phen: Phenotype defined by a bit string 
    :param n_codon: List with the number of codons per gene, defined 
    by the problem.
    :return: A list os float values, representing the chromosome
    """
    i = 0
    count = 0
    chromosome = []
    while i < len(phen):
        gene = phen[i:(i + n_codon[count])]
        chromosome.append(gene)
        i = i + n_codon[count]
        count += 1
    return chromosome


def real_phen_to_float(phen, n_codon):
    """
    This method converts a phenotype, defined by a
    string of decimals with the same number of digits
    defined by n_codons

    :param phen: Phenotype defined by a string of decimals 
    :param n_codon: Number of codons per gene, defined in the grammar
    (don't forget the decimal point)
    :return: A list os float values, representing the chromosome
    """
    i = 0
    count = 0
    chromosome = []
    while i < len(phen):
        gene = phen[i:(i + n_codon)]
        # Append the float value to the chromossome list
        chromosome.append(float(gene))
        i = i + n_codon
        count += 1
    return chromosome
