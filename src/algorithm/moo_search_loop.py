import random

import numpy
from builtins import range

from algorithm.moo_step import weips_step
from algorithm.parameters import params
from fitness.evaluation import evaluate_fitness

from stats.stats import stats


def rawps_search_loop():
    return weips_search_loop(rawps_weight_initialisation)


def unpas_search_loop():
    return weips_search_loop(unpas_weight_initialisation)


def meps_search_loop():
    return weips_search_loop(meps_weight_initialisation)


def stratmeps_search_loop():
    return weips_search_loop(stratmeps_weight_initialisation)


def rawps_weight_initialisation(individuals):
    return None


def unpas_weight_initialisation(individuals):
    n_objectives = len(individuals[0].fitness)
    n_rows = params['WEIPS_NUMBER_WEIGHTS']
    weight_matrix = []
    for i in range(n_rows):
        rnd_weight = [random.random() for e in range(n_objectives)]
        sum_weights = sum(rnd_weight)
        for e in range(n_objectives):
            rnd_weight[e] /= sum_weights
        weight_matrix.append(rnd_weight)
    return weight_matrix


def meps_weight_initialisation(individuals):
    # Number of objectives defined by the problem
    n_objectives = len(individuals[0].fitness)
    # Number of divisions in the grid (by objective)
    length = params['WEIPS_LENGTH']
    coordinates = numpy.linspace(0, 1, length)
    current_index = [0] * (n_objectives - 1)
    current_index.append(length - 1)
    max_index = [length - 1] * (n_objectives - 1)
    weight_matrix = []

    while current_index[0] <= max_index[0]:
        tmp_weigh = []
        # Concatenate the weights and append to the matrix
        for i in range(n_objectives):
            tmp_weigh.append(coordinates[current_index[i]])
        weight_matrix.append(tmp_weigh)
        # Increment the last but one index
        current_index[n_objectives-2] += 1
        # Check if we need to adjust the index of other coordinates
        for i in range(n_objectives - 2, 0, -1):
            if current_index[i] > max_index[i]:
                current_index[i] = 0
                current_index[i-1] += 1
                if current_index[i-1] <= max_index[i-1]:
                    summation = 0
                    for j in range(i):
                        summation += current_index[j]
                    for j in range(i, n_objectives - 1):
                        max_index[j] = length - summation - 1
            else:
                break
        # The index of the last coordinate is simetric to the last but one
        current_index[n_objectives - 1] = max_index[n_objectives - 2] - current_index[n_objectives - 2]
    return weight_matrix


def stratmeps_weight_initialisation(individuals):
    weight_matrix = meps_weight_initialisation(individuals)
    n_objectives = params['FITNESS_FUNCTION'].num_objectives()
    for i in range(len(weight_matrix)):
        weight_matrix[i] = [random.random() * weight_matrix[i][j] for j in range(n_objectives)]
        sum_weights = sum(weight_matrix[i])
        for j in range(n_objectives):
            weight_matrix[i][j] = weight_matrix[i][j]/sum_weights
    return weight_matrix


def weips_search_loop(weight_init_method):
    """
    This is a standard search process for an evolutionary algorithm. Loop over
    a given number of generations.
    
    :param: weight_initialisation: Defines how the weights are initialized. 
    :return: The final population after the evolutionary process has run for
    the specified number of generations.
    """

    # Initialise population
    individuals = params['INITIALISATION'](params['POPULATION_SIZE'])

    # Evaluate initial population
    individuals = evaluate_fitness(individuals)

    # Generate statistics for run so far
    params['STATISTICS'].get_stats(individuals)

    # Generate the uniformly distributed weights
    weight_matrix = weight_init_method(individuals)

    # Traditional GE
    for generation in range(1, (params['GENERATIONS']+1)):
        stats['gen'] = generation

        # New generation
        individuals = weips_step(individuals, weight_matrix)

    return individuals

