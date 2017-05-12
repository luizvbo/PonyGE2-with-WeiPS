import random

import numpy

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


def rawps_weight_initialisation(individuals):
    return None


def unpas_weight_initialisation(individuals):
    n_objectives = len(individuals[0].fitness)
    n_rows = params['WEIPS_NUMBER_WEIGHTS']
    return [[random.random() for e in range(n_objectives)] for e in n_rows]


def meps_weight_initialisation(individuals):
    n_objectives = len(individuals[0].fitness)
    length = params['WEIPS_LENGTH']
    n_rows = length**n_objectives
    coordinates = numpy.linspace(0, 1, length)
    coord_index = [0] * n_objectives
    weight_matrix = []

    for i in range(n_rows):
        tmp_weigh = []
        # Iterate through the coordinates
        for j in range(n_objectives):
            tmp_weigh.append(coordinates[coord_index[j]])
        weight_matrix.append(tmp_weigh)
        # Increment the index of the last coordinate
        coord_index[n_objectives-1] += 1
        # Check if we need to adjust the index of other coordinates
        for j in range(n_objectives-1,0,-1):
            if coord_index[j] >= len(coordinates):
                coord_index[j] = 0
                coord_index[j-1] += 1
            else:
                break
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
    for i in individuals:
        print(str(i.fitness[0])+","+str(i.fitness[1]), end=",")

    return individuals

