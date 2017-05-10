import random

from algorithm.parameters import params
from fitness.evaluation import evaluate_fitness
from operators.crossover import crossover
from operators.moo_replacement import weips_replacement
from operators.moo_selection import weips_selection, unpas_weight_initialisation, rawps_weight_initialisation, \
    meps_weight_initialisation
from operators.mutation import mutation
from operators.replacement import replacement, steady_state
from operators.selection import selection
from stats.stats import stats, get_stats


def nsga2_step(individuals):
    """
    Runs a single generation of the evolutionary algorithm process:
        Selection
        Variation
        Evaluation
        Replacement

    :param individuals: The current generation, upon which a single
    evolutionary generation will be imposed (P_t).
    :return: The next generation of the population.
    """

    # Select parents from the original population.
    parents = selection(individuals)

    # Crossover parents and add to the new population.
    cross_pop = crossover(parents)

    # Mutate the new population.
    new_pop = mutation(cross_pop)

    # Evaluate the fitness of the new population (Q_t).
    new_pop = evaluate_fitness(new_pop)

    # Replace the old population with the new population.
    individuals = replacement(new_pop, individuals)

    # Generate statistics for run so far
    params['STATISTICS'].get_stats(individuals)

    return individuals


def rawps_search_loop():
    return weips_search_loop(rawps_weight_initialisation)


def unpas_search_loop():
    return weips_search_loop(unpas_weight_initialisation)


def meps_search_loop():
    return weips_search_loop(meps_weight_initialisation)


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


def weips_step(individuals, weight_matrix):
    """
    Runs a single generation of the evolutionary algorithm process:
        Selection
        Variation
        Evaluation
        Replacement

    :param individuals: The current generation, upon which a single
    :param weight_matrix: The matrix of weights used by the selection method.
    :return: The next generation of the population.
    """

    # Size of the population
    pop_size = params['GENERATION_SIZE']

    # Select parents from the original population.
    parents = weips_selection(individuals, weight_matrix, pop_size)

    # Crossover parents and add to the new population.
    cross_pop = crossover(parents)

    # Mutate the new population.
    new_pop = mutation(cross_pop)

    # Evaluate the fitness of the new population (Q_t).
    new_pop = evaluate_fitness(new_pop)

    # Replace the old population with the new population.
    individuals = weips_replacement(new_pop, individuals, weight_matrix)

    # Generate statistics for run so far
    params['STATISTICS'].get_stats(individuals)

    return individuals
