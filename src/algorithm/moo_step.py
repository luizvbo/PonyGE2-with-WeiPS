from operators.crossover import crossover
from operators.mutation import mutation
from operators.selection import selection

from operators.moo_replacement import weips_replacement, nsga2_replacement
from operators.moo_selection import weips_selection

from fitness.evaluation import evaluate_fitness
from algorithm.parameters import params


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
    individuals = nsga2_replacement(new_pop, individuals)

    # Generate statistics for run so far
    params['STATISTICS'].get_stats(individuals)

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
