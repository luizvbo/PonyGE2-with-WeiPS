from random import sample

from algorithm.parameters import params
from operators.moo_selection import compute_pareto_metrics, weips_selection, first_pareto_front, get_population_iqr


def nsga2_replacement(new_pop, old_pop):
    """
    Replaces the old population with the new population. The ELITE_SIZE best
    individuals from the previous population are appended to new pop regardless
    of whether or not they are better than the worst individuals in new pop.

    :param new_pop: The new population (e.g. after selection, variation, &
    evaluation).
    :param old_pop: The previous generation population, from which elites
    are taken.
    :return: The 'POPULATION_SIZE' new population with elites.
    """

    # Combine both populations (R_t = P_t union Q_t)
    new_pop.extend(old_pop)

    # Compute the pareto fronts and crowding distance
    pareto = compute_pareto_metrics(new_pop)

    # Size of the new population
    pop_size = params['POPULATION_SIZE']

    # New population to replace the last one
    temp_pop = []

    i = 0
    # The new population is filled with members from the Pareto fronts. The method
    # iterates from the first front until the last one, adding solutions until the
    # population is complete.
    while len(temp_pop) < pop_size:
        if len(pareto.fronts[i]) <= pop_size - len(temp_pop):
            # If the size of the front is smaller or equal to the number of solutions
            # needed to fulfill the population the method just adds them to the population
            temp_pop.extend(pareto.fronts[i])
        else:
            # Otherwise, we sort the solutions using the crowded-comparison operator in descending order
            # and choose the best solutions needed to fill all population slots.
            pareto.fronts[i] = sorted(pareto.fronts[i], key=lambda item: pareto.crowding_distance[item], reverse=True)
            # Number of individuals to add in temp to achieve the pop_size
            diff_size = pop_size - len(temp_pop)
            temp_pop.extend(pareto.fronts[i][:diff_size])
        i += 1
    return temp_pop


def weips_replacement(new_pop, old_pop, weight_matrix):
    """
    Replaces the old population with the new population. First *new_pop*
    
    The pareto front
    defined by the current population is copied to 

    :param new_pop: The new population (e.g. after selection, variation, &
    evaluation).
    :param old_pop: The previous generation population.
    :param weight_matrix: 
    :return: The 'POPULATION_SIZE' new population with elites.
    """

    # Combine both populations (R_t = P_t union Q_t)
    new_pop.extend(old_pop)

    # Size of the new population
    pop_size = params['POPULATION_SIZE']

    non_dominated_pop, dominated_pop = first_pareto_front(new_pop)

    if len(non_dominated_pop) < pop_size:
        final_pop = non_dominated_pop
        final_pop.extend(weips_selection(dominated_pop, weight_matrix,
                                         pop_size - len(non_dominated_pop), False))
    elif len(non_dominated_pop) > pop_size:
        final_pop = weips_selection(non_dominated_pop, weight_matrix,
                                    pop_size, False)
    else:
        final_pop = non_dominated_pop
    return final_pop
