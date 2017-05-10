from __future__ import division

from random import sample, random, randint

import numpy

from algorithm.parameters import params
from collections import defaultdict


######################################
# Non-Dominated Sorting   (NSGA-II)  #
######################################

def nsga2_selection(population):
    """Apply NSGA-II selection operator on the *population*. Usually, the
    size of *population* will be larger than *k* because any individual
    present in *population* will appear in the returned list at most once.
    Having the size of *population* equals to *k* will have no effect other
    than sorting the population according to their front rank. The
    list returned contains references to the input *population*. For more
    details on the NSGA-II operator see [Deb2002]_.

    :param population: A population from which to select individuals.
    :returns: A list of selected individuals.

    .. [Deb2002] Deb, Pratab, Agarwal, and Meyarivan, "A fast elitist
       non-dominated sorting genetic algorithm for multi-objective
       optimization: NSGA-II", 2002.
    """

    selection_size = params['GENERATION_SIZE']
    tournament_size = params['TOURNAMENT_SIZE']

    # Initialise list of tournament winners.
    winners = []

    pareto = compute_pareto_metrics(population)

    while len(winners) < selection_size:
        # Return the single best competitor.
        winners.append(pareto_tournament(population, pareto, tournament_size))

    return winners


def compute_pareto_metrics(population):
    pareto = sort_nondominated(population)
    calculate_crowding_distance(pareto)
    return pareto


def pareto_tournament(population, pareto, tournament_size):
    """
    The Pareto tournament selection uses both the pareto front of the individual 
    and the crowding distance.
    
    :param population: A population from which to select individuals.
    :param pareto: 
    :param tournament_size: The size of the tournament.
    :return: The selected individuals.
    """
    participants = sample(population, tournament_size)
    best = None
    for participant in participants:
        if best is None or crowded_comparison_operator(participant, best, pareto):
            best = participant

    return best


def crowded_comparison_operator(individual, other_individual, pareto):
    if (pareto.rank[individual] < pareto.rank[other_individual]) or \
            (pareto.rank[individual] == pareto.rank[other_individual] and
                pareto.crowding_distance[individual] > pareto.crowding_distance[other_individual]):
        return True
    else:
        return False


def sort_nondominated(population):
    """Sort the first *k* *population* into different nondomination levels 
    using the "Fast Nondominated Sorting Approach" proposed by Deb et al.,
    see [Deb2002]_. This algorithm has a time complexity of :math:`O(MN^2)`, 
    where :math:`M` is the number of objectives and :math:`N` the number of 
    individuals.

    :param population: A list of individuals to select from.
    
    :returns: A list of Pareto fronts (lists), the first list includes 
              nondominated individuals.

    .. [Deb2002] Deb, Pratab, Agarwal, and Meyarivan, "A fast elitist
       non-dominated sorting genetic algorithm for multi-objective
       optimization: NSGA-II", 2002.
       
    """
    pareto = ParetoInfo(population)
    # The naming *p* and *q* is the same adopted in [Deb2002]_
    for p in population:
        # Compute the minimum and maximum fitness values in the population
        for i in range(pareto.n_objectives):
            if p.fitness[i] < pareto.min_fitness[i]:
                pareto.min_fitness[i] = p.fitness[i]
            if p.fitness[i] > pareto.max_fitness[i]:
                pareto.max_fitness[i] = p.fitness[i]
        # Compute the domination counter of p
        for q in population:
            if dominates(p, q):
                # Add *q* to the set of solutions dominated by *p*
                pareto.dominated_solutions[p].append(q)
            elif dominates(q, p):
                # Increment the domination counter of p
                pareto.update_domination_count(p, True)
        # *p* belongs to the first front
        if pareto.get_domination_count(p) == 0:
            pareto.fronts[0].append(p)
            pareto.rank[p] = 0
    # Initialize the front counter
    i = 0
    while len(pareto.fronts[i]) > 0:
        big_q = []
        for p in pareto.fronts[i]:
            for q in pareto.dominated_solutions[p]:
                pareto.update_domination_count(q, False)
                if pareto.get_domination_count(q) == 0:
                    pareto.rank[q] = i + 1
                    big_q.append(q)
        i += 1
        pareto.fronts.append(big_q)
    return pareto


def dominates(individual1, individual2):
    """
    Returns whether or not *indvidual1* dominates *indvidual2*.

    :param individual1: The individual that would be dominated.
    :param individual2: The individual dominant.
    :returns: :obj:`True` if indvidual_2 dominates indvidual_1, :obj:`False`
              otherwise.
    """
    not_equal = False
    for ind1_value, ind2_value in zip(individual1.fitness, individual2.fitness):
        if ind1_value > ind2_value:
            return False
        elif ind1_value < ind2_value:
            not_equal = True
    return not_equal


def calculate_crowding_distance(pareto):
    """
    Compute the crowding distance of each individual in each Pareto front.
    The value is stored inside the dictionary *crowding_distance* kept by
    the *pareto_fronts*.
    
    :param pareto: 
    """
    for front in pareto.fronts:
        if len(front) > 0:
            solutions_num = len(front)

            for individual in front:
                pareto.crowding_distance[individual] = 0

            for m in range(pareto.n_objectives):  # len(front[0].fitness)):
                front = sorted(front, key=lambda item: item.fitness[m])
                pareto.crowding_distance[front[0]] = float("inf")
                pareto.crowding_distance[front[solutions_num - 1]] = float("inf")
                for index, value in enumerate(front[1:solutions_num - 1]):
                    pareto.crowding_distance[front[index]] = \
                        (pareto.get_crowding_distance(front[index + 1]) -
                         pareto.get_crowding_distance(front[index - 1])) / \
                        (pareto.max_fitness[m] - pareto.min_fitness[m])


class ParetoInfo:
    def __init__(self, population):
        self.fronts = [[]]

        self.rank = dict()
        self.domination_count = dict()
        self.crowding_distance = dict()

        self.dominated_solutions = defaultdict(list)

        self.n_objectives = len(population[0].fitness)
        self.min_fitness = [float('inf')] * self.n_objectives
        self.max_fitness = [float('-inf')] * self.n_objectives

    def update_domination_count(self, individual, should_increment=True):
        """
        Update the domination count of the *individual* by incrementing 
        (*should_increment*=:obj:`True`) or decrementing (*should_increment*=:obj:`False`).
        
        :param individual: The referring individual 
        :param should_increment: Indicates if the methods increment or decrement the value.
        """
        if individual in self.domination_count:
            if should_increment:
                self.domination_count[individual] += 1
            else:
                self.domination_count[individual] -= 1
        else:
            if should_increment:
                self.domination_count[individual] = 1
            else:
                self.domination_count[individual] = -1

    def get_domination_count(self, individual):
        """
        Avoids references to unitialised positions in the dictionary.
        
        :param individual: Individual used as key in the dictionar.   
        :return: The value regarding the key, if any, or 0 otherwise.
        """
        if individual in self.domination_count:
            return self.domination_count[individual]
        return 0

    def get_crowding_distance(self, individual):
        """
        Avoids references to unitialised positions in the dictionary.

        :param individual: Individual used as key in the dictionar.   
        :return: The value regarding the key, if any, or 0 otherwise.
        """
        if individual in self.crowding_distance:
            return self.crowding_distance[individual]
        return 0


#######################################
#  Weighted Pareto Selection (WeiPS)  #
#######################################


def weips_selection(population, weight_matrix, selection_size):
    """
    Explain the Weighted Pareto Selection (WeiPS) method here

    :param population: A population from which to select individuals.
    :param weight_matrix: Weights used during the selection.
    :param selection_size: 
    :returns: A list of selected individuals. 
    """
    tournament_size = params['TOURNAMENT_SIZE']

    # Initialise list of tournament winners.
    winners = []

    while len(winners) < selection_size:
        # Return the single best competitor.
        winners.append(weips_tournament(population, weight_matrix, tournament_size))

    return winners


def weips_tournament(population, weight_matrix, tournament_size):
    """
    The Pareto tournament selection uses both the pareto front of the individual 
    and the crowding distance.

    :param population: A population from which to select individuals.
    :param weight_matrix: The matrix of weights used by the selection method.
    :param tournament_size: The size of the tournament.
    :return: The selected individuals.
    """
    participants = sample(population, tournament_size)
    best = None
    for participant in participants:
        if best is None or weips_comparison_operator(participant, best, weight_matrix):
            best = participant

    return best


def weips_comparison_operator(individual, other_individual, weight_matrix=None):
    n_objectives = len(individual.fitness)
    # If the matrix of weights do not exist, the weights are sampled uniformly
    if weight_matrix is None:
        weights = [random() for i in range(n_objectives)]
    # Otherwise, a set of weights is selected uniformly from the matrix
    else:
        weights = weight_matrix[randint(0, len(weight_matrix) - 1)]
    # Compute the fitness induced by the weights for each individual
    individual_f = sum([a * b for a, b in zip(individual.fitness, weights)])
    other_individual_f = sum([a * b for a, b in zip(other_individual.fitness, weights)])
    return individual_f < other_individual_f


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


def first_pareto_front(population):
    nondominated_pop = []
    dominated_pop = []

    i = 0
    while i < len(population):
        nondominated = True
        for individual in population[i + 1:]:
            if dominates(individual, population[i]):
                nondominated = False
                break
        if nondominated:
            nondominated_pop.append(population[i])
        else:
            dominated_pop.append(population[i])
        i += 1
    return nondominated_pop, dominated_pop
