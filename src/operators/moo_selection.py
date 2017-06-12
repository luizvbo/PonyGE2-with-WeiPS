from __future__ import division

from math import isnan
from random import sample, random, randint

from algorithm.parameters import params
from collections import defaultdict
from utilities.fitness.math_functions import percentile

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

    # The flag "INVALID_SELECTION" allows for selection of invalid individuals.
    if params['INVALID_SELECTION']:
        available = population
    else:
        available = [i for i in population if not i.invalid]

    pareto = compute_pareto_metrics(available)

    while len(winners) < selection_size:
        # Return the single best competitor.
        winners.append(pareto_tournament(available, pareto, tournament_size))

    return winners


def compute_pareto_metrics(population):
    pareto = sort_non_dominated(population)
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


def sort_non_dominated(population):
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

    # max_fitness = [float('-inf')] * pareto.n_objectives
    # min_fitness = [float('inf')] * pareto.n_objectives

    # Compute the IQR+1 value used to normalize the crowding distance
    pareto.compute_iqr(population)

    # The naming *p* and *q* is the same adopted in [Deb2002]_
    for p in population:

        # Compute the minimum and maximum fitness values in the population
        # if isinstance(p.fitness, list):
        #     for i in range(pareto.n_objectives):
        #         if p.fitness[i] < min_fitness[i]:
        #             min_fitness[i] = p.fitness[i]
        #         if p.fitness[i] > max_fitness[i]:
        #             max_fitness[i] = p.fitness[i]

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

    # pareto.compute_delta_fitness(min_fitness, max_fitness)

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
    :returns: :obj:`True` if indvidual_1 dominates indvidual_2, :obj:`False`
              otherwise.
    """
    not_equal = False

    if not isinstance(individual1.fitness, list):
        return False
    if not isinstance(individual2.fitness, list):
        return True
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
                front = sorted(front, key=lambda item: params['FITNESS_FUNCTION'].value(item.fitness, m))
                pareto.crowding_distance[front[0]] = float("inf")
                pareto.crowding_distance[front[solutions_num - 1]] = float("inf")
                for index in range(1, solutions_num - 1):
                    # print(pareto.crowding_distance[front[index]], end=" ")
                    # print(params['FITNESS_FUNCTION'].value(front[index + 1].fitness, m), end=" ")
                    # print(params['FITNESS_FUNCTION'].value(front[index - 1].fitness, m), end=" ")
                    # x = (params['FITNESS_FUNCTION'].value(front[index + 1].fitness, m) -
                    #      params['FITNESS_FUNCTION'].value(front[index - 1].fitness, m))
                    # print(x, end="\n")
                    pareto.crowding_distance[front[index]] += \
                        (params['FITNESS_FUNCTION'].value(front[index + 1].fitness, m) -
                         params['FITNESS_FUNCTION'].value(front[index - 1].fitness, m)) / pareto.fitness_iqr[m]


class ParetoInfo:
    def __init__(self, population):
        self.fronts = [[]]

        self.rank = dict()
        self.domination_count = dict()
        self.crowding_distance = dict()

        self.dominated_solutions = defaultdict(list)

        self.n_objectives = params['FITNESS_FUNCTION'].num_objectives()
        self.fitness_iqr = [0] * self.n_objectives

    # def compute_delta_fitness(self, min_fitness, max_fitness):
    #     self.delta_fitness = [max_fitness[i] - min_fitness[i] for i in range(self.n_objectives)]

    def compute_iqr(self, population):
        self.fitness_iqr = get_population_iqr(population, self.n_objectives)
        # If the IQR value is zero, we replace it for 1---which is equivalent to disregard
        # the normalization process for that objective dimension
        self.fitness_iqr = [1 if i == 0 else i for i in self.fitness_iqr]

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

    # The flag "INVALID_SELECTION" allows for selection of invalid individuals.
    if params['INVALID_SELECTION']:
        available = population
    else:
        available = [i for i in population if not i.invalid]

    while len(winners) < selection_size:
        # Return the single best competitor.
        winners.append(weips_tournament(available, weight_matrix, tournament_size))

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
    pop_iqr = get_population_iqr(population, params['FITNESS_FUNCTION'].num_objectives())
    # If the IQR value is zero, we replace it for 1---which is equivalent to disregard
    # the normalization process for that objective dimension
    pop_iqr = [1 if i == 0 else i for i in pop_iqr]

    participants = sample(population, tournament_size)

    best = None
    for participant in participants:
        if best is None or weips_comparison_operator(participant, best, pop_iqr, weight_matrix):
            best = participant

    return best


def weips_comparison_operator(individual, other_individual, population_iqr, weight_matrix=None):
    """
    Compare two individuals according to the WeiPS method. The objectives are aggregated by a weighted sum. 
    The weights are normalized using the IQR of the objectives defined over the population.
    :param individual: The reference individual
    :param other_individual: The second individual to be compared
    :param population_iqr: The IQR of the population used to normalize the weights
    :param weight_matrix: The weight matrix used to compute the fitness
    :return: *True* if the aggregated fitness of the first individual is smaller than the aggregated fitness of 
    the second individual and *False* otherwise. 
    """
    # Check for invalid individuals (with nan fitness)
    if not isinstance(individual.fitness, list):
        return False
    if not isinstance(other_individual.fitness, list):
        return True

    n_objectives = len(individual.fitness)
    # If the matrix of weights do not exist, the weights are sampled uniformly
    if weight_matrix is None:
        weights = [random() for i in range(n_objectives)]
        sum_weights = sum(weights)
        for e in range(n_objectives):
            weights[e] /= sum_weights
    # Otherwise, a set of weights is selected randomly from the matrix
    else:
        weights = weight_matrix[randint(0, len(weight_matrix) - 1)]
    # Normalize the weights according to the IQR of the objectives
    weights = [a / b for a, b in zip(weights, population_iqr)]
    # Compute the fitness induced by the weights for each individual
    individual_f = sum([a * b for a, b in zip(individual.fitness, weights)])
    other_individual_f = sum([a * b for a, b in zip(other_individual.fitness, weights)])
    return individual_f < other_individual_f


def first_pareto_front(population):
    non_dominated_pop = []
    dominated_pop = []

    for i in range(len(population)):
        non_dominated = True
        for j in range(len(population)):
            if i != j and dominates(population[j], population[i]):
                non_dominated = False
                break
        if non_dominated:
            non_dominated_pop.append(population[i])
        else:
            dominated_pop.append(population[i])
        i += 1
    return non_dominated_pop, dominated_pop


def get_population_iqr(population, n_objectives):
    """
    Compute the interquartile range (IQR) of the population regarding
    each objective. 
    :param population: The input population 
    :param n_objectives: Total number of objectives
    :return: List with the IQR regarding each objective 
    """
    iqr = [0] * n_objectives
    for m in range(n_objectives):
        sorted_pop = sorted(population, key=lambda ind: params['FITNESS_FUNCTION'].value(ind.fitness, m))
        iqr[m] = (params['FITNESS_FUNCTION'].value(percentile(sorted_pop, 75).fitness, m) -
                  params['FITNESS_FUNCTION'].value(percentile(sorted_pop, 25).fitness, m))
    return iqr
