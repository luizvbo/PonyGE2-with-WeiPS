from copy import copy
from sys import stdout
from time import time

import numpy as np
from math import sqrt

from algorithm.parameters import params
from operators.moo_selection import first_pareto_front
from stats.stats import stats
from utilities.algorithm.state import create_state
from utilities.stats import trackers
from utilities.stats.file_io import save_stats_to_file, save_stats_headers, \
    save_best_ind_to_file

p_star = []


def get_stats(individuals, end=False):
    """
    Generate the statistics for an evolutionary run. Save statistics to
    utilities.trackers.stats_list. Print statistics. Save fitness plot
    information.

    :param individuals: A population of individuals for which to generate
    statistics.
    :param end: Boolean flag for indicating the end of an evolutionary run.
    :return: Nothing.
    """

    # Find the Pareto front from the population and convert in a
    # *pareto_front* object
    non_dominated, dominated = first_pareto_front(individuals)
    pf_pop = ParetoFront(non_dominated)

    # if not trackers.best_ever or pf_pop > trackers.best_ever:
    # Save best individual in trackers.best_ever.
    trackers.best_ever = pf_pop

    if end or params['VERBOSE'] or not params['DEBUG']:
        # Update all stats.
        update_stats(individuals, end)

    # Print statistics
    if params['VERBOSE'] and not end:
        print_generation_stats()

    elif not params['SILENT']:
        # Print simple display output.
        perc = stats['gen'] / (params['GENERATIONS'] + 1) * 100
        stdout.write("Evolution: %d%% complete\r" % perc)
        stdout.flush()

    # Generate test fitness on regression problems
    if hasattr(params['FITNESS_FUNCTION'], "training_test") and end:
        trackers.best_ever.training_fitness = copy(trackers.best_ever.fitness)
        test_pf = []
        for ind in trackers.best_ever.pf_solutions:
            test_pf.append(params['FITNESS_FUNCTION'](ind, dist='test'))
        trackers.best_ever.test_fitness = TestFitness(test_pf)
        # params['FITNESS_FUNCTION'](trackers.best_ever, dist='test')
        trackers.best_ever.fitness = trackers.best_ever.training_fitness

    # Save stats to list.
    if params['VERBOSE'] or (not params['DEBUG'] and not end):
        trackers.stats_list.append(copy(stats))

    # Save stats to file.
    if not params['DEBUG']:
        if stats['gen'] == 0:
            save_stats_headers(stats)
        save_stats_to_file(stats, end)
        if params['SAVE_ALL']:
            save_best_ind_to_file(stats, end, stats['gen'])
        elif params['VERBOSE'] or end:
            save_best_ind_to_file(stats, end, "best")

    if end and not params['SILENT']:
        print_final_stats()

    if params['SAVE_STATE'] and not params['DEBUG'] and \
                            stats['gen'] % params['SAVE_STATE_STEP'] == 0:
        # Save the state of the current evolutionary run.
        create_state(individuals)


def update_stats(individuals, end):
    """
    Update all stats in the stats dictionary.

    :param individuals: A population of individuals.
    :param end: Boolean flag for indicating the end of an evolutionary run.
    :return: Nothing.
    """

    if not end:
        # Time Stats
        trackers.time_list.append(time() - stats['time_adjust'])
        stats['time_taken'] = trackers.time_list[-1] - \
                              trackers.time_list[-2]
        stats['total_time'] = trackers.time_list[-1] - \
                              trackers.time_list[0]

    # Population Stats
    stats['total_inds'] = params['POPULATION_SIZE'] * (stats['gen'] + 1)
    stats['invalids'] = len(trackers.invalid_cache)
    if params['CACHE']:
        stats['unique_inds'] = len(trackers.cache)
        stats['unused_search'] = 100 - stats['unique_inds'] / \
                                            stats['total_inds'] * 100

    # Genome Stats
    genome_lengths = [len(i.genome) for i in individuals]
    stats['max_genome_length'] = np.nanmax(genome_lengths)
    stats['ave_genome_length'] = np.nanmean(genome_lengths)
    stats['min_genome_length'] = np.nanmin(genome_lengths)

    # Used Codon Stats
    codons = [i.used_codons for i in individuals]
    stats['max_used_codons'] = np.nanmax(codons)
    stats['ave_used_codons'] = np.nanmean(codons)
    stats['min_used_codons'] = np.nanmin(codons)

    # Tree Depth Stats
    depths = [i.depth for i in individuals]
    stats['max_tree_depth'] = np.nanmax(depths)
    stats['ave_tree_depth'] = np.nanmean(depths)
    stats['min_tree_depth'] = np.nanmin(depths)

    # Tree Node Stats
    nodes = [i.nodes for i in individuals]
    stats['max_tree_nodes'] = np.nanmax(nodes)
    stats['ave_tree_nodes'] = np.nanmean(nodes)
    stats['min_tree_nodes'] = np.nanmin(nodes)

    # Fitness Stats
    ave_fitness = [list() for i in range(params['FITNESS_FUNCTION'].num_objectives())]
    for i in individuals:
        if isinstance(i.fitness, list):
            for j in range(len(ave_fitness)):
                ave_fitness[j].append(i.fitness[j])
    # fitnesses = [i.fitness for i in individuals]
    stats['ave_fitness'] = [np.nanmean(i) for i in ave_fitness]
        # np.nanmean(fitnesses)
    stats['best_fitness'] = trackers.best_ever.fitness


def print_generation_stats():
    """
    Print the statistics for the generation and individuals

    :return: Nothing.
    """

    print("______\n")
    for stat in sorted(stats.keys()):
        print(" ", stat, ": \t", stats[stat])
    print("\n")


def print_final_stats():
    """
    Prints a final review of the overall evolutionary process.

    :return: Nothing.
    """
    if hasattr(params['FITNESS_FUNCTION'], "training_test"):
        print("\n\nBest:\n  Training fitness:\t",
              trackers.best_ever.training_fitness)
        print("  Test fitness:\t\t", trackers.best_ever.test_fitness)
    else:
        print("\n\nBest:\n  Fitness:\t", trackers.best_ever.fitness)

    # print("\n\nBest:\n  Fitness:\t", trackers.best_ever.fitness)
    # print("  Objectives:", trackers.best_ever.objectives)
    print("  Phenotype:", trackers.best_ever.phenotype)
    print("  Genome:", trackers.best_ever.genome)
    print_generation_stats()


def uniform_distribution(pf_solutions):
    niche_count = []
    mean_nc = 0
    for i in range(len(pf_solutions)):
        nc = 0
        for j in range(len(pf_solutions)):
            if i != j:
                if euclidean_distance(pf_solutions[i].fitness,
                                      pf_solutions[j].fitness) < params['SIGMA_SHARE']:
                    nc += 1
        niche_count.append(nc)
        mean_nc += nc
    mean_nc /= float(len(pf_solutions))
    s_nc = 0
    for nc in niche_count:
        s_nc += (nc - mean_nc) ** 2
    s_nc = sqrt(s_nc/(len(pf_solutions) - 1))
    return 1/(1+s_nc)


def euclidean_distance(p1, p2):
    return sqrt(sum([(x1 - x2)**2 for x1, x2 in zip(p1, p2)]))


class TestFitness:
    def __init__(self, fitness_list):
        self.fitness_list = fitness_list

    def __str__(self):
        return ''.join([str(fitness) + "\n" for fitness in self.fitness_list])


class ParetoFront:
    """
    This class makes a set of solutions in the Pareto front emulate
    an individual, such that the writing of the statistics in file
    can read the information from the Pareto front as an individual.
    """
    def __init__(self, pf_solutions):
        self.phenotype = PhenotypeParser(pf_solutions)
        self.genome = GenomeParser(pf_solutions)
        self.tree = TreeParser(pf_solutions)
        self.fitness = ObjectiveParser(pf_solutions)
        self.pf_solutions = pf_solutions

    def __lt__(self, other):

        if self.fitness is None:
            # Self.fitness is None, return False as it doesn't
            # matter what the other fitness is.
            return False
        else:
            if other.fitness is None:
                return False
            else:
                return len(other.fitness) > len(self.fitness)


class PhenotypeParser:
    def __init__(self, pf_solutions):
        self.solutions = pf_solutions

    def __str__(self):
        ret_str = ''.join([str(ind.phenotype)+"\n" for ind in self.solutions])
        return ret_str


class GenomeParser(PhenotypeParser):
    def __str__(self):
        # ret_str = ''.join([str(ind.genome)+"\n" for ind in self.solutions])
        ret_str = ""
        return ret_str


class TreeParser(PhenotypeParser):
    def __str__(self):
        # ret_str = ''.join([str(ind.tree) + "\n" for ind in self.solutions])
        ret_str = ""
        return ret_str


class ObjectiveParser(PhenotypeParser):
    def __str__(self):
        ret_str = ''.join([str(ind.fitness) + "\n" for ind in self.solutions])
        return ret_str
