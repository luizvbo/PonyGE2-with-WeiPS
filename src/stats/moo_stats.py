import random
from copy import copy
from sys import stdout
from time import time

import numpy as np
from math import sqrt

from numpy.core.memmap import memmap

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

    if not trackers.best_ever or pf_pop > trackers.best_ever:
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
    fitnesses = [i.fitness for i in individuals]
    stats['ave_fitness'] = np.nanmean(fitnesses)
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

    print("\n\nBest:\n  Fitness:\t", trackers.best_ever.fitness)
#    print("  Objectives:", trackers.best_ever.objectives)
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
                if euclidean_distance(pf_solutions[i], pf_solutions[j]) < params['SIGMA_SHARE']:
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
    return sqrt(sum([(x1 - x2)**2 for x1, x2 in zip(p1.fitness, p2.fitness)]))


class ParetoFront:
    """
    This class makes a set of solutions in the Pareto front emulate
    an inidividual, such that the writing of the statistics in file
    can read the informations from the Pareto front as an individual.
    """
    def __init__(self, pf_solutions):
        self.phenotype = PhenotypeParser(pf_solutions)
        self.genome = ObjectiveParser(pf_solutions)
        self.tree = "" #TreeParser(pf_solutions)
        self.fitness = uniform_distribution(pf_solutions)
#        self.objectives = ObjectiveParser(pf_solutions)

    def __lt__(self, other):
        if np.isnan(self.fitness):
            # Self.fitness is not a number, return False as it doesn't
            # matter what the other fitness is.
            return False
        else:
            if np.isnan(other.fitness):
                return False
            else:
                return other.fitness < self.fitness


class PhenotypeParser:
    def __init__(self, pf_solutions):
        self.solutions = pf_solutions

    def __str__(self):
        ret_str = ""
        return ''.join([str(ind.phenotype)+"\n" for ind in self.solutions])


class GenomeParser(PhenotypeParser):
    def __str__(self):
        ret_str = ""
        return ''.join([str(ind.genome)+"\n" for ind in self.solutions])


class TreeParser(PhenotypeParser):
    def __str__(self):
        ret_str = ""
        return ''.join([str(ind.tree) + "\n" for ind in self.solutions])


class ObjectiveParser(PhenotypeParser):
    def __str__(self):
        ret_str = ""
        return ''.join([str(ind.fitness) + "\n" for ind in self.solutions])
