import sys
sys.path.append("../src")

import ast
import re
import numpy

from os import listdir, path

from src.scripts.hv import HyperVolume
from stats.moo_stats import euclidean_distance


def read_experiment(exp_path, granularity=1):
    """
    Read an experiment, composed of multiple runs.
    :param exp_path: Path to the folder containing the output of each experiment
    :param granularity: Load the generations 1, 1+:granurality:, 1+2*:granularity:, ...
    :return: 
    """
    # Find list of all runs contained in the specified folder.
    run_path_list = [run for run in listdir(exp_path) if
                     path.isdir(path.join(exp_path, run))]
    # Read information regarding each run
    runs_data = []
    for run_path in run_path_list:
        files = listdir(path.join(exp_path, run_path))
        number_gen = len([1 for file_name in files if
                          re.search('^\d+', file_name) is not None ])
        evolution_info = EvolutionData()
        for i in range(1, number_gen, granularity):
            f = open(path.join(exp_path, run_path, str(i) + ".ftn"), 'r')
            training_fitness = None
            test_fitness = None
            data_flag = False
            for line in f:
                # In this case we are reading a Pareto front
                if data_flag:
                    # We found a blank line or other kind of data
                    if '[' not in line and ']' not in line:
                        data_flag = False
                    elif test_fitness is not None:
                        test_fitness.append(ast.literal_eval(line))
                    elif training_fitness is not None:
                        training_fitness.append(ast.literal_eval(line))
                else:
                    if re.search("Test fitness", line):
                        test_fitness = []
                        data_flag = True
                    elif re.search("itness", line):
                        training_fitness = []
                        data_flag = True
            if i < number_gen-1:
                evolution_info.add_fitness(training_fitness)
            else:
                evolution_info.add_last_gen_fitness(training_fitness, test_fitness)
        runs_data.append(evolution_info)
    return runs_data


def get_metric_from_exp_data(metric, exp_data, gen_list=None):
    metric_value = []
    for exp in exp_data:
        gen_value = []
        if gen_list is None:
            training_fitness = exp.training_fitness
        else:
            training_fitness = [exp.training_fitness[i] for i in gen_list]
        for pf in training_fitness:
            gen_value.append(metric(pf))
        metric_value.append(gen_value)
    return metric_value


def uniform_distribution(pareto_front):
    min_distance = [float("inf")] * len(pareto_front)
    for i in range(len(pareto_front)):
        for j in range(len(pareto_front)):
            if i != j:
                dist = euclidean_distance(pareto_front[i], pareto_front[j])
                if dist < min_distance[i]:
                    min_distance[i] = dist
    return numpy.std(numpy.array(min_distance))


def hyper_volume(pareto_front):
    hv = HyperVolume(pareto_front)
    return hv.compute(pareto_front)


def write_stats_in_file(stats_value, output_path):
    f = open(output_path, 'w')
    for row in stats_value:
        row_size = len(row)
        for cell in range(row_size):
            if cell < row_size-1:
                f.write(str(row[cell]) + ',')
            else:
                f.write(str(row[cell]) + '\n')
    f.close()


class EvolutionData:
    def __init__(self):
        self.test_fitness = None
        self.training_fitness = []

    def add_fitness(self, training_fitness):
        self.training_fitness.append(training_fitness)

    def add_last_gen_fitness(self, training_fitness, test_fitness):
        self.training_fitness.append(training_fitness)
        self.test_fitness = test_fitness

    def get_fitness(self, gen):
        return self.training_fitness[gen]

    def get_last_gen_fitness(self):
        return self.training_fitness[-1:], self.test_fitness


def main():
    """
    The main function for running the experiment manager. Calls all functions.

    :return: Nothing.
    """

    input_path = '/mnt/speed/results/fitness'
    output_paht = '/mnt/speed/results/stats'
    for f in listdir(input_path):
        print("Reading the experiment")
        exp = read_experiment(path.join(input_path, f), 250)
        print("Computing the HV metric")
        hv = get_metric_from_exp_data(hyper_volume, exp)
        print("Writing the results to the file")
        write_stats_in_file(hv, path.join(output_paht, "hv_" + f))

if __name__ == "__main__":
    main()