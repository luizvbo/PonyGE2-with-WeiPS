import sys

from scripts.moo_plots import plot_pf, mean_std_plot
from math import sin, pi, sqrt, exp
from scripts.hv import HyperVolume
from os.path import os
from builtins import dict, sorted
from numpy import arange

sys.path.append("../src")

import re
import numpy

from os import listdir, path, scandir

from stats.moo_stats import euclidean_distance

input_path = '/home/luiz/Dados/Trabalho/Pesquisa/Publicacoes/2017/MOGP/results/new'
output_path = '/tmp/stats'
problems = ["zdt1", "zdt2","zdt3", 
            "zdt4", "zdt5", "zdt6",
#             "DowNorm", "Keijzer6", "Paige1", "TowerNorm", "Vladislavleva4"
           ] 
            


def zdt1(x):
    g = 1.0
    h = 1 - sqrt(x / g)
    return [x, g * h]


def zdt2(x):
    g = 1.0
    h = 1 - (x / g) ** 2
    return [x, g * h]


def zdt3(x):
    g = 1.0
    h = 1 - sqrt(x / g) - (x / g) * sin(10 * pi * x)
    return [x, g * h]


def zdt4(x):
    g = 1.00
    h = 1 - sqrt(x / g)
    return [x, g * h]


def zdt5(x):
    # x is an integer in [1,31]
    g = 10.0
    h = 1.0 / x
    return [x, g * h]


def zdt6(x):
    f1 = 1 - exp(-4 * x) * sin(6 * pi * x) ** 6
    g = 1.0
    h = 1 - (f1 / g) ** 2
    return [f1, g * h]


min_points = {'zdt1': [0, 0],
             'zdt2': [0, 0],
             'zdt3': [0, -1],
             'zdt4': [0, 0],
             'zdt5': [1, 0],
             'zdt6': [0, 0]
             }

max_points = {'zdt1': [1, 10],
              'zdt2': [1, 10],
              'zdt3': [1, 10],
              'zdt4': [1, 385],
              'zdt5': [31, 60],
              'zdt6': [1, 10]
             }


pf_functions = {'zdt1': zdt1,
                'zdt2': zdt2,
                'zdt3': zdt3,
                'zdt4': zdt4,
                'zdt5': zdt5,
                'zdt6': zdt6
                }


def read_experiment(exp_path, granularity=1):
    """
    Read an experiment, composed of multiple runs.
    :param exp_path: Path to the folder containing the output of each experiment
    :param granularity: Read generations 1, 1+granurality, 1+2*granularity, ...
    :return: 
    """
    # Find list of all runs contained in the specified folder.
    run_path_list = [run.path for run in scandir(exp_path) if
                     path.isdir(run.path)]
    run_path_list = sorted(run_path_list)
    # Read information regarding each run
    runs_data = []
    for run_path in run_path_list:
        files = listdir(path.join(exp_path, run_path))
        # PonyGE generates one file per generation plus the generation 0
        number_files = len([1 for file_name in files if
                          re.search('^\d+', file_name) is not None ])
        evolution_info = EvolutionData()
        if granularity < 0:
            generations = [number_files-1]
        else:
            generations = list(range(0, number_files, granularity))
        # Ensure the last generation is accounted
#         generations.append(number_gen-1)
        for i in generations:
            f = open(path.join(exp_path, run_path, str(i) + ".txt"), 'r')
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
                        test_fitness.append(list_parser(line))
                    elif training_fitness is not None:
                        training_fitness.append(list_parser(line))
                else:
                    if re.search("Test fitness", line):
                        test_fitness = []
                        data_flag = True
                    elif re.search("itness", line):
                        training_fitness = []
                        data_flag = True
            if i < number_files-1:
                evolution_info.add_tr_fitness(training_fitness)
            else:
                evolution_info.add_tr_fitness(training_fitness)
                evolution_info.add_ts_fitness(test_fitness)
        runs_data.append(evolution_info)
    return runs_data


class EvolutionData:
    def __init__(self):
        self.test_fitness = None
        self.training_fitness = []
        
    def add_tr_fitness(self, training_fitness):
        self.training_fitness.append(training_fitness)
        
    def add_ts_fitness(self, test_fitness):
        self.test_fitness = test_fitness


def list_parser(str_list):
    '''
    Convert a string representing a list of floats in a list of floats. The
    string should be in the form '[n1,n2,...,nk]', where n1, n2, ...
    are elements of the list, preceded of followed by any number of white
    characters
    :param str_list: The string representing the list
    :return: A list of floats
    '''
    parsed_list = []
    # Remove brackets
    str_list = str_list.strip('[]\n\t\r')
    for el in str_list.split(','):
        parsed_list.append(float(el))
    return parsed_list


def get_metric_from_exp_data(exp_data, metric, *metric_args):
    '''
    Compute the value for a given metric given a list of experimental data
    :param exp_data: Experimental data read with *read_experiment*
    :param metric: The metric to be computed
    :param *metric_args: Optional parameters used by the metric 
    :return: A list with the values of the metric for each experiment
    '''
    metric_value = []
    for exp in exp_data:
        gen_value = []
        training_fitness = exp.training_fitness
        for pf in training_fitness:
            gen_value.append(metric(pf, *metric_args))
        metric_value.append(gen_value)
    return metric_value

def get_training_test_stats(exp_data):
    '''
    Get the training and test the value for a given metric given a list of experimental data
    :param exp_data: Experimental data read with *read_experiment* 
    :return: A list with the values of the metric for each experiment
    '''
    tr_error = []
    ts_error = []
    size = []
    for exp in exp_data:
        tr_fitness = exp.training_fitness[-1]
        ts_fitness = exp.test_fitness
        aux_tr_fitness = [x[0] for x in tr_fitness]
        best_index =  aux_tr_fitness.index(min(aux_tr_fitness))
        
        size.append([tr_fitness[best_index][1]])
        tr_error.append([tr_fitness[best_index][0]])
        ts_error.append([ts_fitness[best_index][0]])
    return size, tr_error, ts_error
    

def uniform_distribution(pareto_front):
    min_distance = [float("inf")] * len(pareto_front)
    for i in range(len(pareto_front)):
        for j in range(len(pareto_front)):
            if i != j:
                dist = euclidean_distance(pareto_front[i], pareto_front[j])
                if dist < min_distance[i]:
                    min_distance[i] = dist
    return numpy.std(numpy.array(min_distance))


def hyper_volume(pareto_front, reference_point):
    hv = HyperVolume(pareto_front, reference_point)
    return hv.compute(pareto_front)


def epsilon(solution_front, pareto_front):
    '''
    Compute the unary epsilon additive indicator as proposed in [Zitzler2003]_. 
    The implementation follows the jMetal implementation. 
    
    .. [Zitzler2003] Zitzler, E., Thiele, L., Laumanns, M., Fonseca, C. M., 
       Da Fonseca, V. G. (2003). Performance assessment of multiobjective 
       optimizers: An analysis and review. IEEE Transactions on evolutionary 
       computation, 7(2), 117-132.
    :param solution_front:
    :param solution_front:
    '''
    eps_j = 0.0
    eps_k = 0.0

    n_objectives = len(pareto_front[0])

    eps = float("-inf")

    for i in range(len(pareto_front)): 
        for j in range(len(solution_front)):
            for k in range(n_objectives):
                eps_temp = solution_front[j][k] - pareto_front[i][k]
                # We are maximizing eps_temp according to k
                if k == 0:
                    eps_k = eps_temp
                elif eps_k < eps_temp:
                    eps_k = eps_temp
            # Maximizing regarding j
            if j == 0:
                eps_j = eps_k
            elif eps_j > eps_k:
                eps_j = eps_k
        # Minimize regarding i
        if i == 0:
            eps = eps_j;
        elif eps < eps_j:
            eps = eps_j
    return eps
  
def write_stats_to_file(stats_value, output_path):
    f = open(output_path, 'w')
    for row in stats_value:
        row_size = len(row)
        for cell in range(row_size):
            if cell < row_size-1:
                f.write(str(row[cell]) + ',')
            else:
                f.write(str(row[cell]) + '\n')
    f.close()
    

def generate_stats():
    for problem in problems:
        
        dir_list = [folder.path for folder in scandir(input_path) 
                    if problem in os.path.basename(folder.path)]
        
        if 'zdt' in problem:
            fitness_dict = dict()
            # Load the experiments' data
            for folder in dir_list:
                exp_name = os.path.basename(folder)
                print("Reading experiment " + exp_name)
                fitness_dict[exp_name] = read_experiment(folder, 10)
            
            hv_files = []
            eps_files = []
            pareto_front = get_pareto_front(problem)
            n_objectives = len(pareto_front[0]) 
            # The variation (delta) is given by maximum - minimum 
            delta_pf = [max_points[problem][m] - min_points[problem][m] for m in range(n_objectives)]
            
            # Process the statistics from the data 
            for exp_name in fitness_dict.keys():
                for run in range(len(fitness_dict[exp_name])):
#                     for gen in range(len(fitness_dict[exp_name][run].training_fitness)):
#                         # Plot the Pareto front (original data)
#                         plot_pf(fitness_dict[exp_name][run].training_fitness[gen],
#                                 ['f1', 'f2'], pareto_front, 
#                                 path.join(output_path, "plot_pf_" + exp_name + 
#                                           "_" + str(run) + "_" + str(gen) + ".pdf"))
                    # Normalize the data
                    for pf in fitness_dict[exp_name][run].training_fitness:
                        for sol_id in range(len(pf)):
                            pf[sol_id] = [(pf[sol_id][m] - min_points[problem][m])/delta_pf[m] 
                                        for m in range(n_objectives)]
                # Compute the normalized metrics
                print("Computing the HV metric")
                hv = get_metric_from_exp_data(fitness_dict[exp_name], hyper_volume, [1,1])
                print("Computing the eps metric")
                eps = get_metric_from_exp_data(fitness_dict[exp_name], epsilon, pareto_front)
                print("Writing the results to the file")
                # Write HV stats
                stats_path = path.join(output_path, "hv_" + exp_name + ".csv")
                hv_files.append(stats_path)
                write_stats_to_file(hv, stats_path)
                # Write epsilon stats
                stats_path = path.join(output_path, "eps_" + exp_name + ".csv")
                eps_files.append(stats_path)
                write_stats_to_file(eps, stats_path)
                
            mean_std_plot(hv_files, 
                          path.join(output_path, "plot_hv_" + problem + ".pdf"))
            mean_std_plot(eps_files, 
                          path.join(output_path, "plot_eps_" + problem + ".pdf"))
            
        else:
            # Load the experiments' 
            for folder in dir_list:
                exp_name = os.path.basename(folder)
                print("Reading experiment " + exp_name)
                exp_fitness = read_experiment(folder, 10)
                for run in range(len(exp_fitness)):
                    for gen in range(len(exp_fitness[run].training_fitness)):
                        # Plot the Pareto front (original data)
                        plot_pf(exp_fitness[run].training_fitness[gen], 
                                ['Error', 'Size'], None, 
                                path.join(output_path, "plot_pf_" + exp_name + 
                                          "_" + str(run) + "_" + str(gen) + ".pdf"))
                size, tr_fitness, ts_fitness = get_training_test_stats(exp_fitness)
                stats_path = path.join(output_path, "size_" + exp_name + ".csv")
                write_stats_to_file(size, stats_path)
                stats_path = path.join(output_path, "trFit_" + exp_name + ".csv")
                write_stats_to_file(tr_fitness, stats_path)
                stats_path = path.join(output_path, "tsFit_" + exp_name + ".csv")
                write_stats_to_file(ts_fitness, stats_path)

     
def get_pareto_front(problem_name):
    points = []
    if 'zdt5' in problem_name:
        x_range = list(range(1, 32))
    else:
        x_range = arange(0, 1.001, 0.001)
    for x in x_range:
        points.append(pf_functions[problem_name](x))
    
    pareto_front = []
    for i in range(len(points)):
        non_dominated = True
        for j in range(len(points)):
            if i != j and dominates(points[j], points[i]):
                non_dominated = False
                break
        if non_dominated:
            pareto_front.append(points[i])
    return pareto_front


def dominates(p1, p2):
    not_equal = False
    for m in range(len(p1)):
        if p1[m] > p2[m]:
            return False
        elif p1[m] < p2[m]: 
            not_equal = True
    return not_equal
    

def compute_hv():
    """
    The main function for running the experiment manager. Calls all functions.

    :return: Nothing.
    """
    input_path = '/home/luiz/Dados/Trabalho/Pesquisa/Publicacoes/2017/MOGP/results/iqr/ponyge_output/zdt/zdt'
    output_paht = '/home/luiz/Dados/Trabalho/Pesquisa/Publicacoes/2017/MOGP/results/iqr/ponyge_output/stats'
    for i in range(1, 7):
        for f in scandir(input_path + str(i)):
            print("Reading the experiment")
            exp = read_experiment(path.join(input_path, f.path), 10)
            print("Computing the  metric")
            hv = get_metric_from_exp_data(exp, hyper_volume, ref_point[i-1])
            print("Writing the results to the file")
            write_stats_to_file(hv, path.join(output_paht, "hv_" + path.basename(f.path)))


def plot_pareto_fronts():
    # input_path = '/home/luiz/Dados/Trabalho/Pesquisa/Publicacoes/2017/MOGP/results/iqr/ponyge_output/zdt/zdt'
    input_path = '/home/luiz/Dados/Trabalho/Pesquisa/Publicacoes/2017/MOGP/results/iqr/ponyge_output/ge'
    output_path = '/home/luiz/Dados/Trabalho/Pesquisa/Publicacoes/2017/MOGP/results/iqr/ponyge_output/stats'
    for i in ['']: #range(1, 7):
        for run_folder in scandir(input_path + str(i)):
            exp = read_experiment(run_folder.path, -1)
            for run in range(len(exp)):
                plot_pf(exp[run].training_fitness[-1], None,
                        # zdt_functions['ZDT'+str(i)],
                        path.join(output_path, path.basename(run_folder.path) + "_" + str(run)))

if __name__ == "__main__":
    generate_stats()
    # compute_hv()
    # plot_pareto_fronts()
#     exp = read_experiment('/tmp/nsga-zdt1', 1)
#     for gen in range(len(exp[0].training_fitness)):
#         plot_pf(exp[0].training_fitness[gen], #None,
#                 zdt_functions['ZDT1'],
#                 # zdt_functions['ZDT'+str(i)],
#                 path.join('/tmp/plots', 'meps_zdt1' + "_" + str(gen+1) + '.png'))
