import os
import sys
from math import sin, pi, exp
from os import listdir, path

import math

from matplotlib import pyplot as pl
from matplotlib.ticker import MaxNLocator
import pandas as pd

from numpy.ma.core import arange

sys.path.append("../src")

colors = ["#0000A6", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
          "#FFDBE5", "#7A4900", "#FFFF00", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
          "#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
          "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100",
          "#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F",
          "#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
          "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
          "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",
          "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
          "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00",
          "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
          "#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
          "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C"]


def mean_std_plot(data_path: list, output_path=None):
    """
    Plot the evolution of a metric through the generations. 
    :param data_path: A list containing the input data paths for different 
                      methods/datasets to be compared. Each element in the 
                      list must have *r* x *g* values, where *r* is the 
                      number of experiments executed and *g* is the number
                      of generations.  
    :param output_path: 
    :return: 
    """
    pl.clf()
    pl.hold(1)
    color_index = 0
    legend_handles = []
    data_path = sorted(data_path)
    for method in data_path:
        data = pd.read_csv(method, sep=',', header=None)
        x = range(data.shape[1])
        mean = data.mean(0)
        sd = data.std(0)
        tmp, = pl.plot(x, mean, 'k', color=colors[color_index], label=os.path.basename(method))
        legend_handles.append(tmp)
        pl.fill_between(x, mean - sd, mean + sd,
                        alpha=0.5, lw=0,
                        facecolor=colors[color_index])
        color_index += 1
    pl.xlabel(r'Generations ($\times 10$)')
    pl.legend(handles=legend_handles, loc=0)
    if output_path is None:
        pl.show()
    else:
        pl.savefig(output_path)


def plot_pf(approximated_pf, axis_labels, pareto_front, output_path=None):
    pl.clf()

    pl.figure().gca().yaxis.set_major_locator(MaxNLocator(integer=True))

    pl.margins(0.05)
    x_plot = []
    y_plot = []
    for sol in approximated_pf:
        x_plot.append(sol[0])
        y_plot.append(sol[1])
    tmp, = pl.plot(x_plot, y_plot, '+', color=colors[0], label="Approximated PF")
    legend_handles = [tmp]
    if pareto_front is not None:
        pareto_front = sorted(pareto_front, key=lambda x: x[0])
        x_plot = []
        y_plot = []
        for point in pareto_front:
            x_plot.append(point[0])
            y_plot.append(point[1])
        tmp, = pl.plot(x_plot, y_plot, 'k', color=colors[0], label="Pareto Front")
        legend_handles.append(tmp)
        
    pl.xlabel(axis_labels[0])
    pl.ylabel(axis_labels[1])

    pl.legend(handles=legend_handles)
    if output_path is None:
        pl.show()
    else:
        pl.savefig(output_path)
    pl.close()


if __name__ == "__main__":
    for i in range(1, 7):
        input_path = "/home/luiz/Dados/Trabalho/Pesquisa/Publicacoes/" \
                     "2017/MOGP/results/iqr/ponyge_output/stats/zdt" + str(i)
        file_list = [path.join(input_path, file_path) for file_path in listdir(input_path)]
        mean_std_plot(file_list, '/home/luiz/Dados/Trabalho/Pesquisa/Publicacoes/'
                                 '2017/MOGP/results/iqr/ponyge_output/stats/zdt' + str(i) + '.pdf')
