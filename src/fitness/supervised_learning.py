from math import isnan

import numpy as np
np.seterr(all="raise")

from algorithm.parameters import params
from utilities.fitness.get_data import get_data
from utilities.fitness.math_functions import *
from utilities.fitness.optimize_constants import optimize_constants


class supervised_learning:
    """
    Fitness function for supervised learning, ie regression and
    classification problems. Given a set of training or test data,
    returns the error between y (true labels) and yhat (estimated
    labels).

    We can pass in the error metric and the dataset via the params
    dictionary. Of error metrics, eg RMSE is suitable for regression,
    while F1-score, hinge-loss and others are suitable for
    classification.

    This is an abstract class which exists just to be subclassed:
    should not be instantiated.
    """

    maximise = False
    default_fitness = np.NaN

    def __init__(self):
        # Get training and test data
        self.training_in, self.training_exp, self.test_in, self.test_exp = \
            get_data(params['DATASET_TRAIN'], params['DATASET_TEST'])

        # Find number of variables.
        self.n_vars = np.shape(self.training_in)[0]

        # Regression/classification-style problems use training and test data.
        if params['DATASET_TEST']:
            self.training_test = True

    def __call__(self, ind, dist="training"):
        """
        Note that math functions used in the solutions are imported from either
        utilities.fitness.math_functions or called from numpy.

        :param ind: An individual to be evaluated.
        :param dist: An optional parameter for problems with training/test
        data. Specifies the distribution (i.e. training or test) upon which
        evaluation is to be performed.
        :return: The fitness of the evaluated individual.
        """

        phen = ind.phenotype

        if dist == "training":
            x = self.training_in
            y = self.training_exp
        elif dist == "test":
            x = self.test_in
            y = self.test_exp
        else:
            raise ValueError("Unknown dist: " + dist)

        try:
            if params['OPTIMIZE_CONSTANTS']:
                # if we are training, then optimize the constants by
                # gradient descent and save the resulting phenotype
                # string as ind.phenotype_with_c0123 (eg x[0] +
                # c[0] * x[1]**c[1]) and values for constants as
                # ind.opt_consts (eg (0.5, 0.7). Later, when testing,
                # use the saved string and constants to evaluate.
                if dist == "training":
                    fitness = optimize_constants(x, y, ind)
                else:
                    # this string has been created during training
                    phen = ind.phenotype_consec_consts
                    c = ind.opt_consts
                    # phen will refer to x (ie test_in), and possibly to c
                    yhat = eval(phen)
                    assert np.isrealobj(yhat)

                    # let's always call the error function with the
                    # true values first, the estimate second
                    fitness = params['ERROR_METRIC'](y, yhat)

            else:
                # phenotype won't refer to C
                yhat = eval(phen)
                assert np.isrealobj(yhat)

                # let's always call the error function with the true
                # values first, the estimate second
                fitness = params['ERROR_METRIC'](y, yhat)

        except (FloatingPointError, ZeroDivisionError, OverflowError,
                MemoryError):
            # FP err can happen through eg overflow (lots of pow/exp calls)
            # ZeroDiv can happen when using unprotected operators
            fitness = self.default_fitness
        except Exception as err:
            # other errors should not usually happen (unless we have
            # an unprotected operator) so user would prefer to see them
            print(err)
            raise

        # don't use "not fitness" here, because what if fitness = 0.0?!
        if isnan(fitness):
            fitness = self.default_fitness

        return fitness
