""" Module: Vanilla Levenberg-Marquardt Algorithm
    Used for optimizing paramters to fit a distribtion.
"""
import numpy as np
from numpy import inner, max, diag, eye, Inf, dot
from numpy.linalg import norm, solve
import time


def LM(seed_params, args,
       error_function, jacobian_function):
    """ Levenberg-Marquardt Implementaiton
    @return rmserror
    @param seed_params: initial starting guess for the params we are trying to find
    @param args: the inputs (x) and observations (y)
    @param error_function: describes how error is calculated for the model
    @param jacobian_function: produces and returns the jacobian for model"""


def line_with_noise(params, x, mu=0, sigma=5):
    """ Calculate Line
    @return: a vector containing the output of the line equation with noise
    @param params: parameters for line equation y = mx + b ([m, b])
    @param x: input values"""

    m, b = params[0:2]

    noise = np.random.normal(0, 5, len(x))
    y = m * x + b + noise
    return y


def testLM():
    #####################
    # Test Line Fitting
    #####################

    # Input
    x = np.linspace(-500, 500, 1001)

    # Parameters:   m       b
    line_params = [3.56, -2.34]

    # Observations
    y = line_with_noise(line_params, x)

    # Seed
    start_params = [0, 0]




if __name__ == '__main__':
    print("Test Cases")
    print(testLM())
