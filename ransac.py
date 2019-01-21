
import numpy as np
from numpy import inner, max, diag, eye, Inf, dot
from numpy.linalg import norm, solve
import time


def line_with_noise(params, x, mu=0, sigma=5):
    """ Calculate Line
    :param params: parameters for line equation y = mx + b ([m, b])
    :param x: input values
    :return: a vector containing the output of the line equation with noise
    """

    m, b = params[0:2]

    noise = np.random.normal(mu, sigma, len(x))
    y = m * x + b + noise
    return y


def testRANSAC():
    #############################
    # TEST RANSAC
    #############################

    # Input
    x = np.linspace(-500, 500, 1001)

    # Parameters:   m       b
    line_params = [3.56, -25.36]

    # Observations
    y = line_with_noise(line_params, x, 0, 2)

    



if __name__ == '__main__':
    print("Test Cases")
    print(testRANSAC())