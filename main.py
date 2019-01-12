""" Module: Vanilla Levenberg-Marquardt Algorithm
    Used for optimizing paramters to fit a distribtion.
"""
import numpy as np
from numpy import inner, max, diag, eye, Inf, dot
from numpy.linalg import norm, solve
import time


def line_error(params, x, y):
    """
    Line Error, calculates the error for the line equations y = mx + b
    :param params: values to be used in model
    :param x: inputs
    :param y: observations
    :return: difference between observations and estimates
    """

    m, b = params[0:2]
    y_star = m * x + b

    return y - y_star

def numerical_differentiation(params, args, error_function):
    """ Numerical Differentiation
    Note: we are passing in the effor function for the model we are using, but
    we can substitute the error for the actual model function
        error(x + delta) - error(x) <==> f(x + delta) - f(x)
    :param params: values to be used in model
    :param args: input (x) and observations (y)
    :param error_function: function used to determine error based on params and observations
    :return: The jacobian for the error_function
    """
    delta_factor = 1e-4
    min_delta = 1e-4

    x, y = params
    y_0 = error_function(params, x, y)

    # Jacobian
    J = np.empty(shape=(len(params),) + x.shape, dtype=np.float)

    for i, param in enumerate(params):
        params_star = params
        delta = param * delta_factor

        if abs(delta) < min_delta:
            delta = min_delta

        # Update single param and calculate error with updated value
        params_star[i] += delta
        y_1 = error_function(params_star, x, y)

        # Update Jacobian with gradients
        diff = y_1 - y_0
        J[i] = diff / delta

    return J

def LM(seed_params, args,
       error_function, jacobian_function,
       llambda=1e-3, lambda_multiplier=10):
    """ Levenberg-Marquardt Implementaiton
    :param  seed_params: initial starting guess for the params we are trying to find
    :param  args: the inputs (x) and observations (y)
    :param  error_function: describes how error is calculated for the model
    :param  jacobian_function: produces and returns the jacobian for model
    :param  llambda: initial dampening factor
    :param  lambda_multiplier: scale used to increase/decrease lambda
    :return:  rmserror
    """



def line_with_noise(params, x, mu=0, sigma=5):
    """ Calculate Line
    :param params: parameters for line equation y = mx + b ([m, b])
    :param x: input values
    :return: a vector containing the output of the line equation with noise
    """

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
    start_params = [1, 1]




if __name__ == '__main__':
    print("Test Cases")
    print(testLM())
