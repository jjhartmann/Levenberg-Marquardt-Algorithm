""" Module: Levenberg-Marquardt Algorithm Implementations
    Used for optimizing paramters to fit a distribtion.
"""
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

def line_differentiation(params, args):
    """ Symbolic Differentiation for Line Equation
    Note: we are passing in the effor function for the model we are using, but
    we can substitute the error for the actual model function
        error(x + delta) - error(x) <==> f(x + delta) - f(x)
    :param params: values to be used in model
    :param args: input (x) and observations (y)
    :return: The jacobian for the error_function
    """

    m, b = params
    x, y = args

    # Jacobian
    J = np.empty(shape=(len(params),) + x.shape, dtype=np.float)
    J[0] = x  # d/dm = x
    J[1] = 1  # d/db = 1

    return J


def line_error(params, args):
    """
    Line Error, calculates the error for the line equations y = mx + b
    :param params: values to be used in model
    :param x: inputs
    :param y: observations
    :return: difference between observations and estimates
    """
    x, y = args
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

    # Compute error
    y_0 = error_function(params, args)

    # Jacobian
    J = np.empty(shape=(len(params),) + y_0.shape, dtype=np.float)

    for i, param in enumerate(params):
        params_star = params[:]
        delta = param * delta_factor

        if abs(delta) < min_delta:
            delta = min_delta

        # Update single param and calculate error with updated value
        params_star[i] += delta
        y_1 = error_function(params_star, args)

        # Update Jacobian with gradients
        diff = y_0 - y_1
        J[i] = diff / delta

    return J

def LM(seed_params, args,
       error_function, jacobian_function=numerical_differentiation,
       llambda=1e-3, lambda_multiplier=10, kmax=500, eps=1e-3, verbose=False):
    """ Levenberg-Marquardt Implementaiton
     See: (https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm)
    :param  seed_params: initial starting guess for the params we are trying to find
    :param  args: the inputs (x) and observations (y)
    :param  error_function: describes how error is calculated for the model
        function args (params, x, y)
    :param  jacobian_function: produces and returns the jacobian for model
        function args (params, args, error_function)
    :param  llambda: initial dampening factor
    :param  lambda_multiplier: scale used to increase/decrease lambda
    :param  kmax: max number of iterations
    :return:  rmserror, params
    """

    # Equality : (JtJ + lambda * I * diag(JtJ)) * delta = Jt * error
    # Solve for delta
    params = seed_params

    k = 0
    while k < kmax:
        k += 1

        # Retrieve jacobian of function gradients with respect to the params
        J = jacobian_function(params, args, error_function)
        JtJ = inner(J, J)

        # I * diag(JtJ)
        A = eye(len(params)) * diag(JtJ)

        # == Jt * error
        error = error_function(params, args)
        Jerror = inner(J, error)

        rmserror = norm(error)

        if verbose:
            print("{} RMS: {} Params: {}".format(k, rmserror, params))

        if rmserror < eps:
            reason = "Converged to min epsilon"
            return rmserror, params, reason

        reason = ""
        error_star = error[:]
        rmserror_star = rmserror + 1
        while rmserror_star >= rmserror:
            try:
                delta = solve(JtJ + llambda * A, Jerror)
            except np.linalg.LinAlgError:
                print("Error: Singular Matrix")
                return -1

            # Update params and calculate new error
            params_star = params[:] + delta[:]
            error_star = error_function(params_star, args)
            rmserror_star = norm(error_star)

            if rmserror_star < rmserror:
                params = params_star
                llambda /= lambda_multiplier
                break

            llambda *= lambda_multiplier

            # Return if lambda explodes or if change is small
            if llambda > 1e9:
                reason = "Lambda to large."
                return rmserror, params, reason

        reduction = abs(rmserror - rmserror_star)
        if reduction < 1e-18:
            reason = "Change in error too small"
            return rmserror, params, reason

    return rmserror, params, "Finished kmax iterations"

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


def testLM():
    #####################
    # Test Line Fitting
    #####################

    # Input
    x = np.linspace(-500, 500, 1001)

    # Parameters:   m       b
    line_params = [3.56, -25.36]

    # Observations
    y = line_with_noise(line_params, x, 0, 2)

    # Seed
    start_params = [0, 0]

    return LM(start_params, (x, y), line_error, numerical_differentiation)


if __name__ == '__main__':
    print("Test Cases")
    print(testLM())
