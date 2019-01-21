""" Module: RANSAC Implementations
    Used for optimizing paramters to fit a distribtion using random sampling.
"""
import numpy as np
import random
from numpy import inner, max, diag, eye, Inf, dot
from numpy.linalg import norm, solve
import time

def stub1(test, data):
    pass

def stub2(data, params, epsilon):
    pass


class RansacModel:
    def __init__(self, data, create_model_callback, evaluate_model_callback, epsilon=2):
        self._createModelCallback = create_model_callback
        self._evaluateModelCallback = evaluate_model_callback
        self._data = data
        self._epsilon = epsilon

    def createHypothesis(self):
        self._params = self._createModelCallback(self._data)

    def evaluate(self):
        return self._evaluateModelCallback(self._data, self._params, self._epsilon)

    def getParams(self):
        return self._params

def ransac(model, kmax=100):
    '''
    Ransac implmentation
    :param model: of class RansacModel
    :param kmax: max number of iterations
    :return: maxinliers, best fit
    '''

    k = 0
    maxInliers = -1
    bestParams = None
    while k < kmax:
        k = k + 1

        # Find initial hypothesis
        model.createHypothesis()

        # Evaluate all data
        inliers = model.evaluate()

        # Store best fit
        if inliers > maxInliers:
            maxInliers = inliers
            bestParams = model.getParams()

    return maxInliers, bestParams


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


def createLineModel(data):
    '''
    Samples the data randomly and builds line model from a minimum subset
    :param data:
    :return:
    '''

    i1 = random.randint(0, len(data) - 1)
    i2 = random.randint(0, len(data) - 1)

    X1 = data[i1]
    X2 = data[i2]

    m = (X2[1] - X1[1])/(X2[0] - X1[0])
    b = m * X1[0] - X1[1]

    return [m, b]

def evaluateLineModel(data, params, epsilon):
    '''
    Evaluates all data with respect to a model hypothesis
    :param data: data to fit
    :param params: the model hypothesis
    :param epsilon: min error to consider as a inlier
    :return: number of inliers
    '''

    m, b = params
    inliers = 0
    for point in data:
        y = m * point[0] + b

        delta = abs(y - point[1])
        if delta < epsilon:
            inliers = inliers + 1

    return inliers


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

    points = np.array([x, y]).T
    line_ransac = RansacModel(points, createLineModel, evaluateLineModel, 50)

    return ransac(line_ransac)


if __name__ == '__main__':
    print("Test Cases")
    print(testRANSAC())