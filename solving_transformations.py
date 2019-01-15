""" Toy example: Solve for points under a transformation
"""

import numpy as np
import math
from numpy import inner, max, diag, eye, Inf, dot
from numpy.linalg import norm, solve
import time

import LMA

def construct_projective_matrix(params):
    """
    Construct projection matrix
    See: https://www.scratchapixel.com/lessons/3d-basic-rendering/perspective-and-orthographic-projection-matrix/building-basic-perspective-projection-matrix
    :param params: fov, near, far
    :return: projection matrix
    """
    fov, near, far = params
    P = np.zeros(shape=(4,4), dtype=np.float)
    S = 1 / math.tan((fov/2) * (math.pi/180))

    P[0][0] = S
    P[1][1] = S
    P[2][2] = - far/(far - near)
    P[2][3] = - (far * near)/(far - near)
    P[3][2] = -1

    return P

def projective_transform(params, pointset, mu=0, sigma=10):
    """
    Performa projective transformation
    :param params: fx, fy, cx, cy, k0, k1
    :param pointset:
    :param mu:
    :param sigma:
    :return:
    """
    fx, fy, cx, cy, k0, k1 = params
    f = np.array((fx, fy))
    c = np.array((cx, cy))
    # project into uv frame
    uv = pointset[0:2] / pointset[2]

    # Apply distortion params
    rSquared = np.linalg.norm(uv, axis=0)
    uv = uv * (1 + k0*rSquared + k1*rSquared*rSquared)
    uv = uv.transpose() @ eye(2, 2) * f.transpose()
    uv[:, 0] += c[0]
    uv[:, 1] += c[1]

    return uv.transpose()


def transform(params, pointset, noise=False, mu=0, sigma=10):
    """
    Transforma a point set into another corrdinate system
    :param params:
    :param pointset:
    :param mu:
    :param sigma:
    :return:
    """

    thetas = np.array(params[0:3]) * math.pi/180
    t = params[3:6]

    Rx = np.eye(3, 3)
    Rx[1, 1] = np.cos(thetas[0])
    Rx[1, 2] = -np.sin(thetas[0])
    Rx[2, 1] = np.sin(thetas[0])
    Rx[2, 2] = np.cos(thetas[0])

    Ry = np.eye(3, 3)
    Ry[0, 0] = np.cos(thetas[1])
    Ry[2, 0] = -np.sin(thetas[1])
    Ry[0, 2] = np.sin(thetas[1])
    Ry[2, 2] = np.cos(thetas[1])

    Rz = np.eye(3, 3)
    Rz[0, 0] = np.cos(thetas[2])
    Rz[0, 1] = -np.sin(thetas[2])
    Rz[1, 0] = np.sin(thetas[2])
    Rz[1, 1] = np.cos(thetas[2])

    R = Rz @ Ry @ Rx

    transformed_points = np.empty(shape=pointset.shape, dtype=np.float)
    if noise:
        point_s = pointset + np.random.normal(mu, sigma, pointset.shape)
        transformed_points = point_s.transpose() @ R.transpose()
    else:
        transformed_points = pointset.transpose() @ R.transpose()

    transformed_points[:, 0] += t[0]
    transformed_points[:, 1] += t[1]
    transformed_points[:, 2] += t[2]

    return transformed_points.transpose(), R, t

def projective_error_function(params, args):
    """

    :param params:
    :param args:
    :return:
    """

    #                  fx   fy  cx    cy  k0 k1
    project_params = params[0:5]
    f, cx, cy, k0, k1 = project_params

    K = eye(3, 3)
    K[0,0] = f
    K[1,1] = f
    K[0, 2] = k0
    K[1, 2] = k1


    model, image = args
    tp = params[5:]
    _, R, t = transform(tp, model)
    Rt = np.c_[R, t.transpose()]

    # Reconstruct camera matrix
    P = K @ Rt

    # Project
    X = np.zeros((4, len(model[0])))
    X[0:3] = model
    X[3] = 1

    PX = P @ X
    image_star = PX[0:2] / PX[2]

    dataShape = image.shape
    nData = dataShape[0] * dataShape[1]
    imagevec = image.reshape(1, nData)[0]
    image_star_vec = image_star.reshape(1, nData)[0]

    return imagevec - image_star_vec


def projective_error_function_2(params, args):
    """

    :param params:
    :param args:
    :return:
    """

    model, X = args
    X_star, _, _ = transform(params, model)

    dataShape = X.shape
    nData = dataShape[0] * dataShape[1]
    X_vec = X.reshape(1, nData)[0]
    X_star_vec = X_star.reshape(1, nData)[0]

    return X_vec - X_star_vec


def testTransformation():
    """ Find rotation and translation between corresponding point sets
    by find the homography transformatino between the two.
    Assumption: In camera frames"""

    # Camera A
    model_points = ((2 * np.random.rand(3, 1000)) - 1) * 100
    # model_points = np.random.normal(0, 100, (3, 1000))

    # Ground truth transformation parameters
    #           x    y   z
    R_params = [23, -12, 4]
    t_params = [44, -102, 12]
    transform_parms =  R_params + t_params
    transfomed_points, R, t = transform(transform_parms, model_points, False, 0, 5)

    # Seed Params
    seed = np.zeros(6) # transform_parms + (np.random.normal(0, 11, 6))

    # Run LMA
    out = LMA.LM(seed, (model_points, transfomed_points),
                 projective_error_function_2,
                 lambda_multiplier=10,  kmax=10000, eps=0.001)
    print(out)

def IsPlanar(points, kmax=100, threshold=1e-3):
    """
    Determine if points lay on a plane using principle component analysis
    https://en.wikipedia.org/wiki/Principal_component_analysis

    :param points:
    :param kmax: max iterations
    :return: planar(bool), eigenratio (float)
    """


    # reshape points so row is axis components
    X = points.T

    # find mean
    mu = np.zeros(len(X[0]))
    for x in X:
        mu = mu + x
    mu = mu / len(X)

    # build covariance matrix
    A = np.zeros((3,3))
    for x in X:
        xc = x - mu
        A = A + np.outer(xc.T, xc)

    # compute eigenvalues and eigenvectors (column vectors)
    W, v = np.linalg.eig(A)
    W = np.sort(W)  # ascending order

    # ratio between the smallest and second smallest value
    eigenRatio = W[0] / (W[1] + 0.00001)
    planar = eigenRatio < threshold

    return planar, eigenRatio


def testTransformation2():
    """ Find rotation and translation between corresponding point sets
    by find the homography transformatino between the two.
    Assumption: In camera frames"""

    # Camera A
    model_points = ((2 * np.random.rand(3, 20)) - 1) * 500
    # model_points = np.random.normal(0, 100, (3, 100))

    planar, ratio = IsPlanar(model_points)


    # Ground truth transformation parameters
    #           x    y   z
    R_params = [23, -12, 4]
    t_params = [44, -102, 12]
    transform_parms =  R_params + t_params
    transfomed_points, R, t = transform(transform_parms, model_points, False, 0, 5)

    #                  fx   fy  cx    cy  k0 k1
    project_params = [100, 100, 50, 50, 0, 0]
    image_points = projective_transform(project_params, transfomed_points)

    i = 0
    while i < 40:
        i = i + 1
        # Seed Params
        # seed = np.zeros(12) # transform_parms + (np.random.normal(0, 11, 6))
        # seed = seed + [0,0,0,0,0,0,10, 10, 5, 5, 0, 0]

        noiseval = i * 0.5
        transformNoise = transform_parms + np.random.normal(0, noiseval, 6)
        projectNoise = project_params[1:] + np.random.normal(0, noiseval, 5)
        seed = np.concatenate([projectNoise, transformNoise])
        # seed = seed + (np.random.normal(0, 0.1, 11))


        # Run LMA
        out = LMA.LM(seed, (model_points, image_points),
                     projective_error_function,
                     lambda_multiplier=2,  kmax=1000, eps=1)


        print("\n\nRMS ERR: \t{}".format(out[0]))
        print("Eigen Ratio: \t{}".format(ratio))
        print("Noise Val: {}".format(noiseval))
        print("Projection: \t{}".format(out[1][0:5]))
        print("Angle: \t{}".format(out[1][5:8]))
        print("Translation: \t{}".format(out[1][8:11]))
        print("Reason: \t{}".format(out[2]))



    print(model_points)


def testTransformation3():
    """ Find rotation and translation between corresponding point sets
    by find the homography transformatino between the two.
    Assumption: In camera frames"""



    i = 0
    while i < 40:
        i = i + 1

        N = 20
        model_points = np.ones((3, N))

        variance = 1/40 * i
        var = np.array([0.1, 0.1, variance])
        C = eye(3,3)  #np.outer(var, var)
        # C = eye(3, 3)
        C[0,1] = variance
        C[1,0] = variance
        C[2,1] = variance
        C[1,2] = variance
        C[0,2] = variance/10
        C[2,0] = variance/10

        model_points = np.random.multivariate_normal([0,0,0], C, 15) * 200
        model_points = model_points.T

        planar, ratio = IsPlanar(model_points)

        # flat = False
        # while not flat:
        #     # Camera A
        #     model_points = ((2 * np.random.rand(3, 20)) - 1) * 500
        #     # model_points = np.random.normal(0, 20, (3, 100))
        #
        #     planar, ratio = IsPlanar(model_points)
        #
        #     flat = ratio > 0.8

        # Ground truth transformation parameters
        #           x    y   z
        R_params = [23, -12, 4]
        t_params = [44, -102, 12]
        transform_parms = R_params + t_params
        transfomed_points, R, t = transform(transform_parms, model_points, False, 0, 5)

        #                  fx   fy  cx    cy  k0 k1
        project_params = [100, 100, 50, 50, 0, 0]
        image_points = projective_transform(project_params, transfomed_points)

        # Seed Params
        # seed = np.zeros(12) # transform_parms + (np.random.normal(0, 11, 6))
        # seed = seed + [0,0,0,0,0,0,10, 10, 5, 5, 0, 0]

        noiseval = 5  #i * 1
        transformNoise = transform_parms + np.random.normal(0, noiseval, 6)
        projectNoise = project_params[1:] + np.random.normal(0, noiseval, 5)
        seed = np.concatenate([projectNoise, transformNoise])
        # seed = seed + (np.random.normal(0, 0.1, 11))


        # Run LMA
        out = LMA.LM(seed, (model_points, image_points),
                     projective_error_function,
                     lambda_multiplier=2,  kmax=2000, eps=0.1)


        print("\n\nRMS ERR: \t{}".format(out[0]))
        print("Eigen Ratio: \t{}, {}".format(ratio, variance))
        print("Noise Val: {}".format(noiseval))
        print("Projection: \t{}".format(out[1][0:5]))
        print("Angle: \t{}".format(out[1][5:8]))
        print("Translation: \t{}".format(out[1][8:11]))
        print("Reason: \t{}".format(out[2]))



    print(model_points)



if __name__ == '__main__':
    print("Test Cases")
    print(testTransformation3())
    # print(testTransformation2())
    # print(testTransformation())