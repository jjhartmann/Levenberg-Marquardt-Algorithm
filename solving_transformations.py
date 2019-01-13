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
    uv = uv.transpose() * f.transpose()
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

    thetas = params[0:3]
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

    model, image = args
    transfomed_points, _, _ = transform(params, model)

    #                  fx   fy  cx    cy  k0 k1
    project_params = [700, 700, 350, 350, 0, 0]
    image_star = projective_transform(project_params, transfomed_points)

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
    R_params = [23 * math.pi/180, -12 * math.pi/180, 4 *math.pi/180]
    t_params = [44, -102, 12]
    transform_parms =  R_params + t_params
    transfomed_points, R, t = transform(transform_parms, model_points, False, 0, 5)

    #                  fx   fy  cx    cy  k0 k1
    project_params = [700, 700, 350, 350, 0, 0]
    image_points = projective_transform(project_params, transfomed_points)


    # Seed Params
    seed = np.zeros(6) # transform_parms + (np.random.normal(0, 11, 6))

    # Run LMA
    out = LMA.LM(seed, (model_points, transfomed_points),
                 projective_error_function_2,
                 lambda_multiplier=10,  kmax=10000, eps=1)
    print(out)







if __name__ == '__main__':
    print("Test Cases")
    print(testTransformation())