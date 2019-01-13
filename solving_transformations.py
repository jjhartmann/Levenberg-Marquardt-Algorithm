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


def transform(params, pointset, mu=0, sigma=10):
    """
    Transforma a point set into another corrdinate system
    :param params:
    :param pointset:
    :param mu:
    :param sigma:
    :return:
    """



def testTransformation():
    """ Find rotation and translation between corresponding point sets
    by find the homography transformatino between the two.
    Assumption: In camera frames"""

    # Camera A
    model_points = ((2 * np.random.rand(4, 1000)) - 1) * 500
    model_points[3] = 1  # Homographic coord

    # Transform points
        


    #                  fx   fy  cx    cy  k0 k1
    project_params = [700, 700, 350, 350, 0, 0]
    image_points = projective_transform(project_params, model_points)

    # Ground truth transformation parameters
    #           x    y   z
    R_params = [23, 10, 45]
    t_params = [25, -50]








if __name__ == '__main__':
    print("Test Cases")
    print(testTransformation())