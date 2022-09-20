from typing import List, Union

import numpy as np


class MassDOF:
    def __init__(self, mass: float = 0, neighbors: Union[None, List[int]] = None):
        neighbors = [] if neighbors is None else neighbors
        self.mass = mass
        self.neighbors = neighbors


def get_ab_mats(responses: List[np.ndarray], parameters: dict):
    """
    Returns the A and B matrices for system identification using Least Mean Squares on force summation on a known
    mass of a lumped mass system:

    a_mat x gamma_mat = b_mat,

    where gamma_mat is a column vector containing the values of the unknown parameters.

    :param responses: List of dicts with np.ndarray. Dict keys must be 'x', 'x_dot' and 'x_ddot' for position,
    velocity and acceleration, respectively. The first element in list [0] is the mass-irrelevant reference frame.
    :param parameters: Dict of 'known' and 'unknown' parameters of the
    system to identify. parameters['known'] must be 'm1', 'm2', and so on. parameters['unknown'] must be 'k_i_j' for
    stiffness between dof i and dof j, and c_i_j for damping coefficient between dof i and dof j.
    :returns: a_mat, b_mat np.ndarray s containing matrices A (a diference of
    responses in each column) and B (a column with the independent term)
    """
    mass_dofs = []
    for i, response in enumerate(responses[1:], start=1):
        mass = parameters[f'm{i}']
        parameters_names = parameters.keys()

        mass_dofs.append(MassDOF(mass, ))

    return a_mat, b_mat


# a_mat = np.hstack((x[2] - x[1], x_dot[2] - x_dot[1]))
# b_mat = -parameters['known']['m2']*x_ddot[2]
# gamma_mat = np.dot(np.linalg.pinv(a_mat), b_mat)
# parameters['unknown']['k12'] = gamma_mat[0, 0]
# parameters['unknown']['c12'] = gamma_mat[1, 0]
# gamma_mat = [ k01
#               c01
#               k12
#               c12 ] (4 x 1)
# b_mat = m1 [ x1_ddot ]
x = [responses[i]['x'].reshape(-1, 1) for i in range(len(responses))]
x_dot = [responses[i]['x_dot'].reshape(-1, 1) for i in range(len(responses))]
x_ddot = [responses[i]['x_ddot'].reshape(-1, 1) for i in range(len(responses))]
a_mat = np.hstack((x[1] - x[0], x_dot[1] - x_dot[0], x[1] - x[2], x_dot[1] - x_dot[2]))
b_mat = -parameters['known']['m1'] * x_ddot[1]
