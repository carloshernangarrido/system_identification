import itertools
from typing import List, Union

import numpy as np


class Element:
    def __init__(self, element_type: str, i: int, j: int):
        self.element_type = element_type
        self.i = i
        self.j = j

    def __str__(self, ji: bool = False):
        return f'{self.element_type}_{self.j}_{self.i}' if ji else f'{self.element_type}_{self.i}_{self.j}'

    def aliases(self):
        return [self.__str__(), self.__str__(ji=True)]

    def is_same_as(self, compare_element):
        return True if compare_element.__str__() in self.aliases() else False


class DOF:
    def __init__(self, responses: List[np.ndarray], parameters: dict, index: int = 0):
        self.index = index
        self.mass = parameters['known'][f'm_{self.index}'] if self.index != 0 else 0
        self.neighbors = []
        self.unknowns = parameters['unknown'].keys()
        self.elements = []
        for unknown in self.unknowns:
            element_type, i, j = unknown.split('_')
            i, j = int(i), int(j)
            if i == self.index and j not in self.neighbors:
                self.neighbors.append(j)
            if j == self.index and i not in self.neighbors:
                self.neighbors.append(i)
            if i in self.neighbors and j == self.index:
                self.elements.append(Element(element_type, i=self.index, j=i))
            if j in self.neighbors and i == self.index:
                self.elements.append(Element(element_type, i=self.index, j=j))

    def __str__(self):
        return f'dof_{self.index} ({[_.__str__() for _ in self.elements]})'


def get_ab_mats(responses: List[np.ndarray], parameters: dict, dof_index: int, ret_dof: bool = False):
    """
    Returns the A and B matrices for system identification using Least Mean Squares on force summation on a known
    mass of a lumped mass system:

    a_mat x gamma_mat = b_mat,

    where gamma_mat is a column vector containing the values of the unknown parameters.

    :param ret_dof: Flag to return dof object
    :param dof_index: Index of dof to consider when assembling a_mat.
    :param responses: List of dicts with np.ndarray. Dict keys must be 'x', 'x_dot' and 'x_ddot' for position,
    velocity and acceleration, respectively. The first element in list [0] is the mass-irrelevant reference frame.
    :param parameters: Dict of 'known' and 'unknown' parameters of the
    system to identify. parameters['known'] must be 'm1', 'm2', and so on. parameters['unknown'] must be 'k_i_j' for
    stiffness between dof i and dof j, and c_i_j for damping coefficient between dof i and dof j.
    :returns: a_mat, b_mat np.ndarray s containing matrices A (a difference of
    responses in each column) and B (a column with the independent term). If ret_dof is True, a_mat, b_mat, dof are
    returned.
    """
    dof = DOF(responses, parameters, index=dof_index)
    b_mat = - dof.mass * responses[dof_index]['x_ddot'].reshape((-1, 1))
    a_columns = []
    for element in dof.elements:
        assert dof.index == element.i
        if element.element_type == 'k':
            a_columns.append(responses[dof.index]['x'] - responses[element.j]['x'])
        elif element.element_type == 'c':
            a_columns.append(responses[dof.index]['x_dot'] - responses[element.j]['x_dot'])
        elif element.element_type == 'b':
            a_columns.append(responses[dof.index]['x_ddot'] - responses[element.j]['x_ddot'])
        else:
            raise ValueError(f'Unsupported element type: {element.element_type} in {dof}')
        a_columns[-1] = a_columns[-1].reshape(-1, 1)

    a_mat = np.hstack(a_columns)
    return (a_mat, b_mat) if ret_dof is False else (a_mat, b_mat, dof)


def get_ab_mats_assembly(responses: List[np.ndarray], parameters: dict, dof_indexes: List[int]):
    ab_mats = []
    for dof_index in dof_indexes:
        ab_mats.append(get_ab_mats(responses, parameters, dof_index, ret_dof=True))

    a_mat_assembly, b_mat_assembly, dof_assembly = get_ab_mats(responses, parameters, 0, ret_dof=True)
    print(dof_assembly)
    for dof_index in dof_indexes[1:]:
        a_mat_next, b_mat_next, dof_next = get_ab_mats(responses, parameters, dof_index, ret_dof=True)
        b_mat_assembly = np.vstack((b_mat_assembly, b_mat_next))
        print(dof_next)
        for i_element_next, element_next in enumerate(dof_next.elements):
            for i_element, element in enumerate(dof_assembly.elements):
                if element_next.is_same_as(element):
                    print(element, 'is the same as', element_next)

    # # check for repeated parameters
    # pairs_ab_mat = []
    # for pair_ab_mat in itertools.combinations(ab_mats, 2):
    #     print('')
    #     print(pair_ab_mat[0][2].elements, pair_ab_mat[1][2].elements)
    #     pairs_ab_mat.append(pair_ab_mat)
    #     # for element in ab_mat[2].elements:
    #     #     print(element.is_same_as())
    # return pairs_ab_mat
