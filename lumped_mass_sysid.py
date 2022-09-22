import itertools
from typing import List, Union
import numpy as np
from utils import check_responses_integrity


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


def get_ab_mats(responses: List[np.ndarray], parameters: dict, dof_index: int, solve_and_ret: bool = False):
    """
    Returns the A and B matrices for system identification using Least Mean Squares on force summation on a known
    mass of a lumped mass system:

    a_mat x gamma_mat = b_mat,

    where gamma_mat is a column vector containing the values of the unknown parameters.

    :param solve_and_ret: Flag to ask to solve the System of Equations and return updated parameters
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
    if solve_and_ret:
        gamma_mat = np.dot(np.linalg.pinv(a_mat), b_mat)
        par_result = parameters.copy()
        for i_element, element in enumerate(dof.elements):
            if element.__str__() in par_result['unknown'].keys():
                par_result['unknown'][element.__str__()] = gamma_mat[i_element, 0]
            else:
                par_result['unknown'][element.__str__(ji=True)] = gamma_mat[i_element, 0]
        return a_mat, b_mat, dof, par_result
    else:
        return a_mat, b_mat, dof


def get_ab_mats_assembly(responses: List[np.ndarray], parameters: dict, dof_indexes: List[int],
                         solve_and_ret: bool = False):
    n_samples = check_responses_integrity(responses)
    a_mat_assembly, b_mat_assembly, _ = get_ab_mats(responses, parameters, dof_indexes[0])
    elements_assembly = _.elements
    for i_dof, dof_index in enumerate(dof_indexes[1:], start=1):
        a_mat_next, b_mat_next, dof_next = get_ab_mats(responses, parameters, dof_index)
        b_mat_assembly = np.vstack((b_mat_assembly, b_mat_next))
        a_mat_assembly = np.vstack((a_mat_assembly, np.zeros((n_samples, a_mat_assembly.shape[1]))))
        for i_element_next, element_next in enumerate(dof_next.elements):
            is_present = False
            for i_element, element in enumerate(elements_assembly):
                if element_next.is_same_as(element):
                    a_mat_assembly[-n_samples:, i_element] = a_mat_next[:, i_element_next]
                    is_present = True
                    break
            if not is_present:
                a_mat_assembly = np.hstack((a_mat_assembly, np.zeros((n_samples*(1+i_dof), 1))))
                a_mat_assembly[-n_samples:, -1] = a_mat_next[:, i_element_next]
                elements_assembly.append(element_next)
    if solve_and_ret:
        gamma_mat_assembly = np.dot(np.linalg.pinv(a_mat_assembly), b_mat_assembly)
        par_result_assembly = parameters.copy()
        for i_element, element in enumerate(elements_assembly):
            if element.__str__() in par_result_assembly['unknown'].keys():
                par_result_assembly['unknown'][element.__str__()] = gamma_mat_assembly[i_element, 0]
            else:
                par_result_assembly['unknown'][element.__str__(ji=True)] = gamma_mat_assembly[i_element, 0]
        return a_mat_assembly, b_mat_assembly, elements_assembly, gamma_mat_assembly, par_result_assembly
    else:
        return a_mat_assembly, b_mat_assembly, elements_assembly
