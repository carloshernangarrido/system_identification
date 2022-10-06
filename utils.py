import pickle
from typing import List
from itertools import combinations
import numpy as np


def check_element_type(element_type):
    supported_element_types = ['b', 'c', 'k', 'k2', 'k3', 'muN']
    assert element_type in supported_element_types, f'Unsupported element type: {element_type}'


class Parameters:
    def __init__(self, dof_masses: List[float],
                 fully_connected_k: bool = False, chain_like_k: bool = True,
                 fully_connected_c: bool = False, chain_like_c: bool = True,
                 fully_connected_b: bool = False, chain_like_b: bool = False,
                 fully_connected_k2: bool = False, chain_like_k2: bool = False,
                 fully_connected_k3: bool = False, chain_like_k3: bool = False,
                 fully_connected_muN: bool = False, chain_like_muN: bool = False):
        self.parameters = {'known': {}, 'unknown': {}}
        for i_dof, mass in enumerate(dof_masses):
            self.parameters['known'][f'm_{i_dof+1}'] = mass
        if fully_connected_k:
            self.fully_connected('k')
        elif chain_like_k:
            self.chain_like('k')
        if fully_connected_k2:
            self.fully_connected('k2')
        elif chain_like_k2:
            self.chain_like('k2')
        if fully_connected_k3:
            self.fully_connected('k3')
        elif chain_like_k3:
            self.chain_like('k3')
        if fully_connected_c:
            self.fully_connected('c')
        elif chain_like_c:
            self.chain_like('c')
        if fully_connected_muN:
            self.fully_connected('muN')
        elif chain_like_muN:
            self.chain_like('muN')
        if fully_connected_b:
            self.fully_connected('b')
        elif chain_like_b:
            self.chain_like('b')

    def fully_connected(self, element_type):
        check_element_type(element_type)
        for comb in combinations(range(1 + len(self.parameters['known'])), 2):
            self.parameters['unknown'][f'{element_type}_{comb[0]}_{comb[1]}'] = 0.0

    def chain_like(self, element_type):
        check_element_type(element_type)
        for i in range(len(self.parameters['known'])):
            self.parameters['unknown'][f'{element_type}_{i}_{i + 1}'] = 0.0


def t2delta_t(t):
    t_ = np.array(t)
    assert t_.ndim == 1, 't must be a 1D array-like'
    delta_ts = np.diff(t_)
    delta_t = np.round(delta_ts.mean(), 6)
    assert (np.max(delta_ts) - np.min(delta_ts)) / delta_t < 1E-6, \
        f'Problem with sample time. {np.max(delta_ts)=}, {np.min(delta_ts)=}, {np.mean(delta_ts)=}'
    return delta_t


def get_responses(responses_full_filenames, generate_referenceframe, remove_mean, i_ini=0, i_fin=None):
    responses = []
    t, delta_t = None, None
    for full_filename in responses_full_filenames:
        with open(full_filename, 'rb') as file:
            saving_list = pickle.load(file)
            txy_smoothed = saving_list[2]
        t = txy_smoothed[i_ini:i_fin, 0] if t is None else t
        delta_t = t2delta_t(t) if delta_t is None else delta_t
        x = txy_smoothed[i_ini:i_fin, 1] - txy_smoothed[i_ini:i_fin, 1].mean() if remove_mean else txy_smoothed[
                                                                                                   i_ini:i_fin, 1]
        x = x.copy()
        if generate_referenceframe:
            zeros = np.zeros(x[0:-2].size)
            responses.append({'x': zeros,
                              'x_dot': zeros,
                              'x_ddot': zeros})
            generate_referenceframe = False
        responses.append({'x': x[0:-2],
                          'x_dot': np.diff(x[0:-1]) / delta_t,
                          'x_ddot': np.diff(x, 2) / (delta_t ** 2)})
    return t[0:-2], responses


def check_responses_integrity(responses, t=None):
    def is_1d_ndarray(arr):
        return True if (isinstance(arr, np.ndarray) and len(arr.shape) == 1) else False

    if t is None:
        n_samples = responses[0]['x'].shape[0]
    else:
        assert is_1d_ndarray(t), 't must be 1D ndarray'
        n_samples = t.shape[0]

    for response in responses:
        assert isinstance(response, dict), 'each response must be a dict of 1D ndarray'
        has_x, has_x_dot, has_x_ddot = False, False, False
        for key in response.keys():
            if key == 'x':
                has_x = True
            elif key == 'x_dot':
                has_x_dot = True
            elif key == 'x_ddot':
                has_x_ddot = True
            else:
                raise AssertionError('response keys must be x, x_dot or x_ddot')
            assert is_1d_ndarray(response[key]), 'Each element of response must be 1D ndarray'
            assert response[key].shape[0] == n_samples, 'Each element of response must have the n_samples as t'
        assert has_x and has_x_dot and has_x_ddot, 'response must have x, x_dot and x_ddot'
    return n_samples


def get_mck_mats(parameters):
    m_1 = parameters['known']['m_1']
    m_2 = parameters['known']['m_2']
    m_mat = np.array([[m_1, 0],
                      [0, m_2]])
    c_0_1 = parameters['unknown']['c_0_1'] if 'c_0_1' in parameters['unknown'].keys() else 0
    c_0_2 = parameters['unknown']['c_0_2'] if 'c_0_2' in parameters['unknown'].keys() else 0
    c_1_2 = parameters['unknown']['c_1_2'] if 'c_1_2' in parameters['unknown'].keys() else 0
    c_mat = np.array([[c_0_1 + c_1_2, -c_1_2],
                      [-c_1_2, c_1_2 + c_0_2]])
    k_0_1 = parameters['unknown']['k_0_1'] if 'k_0_1' in parameters['unknown'].keys() else 0
    k_0_2 = parameters['unknown']['k_0_2'] if 'k_0_2' in parameters['unknown'].keys() else 0
    k_1_2 = parameters['unknown']['k_1_2'] if 'k_1_2' in parameters['unknown'].keys() else 0
    k_mat = np.array([[k_0_1 + k_1_2, -k_1_2],
                      [-k_1_2, k_1_2 + k_0_2]])
    return m_mat, c_mat, k_mat
