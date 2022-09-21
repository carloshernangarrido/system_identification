import pickle
from typing import List

import numpy as np
from matplotlib import pyplot as plt


def t2delta_t(t):
    t_ = np.array(t)
    assert t_.ndim == 1, 't must be a 1D array-like'
    delta_ts = np.diff(t_)
    delta_t = np.round(delta_ts.mean(), 6)
    assert (np.max(delta_ts) - np.min(delta_ts)) / delta_t < 1E-6, \
        f'Problem with sample time. {np.max(delta_ts)=}, {np.min(delta_ts)=}, {np.mean(delta_ts)=}'
    return delta_t


def get_responses(responses_full_filenames, generate_referenceframe, remove_mean):
    responses = []
    t, delta_t = None, None
    for full_filename in responses_full_filenames:
        with open(full_filename, 'rb') as file:
            saving_list = pickle.load(file)
            txy_smoothed = saving_list[2]
        t = txy_smoothed[:, 0] if t is None else t
        delta_t = t2delta_t(t) if delta_t is None else delta_t
        x = txy_smoothed[:, 1] - txy_smoothed[:, 1].mean() if remove_mean else txy_smoothed[:, 1]
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


def plot_responses(t: np.ndarray, responses: List[np.ndarray]):
    fig, axs = plt.subplots(len(responses), 3, sharex='all')
    for i, response in enumerate(responses):
        axs[i, 0].set_ylabel(f'DOF {i}')
        axs[i, 0].plot(t, response['x'])
        axs[i, 0].set_title('displacement (m)')
        axs[i, 1].plot(t, response['x_dot'])
        axs[i, 1].set_title('velocity (m/s)')
        axs[i, 2].plot(t, response['x_ddot'])
        axs[i, 2].set_title('acceleration (m/s2)')
    plt.show()
    return fig, axs


def get_mck_mats(parameters):
    m_1 = parameters['known']['m_1']
    m_2 = parameters['known']['m_2']
    m_mat = np.array([[m_1, 0],
                     [0, m_2]])
    c_0_1 = parameters['unknown']['c_0_1']
    c_0_2 = parameters['unknown']['c_0_2'] if 'c_0_2' in parameters['unknown'].keys() else 0
    c_1_2 = parameters['unknown']['c_1_2']
    c_mat = np.array([[c_0_1+c_1_2, -c_1_2],
                      [-c_1_2, c_1_2+c_0_2]])
    k_0_1 = parameters['unknown']['k_0_1']
    k_0_2 = parameters['unknown']['k_0_2'] if 'k_0_2' in parameters['unknown'].keys() else 0
    k_1_2 = parameters['unknown']['k_1_2']
    k_mat = np.array([[k_0_1+k_1_2, -k_1_2],
                      [-k_1_2, k_1_2+k_0_2]])
    return m_mat, c_mat, k_mat
