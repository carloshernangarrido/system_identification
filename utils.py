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
