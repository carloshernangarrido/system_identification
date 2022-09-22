from typing import List
import numpy as np
from matplotlib import pyplot as plt


def plot_responses(t: np.ndarray, responses: List[np.ndarray]):
    fig, axs = plt.subplots(len(responses), 3, sharex='all')
    axs[0, 0].set_title('displacement (m)')
    axs[0, 1].set_title('velocity (m/s)')
    axs[0, 2].set_title('acceleration (m/s2)')
    for i, response in enumerate(responses):
        axs[i, 0].set_ylabel(f'DOF {i}')
        axs[i, 0].plot(t, response['x'])
        axs[i, 1].plot(t, response['x_dot'])
        axs[i, 2].plot(t, response['x_ddot'])
    [axs[-1, col].set_xlabel('time (s)') for col in range(3)]
    return fig, axs


def plot_residuals(force_sum: np.ndarray, inertia_term: np.ndarray, dofs_indices: list, t=None):
    t_length = len(force_sum) // len(dofs_indices)
    t_ = np.linspace(0, t_length, t_length) if t is None else t
    fig, axs = plt.subplots(len(dofs_indices), 1, sharex='all', sharey='all')
    for i in range(len(dofs_indices)):
        axs[i].plot(t_, force_sum[i * len(t_):(i + 1) * len(t_)], label='sum of forces')
        axs[i].plot(t_, inertia_term[i * len(t_):(i + 1) * len(t_)], label='inertial term')
        axs[i].set_ylabel(f'force (N) on DOF {dofs_indices[i]}')
        axs[i].plot(t_, (force_sum - inertia_term)[i * len(t_):(i + 1) * len(t_)], label='error')
    if t is None:
        axs[i].set_xlabel('time (samples)')
    else:
        axs[i].set_xlabel('time (s)')
    axs[i].legend()
    return fig, axs
