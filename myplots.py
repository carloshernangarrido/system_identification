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
    i = 0
    for i in range(len(dofs_indices)):
        axs[i].plot(t_, force_sum[i * len(t_):(i + 1) * len(t_)], label='sum of forces')
        axs[i].plot(t_, inertia_term[i * len(t_):(i + 1) * len(t_)], label='inertial term')
        axs[i].set_ylabel(f'force (N) on DOF {dofs_indices[i]}')
        error = (force_sum - inertia_term)[i * len(t_):(i + 1) * len(t_)]
        axs[i].plot(t_, error, label=f'error RMS={np.round(np.sqrt(np.mean(error ** 2)), 2)}')
        axs[i].legend()
    if t is None:
        axs[i].set_xlabel('time (samples)')
    else:
        axs[i].set_xlabel('time (s)')
    return fig, axs


def plot_fiting_chainlike(responses: list, parameters: dict, d_lim=0, v_lim=0, fe_lim=0):
    figs = []
    fig, axs = plt.subplots(1, len(responses)-1)
    axs[0].set_ylabel('elastic force')
    for i in range(len(responses)-1):
        axs[i].set_title(f'DOFs {i} to {i+1}')
        axs[i].set_xlabel('displacements')
        displacements = responses[i + 1]['x'] - responses[i]['x']
        try:
            elastic_force_linear = displacements * parameters['unknown'][f'k_{i}_{i+1}']
        except KeyError:
            elastic_force_linear = displacements * 0
        try:
            elastic_force_quadratic = displacements**2 * parameters['unknown'][f'k2_{i}_{i + 1}']
        except KeyError:
            elastic_force_quadratic = displacements * 0
        try:
            elastic_force_cubic = displacements**3 * parameters['unknown'][f'k3_{i}_{i + 1}']
        except KeyError:
            elastic_force_cubic = displacements * 0
        elastic_force_total = elastic_force_linear + elastic_force_quadratic + elastic_force_cubic
        axs[i].plot(displacements, elastic_force_total, label='total', linewidth=2)
        axs[i].plot(displacements, elastic_force_linear, label='linear', linewidth=0.5)
        axs[i].plot(displacements, elastic_force_quadratic, label='quadratic', linewidth=0.5)
        axs[i].plot(displacements, elastic_force_cubic, label='cubic', linewidth=0.5)
        axs[i].tick_params(axis='both', which='both')
        axs[i].grid('both')
        if d_lim > 0:
            axs[i].set_xlim((-d_lim, d_lim))
        if fe_lim > 0:
            axs[i].set_ylim((-fe_lim, fe_lim))
    axs[0].legend()
    plt.tight_layout()
    figs.append(fig)

    fig, axs = plt.subplots(1, len(responses) - 1)
    axs[0].set_ylabel('dissipative force')
    for i in range(len(responses) - 1):
        axs[i].set_title(f'DOFs {i} to {i + 1}')
        axs[i].set_xlabel('velocities')
        velocities = responses[i + 1]['x_dot'] - responses[i]['x_dot']
        try:
            dissipative_force_linear = velocities * parameters['unknown'][f'c_{i}_{i + 1}']
        except KeyError:
            dissipative_force_linear = velocities * 0
        try:
            dissipative_force_frictional = np.sign(velocities) * parameters['unknown'][f'muN_{i}_{i + 1}']
        except KeyError:
            dissipative_force_frictional = velocities * 0
        dissipative_force_total = dissipative_force_linear + dissipative_force_frictional
        axs[i].plot(velocities, dissipative_force_total, label='total', linewidth=2)
        axs[i].plot(velocities, dissipative_force_linear, label='linear', linewidth=0.5)
        axs[i].plot(velocities, dissipative_force_frictional, label='frictional', linewidth=0.5)
        axs[i].tick_params(axis='both', which='both')
        axs[i].grid('both')
        if v_lim > 0:
            axs[i].set_xlim((-v_lim, v_lim))
    axs[0].legend()
    plt.tight_layout()
    figs.append(fig)
    return figs
