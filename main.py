import os

import numpy as np

from lumped_mass_sysid import get_ab_mats, get_ab_mats_assembly
from utils import get_responses, plot_responses, get_mck_mats
import matplotlib.pyplot as plt

path = r''
response_filenames = ['txy_dof1_m.dat',
                      'txy_dof2_m.dat']
responses_full_filenames = [os.path.join(path, response_filename) for response_filename in response_filenames]

steel_dens = 7800
floor_vol = 3099962 / (1000 ** 3)
columns_vol = 318713 / (1000 ** 3)

flag_fully_conected = False
if flag_fully_conected:
    parameters = {'known': {'m_1': steel_dens * (floor_vol + columns_vol),
                            'm_2': steel_dens * (floor_vol + (0.5 * columns_vol))},
                  'unknown': {'k_0_1': 0.0,
                              'c_0_1': 0.0,
                              'k_1_2': 0.0,
                              'c_1_2': 0.0,
                              'c_0_2': 0.0,
                              'k_0_2': 0.0}}
else:
    parameters = {'known': {'m_1': steel_dens * (floor_vol + columns_vol),
                            'm_2': steel_dens * (floor_vol + (0.5 * columns_vol))},
                  'unknown': {'k_0_1': 0.0,
                              'c_0_1': 0.0,
                              'k_1_2': 0.0,
                              'c_1_2': 0.0,
                              'c_0_2': 0.0, }}

flags = {'remove_mean': True,
         }

if __name__ == '__main__':
    #  Read responses assumed as positions
    i_ini, i_fin = 10, None
    t, responses = get_responses(responses_full_filenames,
                                 generate_referenceframe=True, remove_mean=flags['remove_mean'],
                                 i_ini=100, i_fin=None)
    plot_responses(t, responses)

    # Estimate parameters:
    dofs_indices = [1, 2]
    a_mat, b_mat, elements, gamma_mat, par_result = \
        get_ab_mats_assembly(responses, parameters, dofs_indices, solve_and_ret=True)
    print([_.__str__() for _ in elements], '\n', par_result)
    m_mat, c_mat, k_mat = get_mck_mats(par_result)
    print(f"{m_mat=}, \n {c_mat=}, \n {k_mat=}")
    print("*** Raleigh hypothesis ***")
    print(f" alpha damping = {c_mat / m_mat}")
    print(f" beta damping = {c_mat / k_mat}")

    # Residuals
    force_sum = np.dot(a_mat, gamma_mat).reshape((-1,))
    inertia_term = b_mat.reshape((-1,))
    fig, axs = plt.subplots(len(dofs_indices), 1, sharex='all')
    for i in range(len(dofs_indices)):
        axs[i].plot(t, force_sum[i * len(t):(i + 1) * len(t)], label='sum of forces')
        axs[i].plot(t, inertia_term[i * len(t):(i + 1) * len(t)], label='inertial term')
        axs[i].set_ylabel(f'force (N) on DOF {dofs_indices[i]}')
        axs[i].plot(t, (force_sum - inertia_term)[i * len(t):(i + 1) * len(t)], label='error')
        axs[i].set_xlabel('time (s)')
    plt.show()

    # Simulation
...
