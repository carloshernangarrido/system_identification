import os
import numpy as np
from lumped_mass_sysid import get_ab_mats_assembly
from plots import plot_residuals, plot_responses
from utils import get_responses, get_mck_mats, Parameters
import matplotlib.pyplot as plt

path = r''
response_filenames = ['txy_dof1_m.dat',
                      'txy_dof2_m.dat']
responses_full_filenames = [os.path.join(path, response_filename) for response_filename in response_filenames]
flags = {'remove_mean': True,
         'fully_connected_k': False,
         'chain_like_k': True,
         'fully_connected_c': True,
         'chain_like_c': False,
         'fully_connected_b': False,
         'chain_like_b': False}

steel_dens = 7800
floor_vol = 3099962 / (1000 ** 3)
columns_vol = 318713 / (1000 ** 3)
dof_masses = [steel_dens * (floor_vol + columns_vol), steel_dens * (floor_vol + (0.5 * columns_vol))]

parameters = Parameters(dof_masses=dof_masses,
                        fully_connected_k=flags['fully_connected_k'], chain_like_k=flags['chain_like_k'],
                        fully_connected_c=flags['fully_connected_c'], chain_like_c=flags['chain_like_c'],
                        fully_connected_b=flags['fully_connected_b'], chain_like_b=flags['chain_like_b']).parameters

print(parameters)

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

    # Plot residuals
    plot_residuals(force_sum=np.dot(a_mat, gamma_mat).reshape((-1,)), inertia_term=b_mat.reshape((-1,)),
                   dofs_indices=[1, 2], t=t)
    plt.show()

    # Simulation
    ...  # TODO
