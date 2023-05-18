import os
import pprint as pp
import numpy as np
from lumped_mass_sysid import get_ab_mats_assembly
from myplots import plot_residuals, plot_responses, plot_fiting_chainlike
from utils import get_responses, get_mck_mats, Parameters
import matplotlib.pyplot as plt

case = '15'
path = r'C:\Users\joses\Mi unidad\TRABAJO\46_cm_inerter\TRABAJO\experimental\ensayos\Campañas\2 - free vibrations\object_tracking'
response_filenames = [f'case_{case}_dof1_m.dat',
                      f'case_{case}_dof2_m.dat']
use_smoothed = True
i_ini, i_fin = 0, None
max_disp = .015
dofs_indices = [1, 2]

# Plot parameters
d_lim = .003  # m
fe_lim = 45  # N

responses_full_filenames = [os.path.join(path, response_filename) for response_filename in response_filenames]
flags = {'remove_mean': True,
         'fully_connected_k': False,
         'chain_like_k': True,
         'fully_connected_k2': False,
         'chain_like_k2': False,
         'fully_connected_k3': False,
         'chain_like_k3': False,
         'fully_connected_c': True,
         'chain_like_c': False,
         'fully_connected_muN': False,
         'chain_like_muN': False,
         'fully_connected_b': False,
         'chain_like_b': False}

dof_masses = [32.37, 26.007]

parameters = Parameters(dof_masses=dof_masses,
                        fully_connected_k=flags['fully_connected_k'], chain_like_k=flags['chain_like_k'],
                        fully_connected_k2=flags['fully_connected_k2'], chain_like_k2=flags['chain_like_k2'],
                        fully_connected_k3=flags['fully_connected_k3'], chain_like_k3=flags['chain_like_k3'],
                        fully_connected_c=flags['fully_connected_c'], chain_like_c=flags['chain_like_c'],
                        fully_connected_muN=flags['fully_connected_muN'], chain_like_muN=flags['chain_like_muN'],
                        fully_connected_b=flags['fully_connected_b'], chain_like_b=flags['chain_like_b']).parameters

print(parameters)

if __name__ == '__main__':
    #  Read responses assumed as positions
    t, responses = get_responses(responses_full_filenames,
                                 generate_referenceframe=True, remove_mean=flags['remove_mean'],
                                 i_ini=i_ini, i_fin=i_fin, use_smoothed=use_smoothed, max_disp=max_disp)
    plot_responses(t, responses)
    for i, response in enumerate(responses):
        file = open(f"output_{i}.txt", "w")
        for item in response['x']:
            print(item, file=file, flush=True)
        file.close()
        file = open(f"output_time.txt", "w")
        for item in t:
            print(item, file=file, flush=True)
        file.close()

    # Estimate parameters:
    a_mat, b_mat, elements, gamma_mat, par_result = \
        get_ab_mats_assembly(responses, parameters, dofs_indices, solve_and_ret=True)
    pp.pprint(par_result)
    m_mat, c_mat, k_mat = get_mck_mats(par_result)
    print(f"{m_mat=}, \n {c_mat=}, \n {k_mat=}")
    print("*** Raleigh hypothesis ***")
    print(f" alpha damping = {c_mat / m_mat}")
    print(f" beta damping = {c_mat / k_mat}")

    # Plot residuals
    plot_residuals(force_sum=np.dot(a_mat, gamma_mat).reshape((-1,)), inertia_term=b_mat.reshape((-1,)),
                   dofs_indices=dofs_indices, t=t)
    figs = plot_fiting_chainlike(responses=responses, parameters=parameters, d_lim=d_lim, fe_lim=fe_lim)
    figs[0].savefig(f'case_{case}.pdf')
    plt.show()

