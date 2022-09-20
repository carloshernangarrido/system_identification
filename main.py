import os

import numpy as np

from lumped_mass_sysid import get_ab_mats, get_ab_mats_assembly
from utils import get_responses, plot_responses
import matplotlib.pyplot as plt


path = r'C:\Users\joses\Mi unidad\TRABAJO\46_cm_inerter\TRABAJO\experimental\ensayos\object_tracking'
response_filenames = ['txy_dof1_m.dat',
                      'txy_dof2_m.dat']
responses_full_filenames = [os.path.join(path, response_filename) for response_filename in response_filenames]

steel_dens = 7800
floor_vol = 3099962/(1000**3)
columns_vol = 318713/(1000**3)

parameters = {'known': {'m_1': steel_dens*(floor_vol+columns_vol),
                        'm_2': steel_dens*(floor_vol+(0.5*columns_vol))},
              'unknown': {'k_0_1': 0.0,
                          'c_0_1': 0.0,
                          'k_1_2': 0.0,
                          'c_1_2': 0.0}}

flags = {'remove_mean': True,
         }

if __name__ == '__main__':
    #  Read responses assumed as positions
    t, responses = get_responses(responses_full_filenames,
                                 generate_referenceframe=True, remove_mean=flags['remove_mean'])
    # plot_responses(t, responses)

    # Estimate parameters as: a_mat x gamma_mat = b_mat, gamma_mat = pinv(a_mat) x b_mat
    a_mat, b_mat, dof = get_ab_mats(responses, parameters, 2, ret_dof=True)
    gamma_mat = np.dot(np.linalg.pinv(a_mat), b_mat)
    print(gamma_mat, dof)

    ab_mats = get_ab_mats_assembly(responses, parameters, [0, 1, 2])
pass
