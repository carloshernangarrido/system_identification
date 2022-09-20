import os

import numpy as np

from lumped_mass_sysid import get_ab_mats
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
    plot_responses(t, responses)

    # Construct parameters as: a_mat x gamma_mat = b_mat, gamma_mat = pinv(a_mat) x b_mat
    # a_mat = [x1-x0 x1_dot-x0_dot x1-x2 x1_dot-x2_dot] (n_samples x 4)
    a_mat, b_mat = get_ab_mats(responses, parameters)

    gamma_mat = np.dot(np.linalg.pinv(a_mat), b_mat)
    parameters['unknown']['k_0_1'] = gamma_mat[0, 0]
    parameters['unknown']['c_0_1'] = gamma_mat[1, 0]
    parameters['unknown']['k_1_2'] = gamma_mat[2, 0]
    parameters['unknown']['c_1_2'] = gamma_mat[3, 0]

    print(parameters)
# {'known': {'m1': 26.665665, 'm2': 25.4226843},
# 'unknown': {'k01': 17798.46540887374, 'c01': 44.410658362079474, 'k12': 20081.57870784271, 'c12': 42.152059732163565}}
# {'known': {'m1': 26.665665, 'm2': 25.4226843},
# 'unknown': {'k01': 0.0, 'c01': 0.0, 'k12': 18055.119522237237, 'c12': 41.84862837339022}}
pass
