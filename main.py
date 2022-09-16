import os

import numpy as np

from utils import get_responses, plot_responses
import matplotlib.pyplot as plt


path = r'C:\Users\joses\Mi unidad\TRABAJO\46_cm_inerter\TRABAJO\experimental\ensayos\object_tracking'
response_filenames = ['txy_dof1_ok.dat',
                      'txy_dof2_ok.dat']
responses_full_filenames = [os.path.join(path, response_filename) for response_filename in response_filenames]

steel_dens = 7800
floor_vol = 3099962/(1000**3)
columns_vol = 318713/(1000**3)

parameters = {'known': {'m1': steel_dens*(floor_vol+columns_vol),
                        'm2': steel_dens*(floor_vol+(0.5*columns_vol))},
              'unknown': {'k01': 0.0,
                          'c01': 0.0,
                          'k12': 0.0,
                          'c12': 0.0}}

flags = {'remove_mean': True,
         }

if __name__ == '__main__':
    #  Read responses assumed as positions
    t, responses = get_responses(responses_full_filenames,
                                 generate_referenceframe=True, remove_mean=flags['remove_mean'])
    plot_responses(t, responses)

    # Construct parameters as: a_mat x gamma_mat = b_mat, gamma_mat = pinv(a_mat) x b_mat
    # a_mat = [x1-x0 x1_dot-x0_dot x1-x2 x1_dot-x2_dot] (n_samples x 4)
    # gamma_mat = [ k01
    #               c01
    #               k12
    #               c12 ] (4 x 1)
    # b_mat = m1 [ x1_ddot ]
    x = [responses[i]['x'].reshape(-1, 1) for i in range(len(responses))]
    x_dot = [responses[i]['x_dot'].reshape(-1, 1) for i in range(len(responses))]
    x_ddot = [responses[i]['x_ddot'].reshape(-1, 1) for i in range(len(responses))]
    # a_mat = np.hstack((x[1] - x[0], x_dot[1] - x_dot[0], x[1] - x[2], x_dot[1] - x_dot[2]))
    # b_mat = -parameters['known']['m1']*x_ddot[1]
    # gamma_mat = np.dot(np.linalg.pinv(a_mat), b_mat)
    # parameters['unknown']['k01'] = gamma_mat[0, 0]
    # parameters['unknown']['c01'] = gamma_mat[1, 0]
    # parameters['unknown']['k12'] = gamma_mat[2, 0]
    # parameters['unknown']['c12'] = gamma_mat[3, 0]
    a_mat = np.hstack((x[2] - x[1], x_dot[2] - x_dot[1]))
    b_mat = -parameters['known']['m2']*x_ddot[2]
    gamma_mat = np.dot(np.linalg.pinv(a_mat), b_mat)
    parameters['unknown']['k12'] = gamma_mat[0, 0]
    parameters['unknown']['c12'] = gamma_mat[1, 0]
    print(parameters)
# {'known': {'m1': 26.665665, 'm2': 25.4226843},
# 'unknown': {'k01': 17798.46540887374, 'c01': 44.410658362079474, 'k12': 20081.57870784271, 'c12': 42.152059732163565}}
# {'known': {'m1': 26.665665, 'm2': 25.4226843},
# 'unknown': {'k01': 0.0, 'c01': 0.0, 'k12': 18055.119522237237, 'c12': 41.84862837339022}}
pass
