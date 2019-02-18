# -*- coding: utf-8 -*-
"""
Get standard deviations from performance sheets
@author: Lynn

Last update on 14.02.2019
-> first draft
"""

import re
import numpy as np
import tensorflow as tf
import os

from my_parameters import parameters

###################################################

print('-------------------------------------------------------')
print('TF version:', tf.__version__)
print('Starting script...')
print('Chosen training procedure:', parameters.train_procedure)
print('-------------------------------------------------------')


n_iterations = parameters.n_iterations
n_rounds = parameters.n_rounds
n_categories = len(parameters.test_crowding_data_paths)
n_idx = 3
results = np.zeros(shape=(n_categories, n_idx, n_iterations))


for idx_execution in range(n_iterations):
    log_dir = parameters.logdir + str(idx_execution) + '/'

    for n_category in range(n_categories):
        # Getting data:
        txt_file_name = log_dir + '/uncrowding_results_step_' + str(parameters.n_steps*n_rounds) + \
        '_noise_' + str(parameters.test_noise[0]) + '_' + str(parameters.test_noise[1]) + '.txt'

        with open(txt_file_name, 'r') as f:
            lines = f.read()
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", lines)
            numbers = np.float32(numbers)
            results[:, :, idx_execution] = np.reshape(numbers, [-1, 3])


# Saving:
final_result_file = parameters.logdir + '/final_results_std_iterations_' + str(n_iterations) + '.txt'
final_results = np.round(np.std(results, 2), 3)
for n_category in range(n_categories):
    category = parameters.test_crowding_data_paths[n_category]
    if not os.path.exists(final_result_file):
        with open(final_result_file, 'w') as f:
            f.write(category + ' : \t' + str(final_results[n_category, :]) + '\n')
    else:
        with open(final_result_file, 'a') as f:
            f.write(category + ' : \t' + str(final_results[n_category, :]) + '\n')

print('... Finished script!')
print('-------------------------------------------------------')
