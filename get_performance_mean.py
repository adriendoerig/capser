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
n_idx = 2
results = np.zeros(shape=(n_categories, n_idx, n_iterations))

idx_routing = 2

for idx_execution in range(n_iterations):
    log_dir = parameters.logdir + str(idx_execution) + '/'
    log_dir_results = log_dir + 'iter_routing_' + str(idx_routing) + '/'

    for n_category in range(n_categories):
        # Getting data:
        txt_file_name = log_dir_results + '/uncrowding_results_step_' + str(parameters.n_steps*n_rounds) + \
        '_noise_' + str(parameters.test_noise[0]) + '_' + str(parameters.test_noise[1]) + '.txt'

        with open(txt_file_name, 'r') as f:
            lines = f.read()
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", lines)
            numbers = np.float32(numbers)
            # Since the categories 4stars and 6stars involve numbers, we have to get rid of them
            numbers = numbers[numbers!=4]
            numbers = numbers[numbers!=6]
            results[:, :, idx_execution] = np.reshape(numbers, [-1, n_idx])


# Saving:
final_result_file = parameters.logdir + '/final_results_iterations_' + str(n_iterations) + '.txt'
final_results = np.mean(results, 2)
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
