# -*- coding: utf-8 -*-
"""
Get standard deviations from performance sheets
@author: Lynn

Last update on 14.02.2019
-> first draft
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
#import os

from my_parameters import parameters

###################################################

print('-------------------------------------------------------')
print('TF version:', tf.__version__)
print('Starting script...')
print('Chosen training procedure:', parameters.train_procedure)
print('-------------------------------------------------------')

def plot_uncrowding_results(results, error, save=None):
    N = len(results)
    ind = np.arange(N)  # the x locations for the groups
    fig, ax = plt.subplots()
    n_shapes = 5
    ax.bar(ind[0:n_shapes], results[0:n_shapes], yerr=error[0:n_shapes], align='center', alpha=0.5, ecolor='black', capsize=4)
    ax.bar(ind[n_shapes:], results[n_shapes:], yerr=error[n_shapes:], align='center', alpha=0.5, ecolor='black', capsize=4)
    
    # In case, we only have one trained network:
#    ax.bar(ind[0:n_shapes], results[0:n_shapes], alpha=0.5)
#    ax.bar(ind[n_shapes:], results[n_shapes:], alpha=0.5)

    # add some text for labels, title and axes ticks, and save figure
    ax.set_ylabel('%correct [Uncrowding - Crowding]')
    ax.set_title("Vernier decoder performance")
    ax.set_xticks([])
    ax.set_xticklabels([])
#    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')


    if save is None:
        plt.show()
        plt.close()
    else:
        plt.savefig(save)
        plt.close()


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
            # Since the categories 4stars and 6stars involve numbers, we have to get rid of them
            numbers = numbers[numbers!=4]
            numbers = numbers[numbers!=6]
            # Sometimes, I wrote the number of training steps into the file. We have to get rid of them, too
            numbers = numbers[numbers<=1000]
            results[:, :, idx_execution] = np.reshape(numbers, [-1, 3])


# Saving:
performance_txt_file = parameters.logdir + '/performance_plots.txt'
diff_results = np.squeeze(results[:, 2, :] - results[:, 1, :])
mean_diff_results = np.squeeze(np.mean(diff_results, 1)) * 100
std_diff_results = np.squeeze(np.std(diff_results, 1)) * 100 / np.sqrt(n_iterations)

# In case, we only have one trained network:
#mean_diff_results = np.squeeze(diff_results) * 100
#std_diff_results = 0

#for n_category in range(n_categories):
#    category = parameters.test_crowding_data_paths[n_category]
#    if not os.path.exists(performance_txt_file):
#        with open(performance_txt_file, 'w') as f:
#            f.write(category + ' : \t' + str(mean_diff_results[n_category]) + '\n')
#    else:
#        with open(performance_txt_file, 'a') as f:
#            f.write(category + ' : \t' + str(mean_diff_results[n_category]) + '\n')
#
#for n_category in range(n_categories):
#    category = parameters.test_crowding_data_paths[n_category]
#    if not os.path.exists(performance_txt_file):
#        with open(performance_txt_file, 'w') as f:
#            f.write(category + ' : \t' + str(std_diff_results[n_category]) + '\n')
#    else:
#        with open(performance_txt_file, 'a') as f:
#            f.write(category + ' : \t' + str(std_diff_results[n_category]) + '\n')

performance_png_file = parameters.logdir + '/performance_plots.png'
plot_uncrowding_results(mean_diff_results, std_diff_results, performance_png_file)

print('... Finished script!')
print('-------------------------------------------------------')
