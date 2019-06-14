
import re
import numpy as np
import os
from my_parameters import parameters


print('Starting performance script...')

###########################
#      Preparations:      #
###########################
n_idx = 3
#n_iterations = parameters.n_iterations
n_iterations = 7
n_rounds = parameters.n_rounds
n_categories = len(parameters.test_crowding_data_paths)
routing_min = parameters.routing_min
routing_max = parameters.routing_max


# Getting final performance means:
for idx_routing in range(routing_min, routing_max+1):
    results = np.zeros(shape=(n_categories, n_idx, n_iterations))

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


    # Saving final means:
    final_result_file = parameters.logdir + '/final_results_mean_iterations_' + str(n_iterations) + '_iter_routing_' + str(idx_routing) + '.txt'
    final_results = np.mean(results, 2)
    for n_category in range(n_categories):
        category = parameters.test_crowding_data_paths[n_category]
        if not os.path.exists(final_result_file):
            with open(final_result_file, 'w') as f:
                f.write(category + ' : \t' + str(final_results[n_category, :]) + '\n')
        else:
            with open(final_result_file, 'a') as f:
                f.write(category + ' : \t' + str(final_results[n_category, :]) + '\n')


# Getting final performance stds:
for idx_routing in range(routing_min, routing_max+1):
    results = np.zeros(shape=(n_categories, n_idx, n_iterations))

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


    # Saving final means:
    final_result_file = parameters.logdir + '/final_results_std_iterations_' + str(n_iterations) + '_iter_routing_' + str(idx_routing) + '.txt'
    final_results = np.std(results, 2)
    for n_category in range(n_categories):
        category = parameters.test_crowding_data_paths[n_category]
        if not os.path.exists(final_result_file):
            with open(final_result_file, 'w') as f:
                f.write(category + ' : \t' + str(final_results[n_category, :]) + '\n')
        else:
            with open(final_result_file, 'a') as f:
                f.write(category + ' : \t' + str(final_results[n_category, :]) + '\n')

print('... Finished performance script!')
print('-------------------------------------------------------')