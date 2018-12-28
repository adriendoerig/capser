# -*- coding: utf-8 -*-
"""
My capsnet: Main script
Execute the training, evaluation and prediction of the capsnet
@author: Lynn

Last update on 28.12.2018
-> insertion of eval_throttle_secs parameter
-> small change for save_params
-> new validation and testing procedures
"""

import logging
import numpy as np
import tensorflow as tf
import os
from tensorboard.plugins.beholder import Beholder
from tensorboard.plugins.beholder import BeholderHook

from my_parameters import parameters
from my_capser_model_fn import model_fn
from my_capser_input_fn import train_input_fn, eval_input_fn, predict_input_fn
from my_capser_functions import save_params

###################################################

# Disable annoying logs:
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
#tf.logging.set_verbosity(tf.logging.ERROR)

print('-------------------------------------------------------')
print('TF version:', tf.__version__)
print('Starting capsnet script...')
print('-------------------------------------------------------')

###########################
#      Preparations:      #
###########################
# For reproducibility:
tf.reset_default_graph()
np.random.seed(42)
tf.set_random_seed(42)


##################################
#    Training and evaluation:    #
##################################
# Output the loss in the terminal every few steps:
logging.getLogger().setLevel(logging.INFO)

# Beholder to check on weights during training in tensorboard:
beholder = Beholder(parameters.logdir)
beholder_hook = BeholderHook(parameters.logdir)

# Create the estimator:
capser = tf.estimator.Estimator(model_fn=model_fn, model_dir=parameters.logdir)
train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=parameters.n_steps, hooks=[beholder_hook])
eval_spec = tf.estimator.EvalSpec(lambda: eval_input_fn(parameters.val_data_path), steps=parameters.eval_steps, throttle_secs=parameters.eval_throttle_secs)

# Save parameters from parameter file for reproducability
save_params(parameters.logdir, parameters)

# Lets go!
tf.estimator.train_and_evaluate(capser, train_spec, eval_spec)


##################################
#     Testing / Predictions:     #
##################################
# Lets have less annoying logs:
logging.getLogger().setLevel(logging.CRITICAL)


# Testing with training stimuli:
for n_category in range(len(parameters.test_data_paths)):
    test_filename = parameters.test_data_paths[n_category]
    print('-------------------------------------------------------')
    print('Compute vernier offset for ' + test_filename)
    
    # Determine vernier_accuracy for each shape type
    capser_out = list(capser.predict(lambda: predict_input_fn(test_filename)))
    vernier_accuracy = [p['vernier_accuracy'] for p in capser_out]
    results = np.mean(vernier_accuracy)
    print('Result: ' + str(results) + '; test_samples used: ' + str(len(vernier_accuracy)))
    
    txt_file_name = parameters.logdir + '/testing_results_step_' + str(parameters.n_steps) + \
    '_noise_' + str(parameters.test_noise[0]) + '_' + str(parameters.test_noise[1]) + '.txt'
    if not os.path.exists(txt_file_name):
        with open(txt_file_name, 'w') as f:
            f.write(test_filename + ' : \t' + str(results) + '\n')
    else:
        with open(txt_file_name, 'a') as f:
            f.write(test_filename + ' : \t' + str(results) + '\n')


# Testing with crowding/uncrowding:
for n_category in range(len(parameters.test_data_paths)):
    category = parameters.test_data_paths[n_category]
    print('-------------------------------------------------------')
    print('Compute vernier offset for ' + category)
    
    # Determine vernier_accuracy for our vernier/crowding/uncrowding stimuli 
    # (associated indices: 0/1/2) & save everything in a txt-file
    n_idx = 3
    results = np.zeros(shape=(n_idx,))
    for stim_idx in range(n_idx):
        test_filename = category + '/' + str(stim_idx) + '.tfrecords'
        capser_out = list(capser.predict(lambda: predict_input_fn(test_filename)))
        vernier_accuracy = [p['vernier_accuracy'] for p in capser_out]
        results[stim_idx] = np.mean(vernier_accuracy)
        print('Finished calculations for stimulus type ' + str(stim_idx))
        print('Result: ' + str(results[stim_idx]) + '; test_samples used: ' + str(len(vernier_accuracy)))
    
    txt_file_name = parameters.logdir + '/uncrowding_results_step_' + str(parameters.n_steps) + \
    '_noise_' + str(parameters.test_noise[0]) + '_' + str(parameters.test_noise[1]) + '.txt'
    if not os.path.exists(txt_file_name):
        with open(txt_file_name, 'w') as f:
            f.write(category + ' : \t' + str(results) + '\n')
    else:
        with open(txt_file_name, 'a') as f:
            f.write(category + ' : \t' + str(results) + '\n')


print('... Finished capsnet script!')
print('-------------------------------------------------------')
