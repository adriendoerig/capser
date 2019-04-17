# -*- coding: utf-8 -*-
"""
My capsnet: Main script
Execute the training, evaluation and prediction of the capsnet
@author: Lynn

Last update on 14.02.2019
-> insertion of eval_throttle_secs parameter
-> small change for save_params
-> new validation and testing procedures
-> implemented n_rounds to decide how often we evaluate the test sets
-> reconstructions for prediction mode
-> update and check to guarantee the similarity to the main script
-> added final stds
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
from my_capser_functions import save_params, plot_uncrowding_results

###################################################

# Disable annoying logs:
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
#tf.logging.set_verbosity(tf.logging.ERROR)

print('-------------------------------------------------------')
print('TF version:', tf.__version__)
print('Starting capsnet script...')
print('Chosen training procedure:', parameters.train_procedure)
print('-------------------------------------------------------')

###########################
#      Preparations:      #
###########################
# For reproducibility:
tf.reset_default_graph()
np.random.seed(42)
tf.set_random_seed(42)

n_iterations = parameters.n_iterations
n_categories = len(parameters.test_crowding_data_paths)
n_idx = 3
results = np.zeros(shape=(n_categories, n_idx, n_iterations))


for idx_execution in range(n_iterations):
    log_dir = parameters.logdir + str(idx_execution) + '/'
    
    ##################################
    #    Training and evaluation:    #
    ##################################
    # Output the loss in the terminal every few steps:
    logging.getLogger().setLevel(logging.INFO)
    
    # Beholder to check on weights during training in tensorboard:
    beholder = Beholder(log_dir)
    beholder_hook = BeholderHook(log_dir)
    
    # Create the estimator:
    my_checkpointing_config = tf.estimator.RunConfig(keep_checkpoint_max = 2)  # Retain the 2 most recent checkpoints.
    
    capser = tf.estimator.Estimator(model_fn=model_fn, model_dir=log_dir, config=my_checkpointing_config, params={'log_dir': log_dir})
    eval_spec = tf.estimator.EvalSpec(lambda: eval_input_fn(parameters.val_data_path), steps=parameters.eval_steps, throttle_secs=parameters.eval_throttle_secs)
    
    # Save parameters from parameter file for reproducability
    save_params(log_dir, parameters)
    
    # Lets go!
    n_rounds = parameters.n_rounds
    
    
    for idx_round in range(1, n_rounds+1):
        train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=parameters.n_steps*idx_round)
        tf.estimator.train_and_evaluate(capser, train_spec, eval_spec)
    
        ##################################
        #     Testing / Predictions:     #
        ##################################
        # Lets have less annoying logs:
        logging.getLogger().setLevel(logging.CRITICAL)        
        
        # Testing with crowding/uncrowding:
        cats = []
        res = []
        for n_category in range(n_categories):
            category = parameters.test_crowding_data_paths[n_category]
            cats.append(category[21:])
            print('-------------------------------------------------------')
            print('Compute vernier offset for ' + category)
            
            # Determine vernier_accuracy for our vernier/crowding/uncrowding stimuli
            results0 = np.zeros(shape=(n_idx,))
            for stim_idx in range(n_idx):
                test_filename = category + '/' + str(stim_idx) + '.tfrecords'
                
                # Lets also get some reconstructions for prediction mode using the following path:
                capser = tf.estimator.Estimator(model_fn=model_fn, model_dir=log_dir, params={'log_dir': log_dir, 'get_reconstructions': False})
                capser_out = list(capser.predict(lambda: predict_input_fn(test_filename)))
                vernier_accuracy = [p['vernier_accuracy'] for p in capser_out]
                rank_pred_shapes = [p['rank_pred_shapes'] for p in capser_out]
                rank_pred_proba = [p['rank_pred_proba'] for p in capser_out]
                
                # Get the the results for averaging over several trained networks only from the final round:
                if idx_round==n_rounds:
                    results[n_category, stim_idx, idx_execution] = np.mean(vernier_accuracy)
                
                # Get all the other results per round:
                results0[stim_idx] = np.mean(vernier_accuracy)
                results1 = np.unique(rank_pred_shapes)
                results2 = np.mean(rank_pred_proba, 0)
                res.append(np.mean(vernier_accuracy))

                print('Finished calculations for stimulus type ' + str(stim_idx))
                print('Result: ' + str(results0[stim_idx]) + '; test_samples used: ' + str(len(vernier_accuracy)))
                
                # Saving:
                txt_ranking_file_name = log_dir + '/ranking_step_' + str(parameters.n_steps*idx_round) + '.txt'
                if not os.path.exists(txt_ranking_file_name):
                    with open(txt_ranking_file_name, 'w') as f:
                        f.write(category + str(stim_idx) + ' : \t' + str(results1) + ' : \t' + str(results2) + '\n')
                else:
                    with open(txt_ranking_file_name, 'a') as f:
                        f.write(category + str(stim_idx) + ' : \t' + str(results1) + ' : \t' + str(results2) + '\n')
            
            # Saving:
            txt_file_name = log_dir + '/uncrowding_results_step_' + str(parameters.n_steps*idx_round) + \
            '_noise_' + str(parameters.test_noise[0]) + '_' + str(parameters.test_noise[1]) + '.txt'
            if not os.path.exists(txt_file_name):
                with open(txt_file_name, 'w') as f:
                    f.write(category + ' : \t' + str(results0) + '\n')
            else:
                with open(txt_file_name, 'a') as f:
                    f.write(category + ' : \t' + str(results0) + '\n')
        
        # Plotting:
        plot_uncrowding_results(res, cats, save=log_dir + '/uncrowding_results_step_' + str(parameters.n_steps*idx_round) +
                                '_noise_' + str(parameters.test_noise[0]) + '_' + str(parameters.test_noise[1]) + '.png')


# Saving final means:
final_result_file = parameters.logdir + '/final_results_iterations_' + str(n_iterations) + '.txt'
final_results_mean = np.mean(results, 2)
for n_category in range(n_categories):
    category = parameters.test_crowding_data_paths[n_category]
    if not os.path.exists(final_result_file):
        with open(final_result_file, 'w') as f:
            f.write(category + ' : \t' + str(final_results_mean[n_category, :]) + '\n')
    else:
        with open(final_result_file, 'a') as f:
            f.write(category + ' : \t' + str(final_results_mean[n_category, :]) + '\n')

# Saving final stds:
final_result_file = parameters.logdir + '/final_results_std_iterations_' + str(n_iterations) + '.txt'
final_results_std = np.std(results, 2)
for n_category in range(n_categories):
    category = parameters.test_crowding_data_paths[n_category]
    if not os.path.exists(final_result_file):
        with open(final_result_file, 'w') as f:
            f.write(category + ' : \t' + str(final_results_std[n_category, :]) + '\n')
    else:
        with open(final_result_file, 'a') as f:
            f.write(category + ' : \t' + str(final_results_std[n_category, :]) + '\n')

print('... Finished capsnet script!')
print('-------------------------------------------------------')


# Get reconstructions:
#os.system('python my_capser_get_reconstructions.py')