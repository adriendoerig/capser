# -*- coding: utf-8 -*-
"""
My capsnet: Get reconstructions
@author: Lynn

Last update on 01.02.2019
-> first draft
"""

import os
import logging
import tensorflow as tf
import numpy as np

from my_parameters import parameters
from my_batchmaker import stim_maker_fn
from my_capser_model_fn import model_fn


###########################
#      Preparations:      #
###########################
chosen_shape = 1
stim_idx = 0
row = 3
column = 5
batch_size = row*column

n_rounds = parameters.n_rounds
n_iterations = parameters.n_iterations
n_categories = parameters.test_shape_types
n_idx = 3


def predict_input_fn(chosen_shape, stim_idx, batch_size, parameters):
    im_size = parameters.im_size
    shape_size = parameters.shape_size
    bar_with = parameters.bar_width
    n_shapes = parameters.n_shapes
    centralize = parameters.centralized_shapes
    reduce_df = parameters.reduce_df
    
    stim_maker = stim_maker_fn(im_size, shape_size, bar_with)
    [shape_1_images, shape_2_images, shapelabels, vernierlabels, nshapeslabels, nshapeslabels_idx, 
     x_shape_1, y_shape_1, x_shape_2, y_shape_2] = stim_maker.makeTestBatch(chosen_shape, n_shapes, batch_size, stim_idx, centralize, reduce_df)

    feed_dict = {'shape_1_images': shape_1_images,
                 'shape_2_images': shape_2_images,
                 'shapelabels': shapelabels,
                 'nshapeslabels': nshapeslabels,
                 'vernier_offsets': vernierlabels,
                 'x_shape_1': x_shape_1,
                 'y_shape_1': y_shape_1,
                 'x_shape_2': x_shape_2,
                 'y_shape_2': y_shape_2,
                 'mask_with_labels': False,
                 'is_training': False}
    
    return tf.estimator.inputs.numpy_input_fn(feed_dict, batch_size=batch_size)


###########################
#       Main script:      #
###########################
for idx_execution in range(n_iterations):
    log_dir = parameters.logdir + str(idx_execution) + '/'
    
    ##################################
    #    Training and evaluation:    #
    ##################################
    # Output the loss in the terminal every few steps:
    logging.getLogger().setLevel(logging.INFO)
    
    ##################################
    #     Testing / Predictions:     #
    ##################################
    # Lets have less annoying logs:
    logging.getLogger().setLevel(logging.CRITICAL)        
    
    # Testing with crowding/uncrowding:
    for n_category in range(n_categories):
        category = parameters.test_shape_types[n_category]
        print('-------------------------------------------------------')
        print('Reconstruct for ' + category)
        
        for stim_idx in range(n_idx):
            test_filename = category + '/' + str(stim_idx) + '.tfrecords'
            
            # Lets also get some reconstructions for prediction mode using the following path:
            capser = tf.estimator.Estimator(model_fn=model_fn, model_dir=log_dir,
                                            params={'log_dir': log_dir,
                                                    'idx_round': n_rounds,
                                                    'save_path': '/uncrowding/' + category[21:] + str(stim_idx) + '_step' + str(parameters.n_steps*n_rounds) +
                                                    '_noise_' + str(parameters.test_noise[0]) + '_' + str(parameters.test_noise[1])})
            capser_out = list(capser.predict(lambda: predict_input_fn(category, stim_idx, batch_size, parameters)))
            vernier_accuracy = [p['vernier_accuracy'] for p in capser_out]
            
            # Saving:
            txt_ranking_file_name = log_dir + '/ranking_step_' + str(parameters.n_steps*n_rounds) + '.txt'
            if not os.path.exists(txt_ranking_file_name):
                with open(txt_ranking_file_name, 'w') as f:
                    f.write(category + str(stim_idx) + ' : \t' + str(results1) + ' : \t' + str(results2) + '\n')
            else:
                with open(txt_ranking_file_name, 'a') as f:
                    f.write(category + str(stim_idx) + ' : \t' + str(results1) + ' : \t' + str(results2) + '\n')


print('... Finished capsnet script!')
print('-------------------------------------------------------')
