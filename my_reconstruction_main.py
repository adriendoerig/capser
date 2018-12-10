# -*- coding: utf-8 -*-
"""
My capsnet: let's decode the reconstructions seperately
Main script
@author: Lynn

Last update on 10.12.18
-> first draft of the reconstruction code
-> restore necessary parameters via params in Estimator API
-> unfortunately, the code is not as flexible as I would like it to be
-> decide whether to train whole model or only the reconstruction decoder
"""


import logging
import numpy as np
import tensorflow as tf
import os
from tensorboard.plugins.beholder import Beholder
from tensorboard.plugins.beholder import BeholderHook

from my_parameters import parameters
from my_reconstruction_model_fn import model_fn
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


#########################################
#    Restore all relevant parameters:   #
#########################################
imported_meta = tf.train.import_meta_graph(parameters.logdir + parameters.restoration_file)

with tf.Session() as sess:
    # accessing the restored default graph and all operations:
    imported_meta.restore(sess, tf.train.latest_checkpoint(parameters.logdir))
    graph = tf.get_default_graph()
    
    # get kernels and biases for all conv layers:
    conv1_kernel_restored = graph.get_tensor_by_name('conv1/kernel:0').eval()
    conv1_bias_restored = graph.get_tensor_by_name('conv1/bias:0').eval()
    
    conv2_kernel_restored = graph.get_tensor_by_name('conv2/kernel:0').eval()
    conv2_bias_restored = graph.get_tensor_by_name('conv2/bias:0').eval()
    
    conv3_kernel_restored = graph.get_tensor_by_name('conv3/kernel:0').eval()
    conv3_bias_restored = graph.get_tensor_by_name('conv3/bias:0').eval()
    
    # restore weights between first and second caps layer
    W_restored = graph.get_tensor_by_name('3_secondary_caps_layer/W:0').eval()


params = {'conv1_kernel_restored': conv1_kernel_restored,
          'conv1_bias_restored': conv1_bias_restored,
          'conv2_kernel_restored': conv2_kernel_restored,
          'conv2_bias_restored': conv2_bias_restored,
          'conv3_kernel_restored': conv3_kernel_restored,
          'conv3_bias_restored': conv3_bias_restored,
          'W_restored': W_restored}

##################################
#    Training and evaluation:    #
##################################
# Output the loss in the terminal every few steps:
logging.getLogger().setLevel(logging.INFO)

# Beholder to check on weights during training in tensorboard:
beholder = Beholder(parameters.logdir_reconstruction)
beholder_hook = BeholderHook(parameters.logdir_reconstruction)

# Create the estimator:
capser = tf.estimator.Estimator(model_fn=model_fn, params=params, model_dir=parameters.logdir_reconstruction)
train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=parameters.n_steps, hooks=[beholder_hook])
eval_spec = tf.estimator.EvalSpec(eval_input_fn, steps=parameters.eval_steps, throttle_secs=parameters.eval_throttle_secs)

# Save parameters from parameter file for reproducability
save_params(parameters.logdir_reconstruction, parameters)

# Lets go!
tf.estimator.train_and_evaluate(capser, train_spec, eval_spec)


##################################
#     Testing / Predictions:     #
##################################
# Lets have less annoying logs:
logging.getLogger().setLevel(logging.CRITICAL)

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
    
    txt_file_name = parameters.logdir_reconstruction + '/uncrowding_exp_results_step_' + str(parameters.n_steps) + \
    '_noise_' + str(parameters.test_noise) + '_shape_size_' + str(parameters.shape_size) + '.txt'
    if not os.path.exists(txt_file_name):
        with open(txt_file_name, 'w') as f:
            f.write(category + ' : \t' + str(results) + '\n')
    else:
        with open(txt_file_name, 'a') as f:
            f.write(category + ' : \t' + str(results) + '\n')


print('... Finished capsnet script!')
print('-------------------------------------------------------')
