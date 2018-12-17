# -*- coding: utf-8 -*-
"""
My capsnet: let's decode the reconstructions seperately
Main script
@author: Lynn

Last update on 14.12.18
-> first draft of the reconstruction code
-> restore necessary parameters via params in Estimator API
-> unfortunately, the code is not as flexible as I would like it to be
-> decide whether to train whole model or only the reconstruction decoder
-> deletion of save_params for reconstruction
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
#imported_meta = tf.train.import_meta_graph(parameters.logdir + parameters.restoration_file + '.meta')
#
#with tf.Session() as sess:
#    # accessing the restored default graph and all operations:
#    imported_meta.restore(sess, tf.train.latest_checkpoint(parameters.logdir))
#    graph = tf.get_default_graph()
#    
#    trainables = tf.trainable_variables()
#    
#    conv1_kernel = graph.get_tensor_by_name('conv1/kernel:0').eval()
#    conv1_bias = graph.get_tensor_by_name('conv1/bias:0').eval()
#    conv1_bn_gamma = graph.get_tensor_by_name('conv1/bias:0').eval()
#    conv1_bn_beta = graph.get_tensor_by_name('conv1/bias:0').eval()
#    
#    conv2_kernel = graph.get_tensor_by_name('conv2/kernel:0').eval()
#    conv2_bias = graph.get_tensor_by_name('conv2/bias:0').eval()
#    conv2_bn_gamma = graph.get_tensor_by_name('conv2/bias:0').eval()
#    conv2_bn_beta = graph.get_tensor_by_name('conv2/bias:0').eval()
#    
#    conv3_kernel = graph.get_tensor_by_name('conv3/kernel:0').eval()
#    conv3_bias = graph.get_tensor_by_name('conv3/bias:0').eval()
#    
#    W = graph.get_tensor_by_name('3_secondary_caps_layer/W:0').eval()
#    
#    hidden_vernieroffset_kernel = graph.get_tensor_by_name('hidden_vernieroffset/kernel:0').eval()
#    hidden_vernieroffset_bn_gamma = graph.get_tensor_by_name('hidden_vernieroffset_bn/gamma:0').eval()
#    hidden_vernieroffset_bn_beta = graph.get_tensor_by_name('hidden_vernieroffset_bn/beta:0').eval()
#    
#    hidden_nshapes_kernel = graph.get_tensor_by_name('hidden_nshapes/kernel:0').eval()
#    hidden_nshapes_bn_gamma = graph.get_tensor_by_name('hidden_nshapes_bn/gamma:0').eval()
#    hidden_nshapes_bn_beta = graph.get_tensor_by_name('hidden_nshapes_bn/beta:0').eval()
#    
#    hidden_xshape_kernel = graph.get_tensor_by_name('hidden_xshape/kernel:0').eval()
#    hidden_x_bnshape_gamma = graph.get_tensor_by_name('hidden_x_bnshape/gamma:0').eval()
#    hidden_x_bnshape_beta = graph.get_tensor_by_name('hidden_x_bnshape/beta:0').eval()
#    
#    hidden_yshape_kernel = graph.get_tensor_by_name('hidden_yshape/kernel:0').eval()
#    hidden_y_bnshape_gamma = graph.get_tensor_by_name('hidden_y_bnshape/gamma:0').eval()
#    hidden_y_bnshape_beta = graph.get_tensor_by_name('hidden_y_bnshape/beta:0').eval()
#    
#    hidden_xvernier_kernel = graph.get_tensor_by_name('hidden_xvernier/kernel:0').eval()
#    hidden_x_bnvernier_gamma = graph.get_tensor_by_name('hidden_x_bnvernier/gamma:0').eval()
#    hidden_x_bnvernier_beta = graph.get_tensor_by_name('hidden_x_bnvernier/beta:0').eval()
#    
#    hidden_yvernier_kernel = graph.get_tensor_by_name('hidden_yvernier/kernel:0').eval()
#    hidden_y_bnvernier_gamma = graph.get_tensor_by_name('hidden_y_bnvernier/gamma:0').eval()
#    hidden_y_bnvernier_beta = graph.get_tensor_by_name('hidden_y_bnvernier/beta:0').eval()
#
#
#
#params = {'conv1_kernel': conv1_kernel,
#          'conv1_bias': conv1_bias,
#          'conv1_bn_gamma': conv1_bn_gamma,
#          'conv1_bn_beta': conv1_bn_beta,
#          'conv2_kernel': conv2_kernel,
#          'conv2_bias': conv2_bias,
#          'conv2_bn_gamma': conv2_bn_gamma,
#          'conv2_bn_beta': conv2_bn_beta,
#          'conv3_kernel': conv3_kernel,
#          'conv3_bias': conv3_bias,
#          'W': W,
#          'hidden_vernieroffset_kernel': hidden_vernieroffset_kernel,
#          'hidden_vernieroffset_bn_gamma': hidden_vernieroffset_bn_gamma,
#          'hidden_vernieroffset_bn_beta': hidden_vernieroffset_bn_beta,
#          'hidden_nshapes_kernel': hidden_nshapes_kernel,
#          'hidden_nshapes_bn_gamma': hidden_nshapes_bn_gamma,
#          'hidden_nshapes_bn_beta': hidden_nshapes_bn_beta,
#          'hidden_xshape_kernel': hidden_xshape_kernel,
#          'hidden_x_bnshape_gamma': hidden_x_bnshape_gamma,
#          'hidden_x_bnshape_beta': hidden_x_bnshape_beta,
#          'hidden_yshape_kernel': hidden_yshape_kernel,
#          'hidden_y_bnshape_gamma': hidden_y_bnshape_gamma,
#          'hidden_y_bnshape_beta': hidden_y_bnshape_beta,
#          'hidden_xvernier_kernel': hidden_xvernier_kernel,
#          'hidden_x_bnvernier_gamma': hidden_x_bnvernier_gamma,
#          'hidden_x_bnvernier_beta': hidden_x_bnvernier_beta,
#          'hidden_yvernier_kernel': hidden_yvernier_kernel,
#          'hidden_y_bnvernier_gamma': hidden_y_bnvernier_gamma,
#          'hidden_y_bnvernier_beta': hidden_y_bnvernier_beta}


#trainables = ['conv1/kernel:0',
#              'conv1/bias:0',
#              'conv1_bn/gamma:0',
#              'conv1_bn/beta:0',
#              'conv2/kernel:0',
#              'conv2/bias:0',
#              'conv2_bn/gamma:0',
#              'conv2_bn/beta:0',
#              'conv3/kernel:0',
#              'conv3/bias:0',
#              '3_secondary_caps_layer/W:0',
#              'hidden_vernieroffset/kernel:0',
#              'hidden_vernieroffset_bn/gamma:0',
#              'hidden_vernieroffset_bn/beta:0',
#              'hidden_nshapes/kernel:0',
#              'hidden_nshapes_bn/gamma:0',
#              'hidden_nshapes_bn/beta:0',
#              'hidden_xshape/kernel:0',
#              'hidden_x_bnshape/gamma:0',
#              'hidden_x_bnshape/beta:0',
#              'hidden_yshape/kernel:0',
#              'hidden_y_bnshape/gamma:0',
#              'hidden_y_bnshape/beta:0',
#              'hidden_xvernier/kernel:0',
#              'hidden_x_bnvernier/gamma:0',
#              'hidden_x_bnvernier/beta:0',
#              'hidden_yvernier/kernel:0',
#              'hidden_y_bnvernier/gamma:0',
#              'hidden_y_bnvernier/beta:0']

#params = {'trainables': trainables}

##################################
#    Training and evaluation:    #
##################################
# Reset default graph again:
tf.reset_default_graph()

# Output the loss in the terminal every few steps:
logging.getLogger().setLevel(logging.INFO)

#ws = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=parameters.logdir, vars_to_warm_start=trainables)

# Beholder to check on weights during training in tensorboard:
beholder = Beholder(parameters.logdir_reconstruction)
beholder_hook = BeholderHook(parameters.logdir_reconstruction)

# Create the estimator:
capser = tf.estimator.Estimator(model_fn=model_fn, model_dir=parameters.logdir_reconstruction)
train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=parameters.n_steps, hooks=[beholder_hook])
eval_spec = tf.estimator.EvalSpec(eval_input_fn, steps=parameters.eval_steps, throttle_secs=parameters.eval_throttle_secs)

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
    '_noise_' + str(parameters.test_noise[0]) + '_' + str(parameters.test_noise[1]) + '.txt'
    if not os.path.exists(txt_file_name):
        with open(txt_file_name, 'w') as f:
            f.write(category + ' : \t' + str(results) + '\n')
    else:
        with open(txt_file_name, 'a') as f:
            f.write(category + ' : \t' + str(results) + '\n')


print('... Finished capsnet script!')
print('-------------------------------------------------------')
