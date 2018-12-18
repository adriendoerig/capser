# -*- coding: utf-8 -*-
"""
My capsnet: let's decode the reconstructions seperately
@author: Lynn

Last update on 18.12.18
-> first draft of the reconstruction code
-> unfortunately, the code is not as flexible as I would like it to be
-> decide whether to train whole model or only the reconstruction decoder
-> small change in secondary_caps_layer() with regards to reconstruction decoder script
-> loss is in the if clause now
-> finally working (thanks to Adrien)
-> combined reconstruction_main and reconstruction_model_fn
"""

import logging
import numpy as np
import tensorflow as tf
from tensorboard.plugins.beholder import Beholder
from tensorboard.plugins.beholder import BeholderHook

from my_parameters import parameters
from my_capser_functions import \
conv_layers, primary_caps_layer, secondary_caps_layer, \
predict_shapelabels, create_masked_decoder_input, compute_margin_loss, \
compute_accuracy, compute_reconstruction, compute_reconstruction_loss, \
compute_vernieroffset_loss
from my_capser_input_fn import train_input_fn, eval_input_fn
from my_capser_functions import save_params


print('-------------------------------------------------------')
print('TF version:', tf.__version__)
print('Starting capsnet script...')
print('-------------------------------------------------------')


##################################
#           model_fn:            #
##################################
def model_fn(features, labels, mode, params):
    vernier_images = features['vernier_images']
    shape_images = features['shape_images']
    shapelabels = labels
    vernierlabels = features['vernier_offsets']

    mask_with_labels = tf.placeholder_with_default(features['mask_with_labels'], shape=(), name='mask_with_labels')
    is_training = tf.placeholder_with_default(features['is_training'], shape=(), name='is_training')

    tf.summary.histogram('input_vernier_images', vernier_images)
    tf.summary.histogram('input_shape_images', shape_images)

    X = tf.add(vernier_images, shape_images)
    tf.summary.image('input_images', X, 6)

    ##########################################
    #          Build the capsnet:            #
    ##########################################
    # Create convolutional layers and their output:
    conv_output = conv_layers(X, parameters, is_training)
    
    # Create primary caps and their output:
    caps1_output = primary_caps_layer(conv_output, parameters)
    
    # Create secondary caps and their output and also divide vernier caps activation and shape caps activation:
    caps2_output, caps2_output_norm = secondary_caps_layer(caps1_output, parameters)
    vernier_caps_activation = caps2_output[:, :, 0, :, :]
    vernier_caps_activation = tf.expand_dims(vernier_caps_activation, 2)

    # Give me vernier acuity based on what the vernier capsule is doing:
    with tf.name_scope('1_vernier_acuity'):
        pred_vernierlabels, vernieroffset_loss, vernieroffset_accuracy = compute_vernieroffset_loss(vernier_caps_activation,
                                                                                                    vernierlabels, parameters,
                                                                                                    is_training)
        tf.summary.scalar('vernieroffset_loss', parameters.alpha_vernieroffset * vernieroffset_loss)
        tf.summary.scalar('vernieroffset_accuracy', vernieroffset_accuracy)


    ##########################################
    #       Train or Evaluation mode:        #
    ##########################################
    # How many shapes have to be predicted? Predict them:
    with tf.name_scope('2_margin'):
        n_shapes = shapelabels.shape[1]
        shapelabels_pred = predict_shapelabels(caps2_output, n_shapes)

        # Compute accuracy:
        accuracy = compute_accuracy(shapelabels, shapelabels_pred)
        tf.summary.scalar('margin_accuracy', accuracy)
        
        # Define the loss-function to be optimized
        margin_loss = compute_margin_loss(caps2_output_norm, shapelabels, parameters)
        tf.summary.scalar('margin_loss', parameters.alpha_margin * margin_loss)


    ##########################################
    #     Create masked decoder input        #
    ##########################################
    with tf.name_scope('3_Masked_decoder_input'):
        vernier_decoder_input = create_masked_decoder_input(
                mask_with_labels, shapelabels[:, 0], shapelabels_pred[:, 0], caps2_output, parameters)
        shape_decoder_input = create_masked_decoder_input(
                mask_with_labels, shapelabels[:, 1], shapelabels_pred[:, 1], caps2_output, parameters)


    ##########################################
    #         Decode reconstruction          #
    ##########################################
    # Restore parameters:
    var_list = tf.contrib.framework.get_variables_to_restore()
    tf.train.init_from_checkpoint(parameters.logdir, {v.name.split(':')[0]: v for v in var_list})
    
    with tf.name_scope('4_Reconstruction_loss'):
        # Create decoder outputs for vernier and shape images batch
        vernier_reconstructed_output, vernier_reconstructed_output_img = compute_reconstruction(vernier_decoder_input, parameters, is_training)
        shape_reconstructed_output, shape_reconstructed_output_img = compute_reconstruction(shape_decoder_input, parameters, is_training)

        decoder_output_img = vernier_reconstructed_output_img + shape_reconstructed_output_img

        tf.summary.image('decoder_output_img', decoder_output_img, 6)
        tf.summary.image('vernier_reconstructed_output_img', vernier_reconstructed_output_img, 6)
        tf.summary.image('shape_reconstructed_output_img', shape_reconstructed_output_img, 6)
        
        # Calculate reconstruction loss for vernier and shapes images batch
        vernier_reconstruction_loss = compute_reconstruction_loss(vernier_images, vernier_reconstructed_output, parameters)
        shape_reconstruction_loss = compute_reconstruction_loss(shape_images, shape_reconstructed_output, parameters)
        
        reconstruction_loss = vernier_reconstruction_loss + shape_reconstruction_loss

        tf.summary.scalar('vernier_reconstruction_loss', parameters.alpha_vernier_reconstruction * vernier_reconstruction_loss)
        tf.summary.scalar('shape_reconstruction_loss', parameters.alpha_shape_reconstruction * shape_reconstruction_loss)
        tf.summary.scalar('reconstruction_loss', reconstruction_loss)
    
    
    ##########################################
    #        Training operations             #
    ##########################################
    train_vars = [var for var in tf.trainable_variables() if 'reconstruct' in var.name]

    final_loss = tf.add_n([parameters.alpha_vernier_reconstruction * vernier_reconstruction_loss,
                           parameters.alpha_shape_reconstruction * shape_reconstruction_loss],
                          name='final_loss')

    # The following is needed due to how tf.layers.batch_normalzation works:
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate=parameters.learning_rate)
        train_op = optimizer.minimize(loss=final_loss, var_list=train_vars,
                                      global_step=tf.train.get_global_step(), name='train_op')
    
    # write summaries during evaluation
    eval_summary_hook = tf.train.SummarySaverHook(save_steps=100,
                                                  output_dir=parameters.logdir_reconstruction + '/eval',
                                                  summary_op=tf.summary.merge_all())
    
    # Wrap all of this in an EstimatorSpec.
    spec = tf.estimator.EstimatorSpec(
        mode=mode,
        loss=final_loss,
        train_op=train_op,
        eval_metric_ops={},
        evaluation_hooks=[eval_summary_hook])
    
    return spec


##################################
#    Training and evaluation:    #
##################################
# For reproducibility:
tf.reset_default_graph()
np.random.seed(42)
tf.set_random_seed(42)

# Output the loss in the terminal every few steps:
logging.getLogger().setLevel(logging.INFO)

# Beholder to check on weights during training in tensorboard:
beholder = Beholder(parameters.logdir_reconstruction)
beholder_hook = BeholderHook(parameters.logdir_reconstruction)

# Create the estimator:
capser = tf.estimator.Estimator(model_fn=model_fn, model_dir=parameters.logdir_reconstruction)
train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=parameters.n_steps, hooks=[beholder_hook])
eval_spec = tf.estimator.EvalSpec(eval_input_fn, steps=parameters.eval_steps, throttle_secs=parameters.eval_throttle_secs)

# Save parameters from parameter file for reproducability
save_params(parameters.logdir_reconstruction, parameters)

# Lets go!
tf.estimator.train_and_evaluate(capser, train_spec, eval_spec)


print('... Finished capsnet script!')
print('-------------------------------------------------------')