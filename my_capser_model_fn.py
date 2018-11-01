# -*- coding: utf-8 -*-
"""
Second try: model fn with tfrecords files
Last update on 01.11.2018
@author: Lynn

All functions that are called in this script are described in more detail in
my_capser_functions.py
"""

import tensorflow as tf
import numpy as np

from my_parameters import parameters
from my_parameters import conv1_params, conv2_params, conv3_params
from my_capser_functions import \
conv_layers, primary_caps_layer, secondary_caps_layer, \
predict_shapelabels, \
compute_margin_loss, compute_reconstruction, compute_reconstruction_loss, \
compute_accuracy, compute_vernieroffset_loss


def model_fn(features, labels, mode, params):
    # features: This is the x-arg from the input_fn.
    # labels:   This is the y-arg from the input_fn.
    # mode:     Either TRAIN, EVAL, or PREDICT
    # params:   Optional parameters; here not needed because of parameter-file
    
    # Inputs
    vernier_images = features['vernier_images']
    shape_images = features['shape_images']
    shapelabels = labels
    vernierlabels = features['vernier_offsets']
    mask_with_labels = tf.placeholder_with_default(features['mask_with_labels'], shape=(), name='mask_with_labels')
    is_training = tf.placeholder_with_default(features['is_training'], shape=(), name='is_training')

    # For the case that we want a vernier only condition (not implemented yet and also not sure whether this makes sense):
    if parameters.only_venier and features['is_training']:
        only_vernier_cond_idx = np.random.choice(parameters.batch_size, int(parameters.batch_size * parameters.only_venier_percent), False)
        print(only_vernier_cond_idx, '\n')
        tf.assign(shape_images[only_vernier_cond_idx, :, :, :], vernier_images[only_vernier_cond_idx, :, :, :])
        shapelabels[only_vernier_cond_idx, 1] = 0

    X = tf.add(vernier_images, shape_images)
    tf.summary.image('input_images', X, 6)
    
    ###################################################
    
    # Create convolutional layers and their output:
    conv_output = conv_layers(X, conv1_params, conv2_params, conv3_params, parameters, is_training)
    
    # Create primary caps and their output:
    caps1_output = primary_caps_layer(conv_output, parameters)
    
    # Create secondary caps and their output and also divide vernier caps activation and shape caps activation:
    caps2_output, caps2_output_norm = secondary_caps_layer(caps1_output, parameters)
    vernier_caps_activation = caps2_output[:, :, 0, :, :]
    vernier_caps_activation = tf.expand_dims(vernier_caps_activation, 2)

    # Give me different measures of vernier acuity only based on what the vernier capsule is doing:
    pred_vernierlabels, vernieroffset_loss, vernieroffset_accuracy = compute_vernieroffset_loss(vernier_caps_activation, vernierlabels)
    
    ###################################################
    
    # Training
    if mode == tf.estimator.ModeKeys.PREDICT:
        # If in prediction-mode use (one of) the following for predictions:
        # Since accuracy is calculated over whole batch, we have to repeat it
        # batch_size times (coz all prediction vectors must be same length)
        predictions = {'vernier_offsets': pred_vernierlabels,
                       'vernier_labels': tf.squeeze(vernierlabels),
                       'vernier_accuracy': tf.ones(shape=parameters.batch_size) * vernieroffset_accuracy}
        spec = tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions)
        
    else:
        # Otherwise the estimator is either in train or eval mode
        n_shapelabels = shapelabels.shape[1]

        # Estimated class probabilities
        shapelabels_pred = predict_shapelabels(caps2_output, n_shapelabels)

        # Compute accuracy:
        accuracy = compute_accuracy(shapelabels, shapelabels_pred)
        tf.summary.scalar('1_accuracy', accuracy)
        
        # Define the loss-function to be optimized
        margin_loss = compute_margin_loss(caps2_output_norm, shapelabels, parameters)
        tf.summary.scalar('2_margin_loss', parameters.alpha_margin * margin_loss)
        
        with tf.name_scope('3_Reconstruction_loss'):
            # Create decoder outputs for vernier and shape images batch
            # Note: instead of passing vernier_caps_activation, you can also pass caps2_output but I think it slows down the calculations
            vernier_decoder_output, vernier_decoder_output_img = compute_reconstruction(
                    mask_with_labels, shapelabels[:, 0], shapelabels_pred[:, 0], vernier_caps_activation, parameters, is_training, '_vernier')
            shape_decoder_output, shape_decoder_output_img = compute_reconstruction(
                    mask_with_labels, shapelabels[:, 1], shapelabels_pred[:, 1], caps2_output, parameters, is_training, '_shape')
            decoder_output_img = vernier_decoder_output_img + shape_decoder_output_img

            tf.summary.image('decoder_output_img', decoder_output_img, 6)
            tf.summary.image('vernier_decoder_output_img', vernier_decoder_output_img, 6)
            tf.summary.image('shape_decoder_output_img', shape_decoder_output_img, 6)
            
            # Calculate reconstruction loss for vernier and shapes images batch

            vernier_reconstruction_loss = compute_reconstruction_loss(vernier_images, vernier_decoder_output, parameters)
            shape_reconstruction_loss = compute_reconstruction_loss(shape_images, shape_decoder_output, parameters)
            reconstruction_loss = vernier_reconstruction_loss + shape_reconstruction_loss

            tf.summary.scalar('vernier_reconstruction_loss', parameters.alpha_vernier_reconstruction * vernier_reconstruction_loss)
            tf.summary.scalar('shape_reconstruction_loss', parameters.alpha_shape_reconstruction * shape_reconstruction_loss)
            tf.summary.scalar('reconstruction_loss', reconstruction_loss)


        tf.summary.scalar('4_vernieroffset_loss', parameters.alpha_vernieroffset * vernieroffset_loss)
        tf.summary.scalar('5_vernieroffset_accuracy', vernieroffset_accuracy)

        # Combine all previously defined losses:
        final_loss = tf.add_n([parameters.alpha_margin * margin_loss,
                               parameters.alpha_vernier_reconstruction * vernier_reconstruction_loss,
                               parameters.alpha_shape_reconstruction * shape_reconstruction_loss,
                               parameters.alpha_vernieroffset * vernieroffset_loss],
                              name='final_loss')


        # Training operations: learning rate
        global_step = tf.train.get_global_step()
        if parameters.exp_learning_decay:
            learning_rate = tf.train.exponential_decay(parameters.learning_rate, global_step, parameters.decay_steps, parameters.decay_rate, staircase=True)
        else:
            learning_rate = parameters.learning_rate

        # Training operations: Adam optimizer considering bn if applied
        if parameters.batch_norm_conv or parameters.batch_norm_decoder:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                train_op = optimizer.minimize(loss=final_loss, global_step=global_step, name='train_op')
        else:
            optimizer = tf.train.AdamOptimizer(learning_rate=parameters.learning_rate)
            train_op = optimizer.minimize(loss=final_loss, global_step=tf.train.get_global_step(), name='train_op')
        
        # write summaries during evaluation
        eval_summary_hook = tf.train.SummarySaverHook(save_steps=100,
                                                      output_dir=parameters.logdir + '/eval',
                                                      summary_op=tf.summary.merge_all())
        
        # Wrap all of this in an EstimatorSpec.
        spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=final_loss,
            train_op=train_op,
            eval_metric_ops={},
            evaluation_hooks=[eval_summary_hook])
    
    return spec


