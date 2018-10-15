# -*- coding: utf-8 -*-
"""
My capsnet: my model_fn
Version 3
Created on Thu Oct 11 10:57:08 2018
@author: Lynn
"""

import ipdb
import tensorflow as tf
import os
from time import time

from my_capser_functions import squash, safe_norm, routing_by_agreement, \
compute_margin_loss, compute_reconstruction, compute_reconstruction_loss, \
primary_caps_layer
from my_parameters import params as myparams
from my_parameters import conv1_params, conv2_params

def model_fn(features, labels, mode, params):
    
    if not os.path.exists(myparams.logdir):
        os.makedirs(myparams.logdir)

    # get inputs from .trecords file.
    X = features['X']
    imgs = tf.reshape(X, [myparams.batch_size, myparams.im_size[0], myparams.im_size[1], myparams.im_depth])
    tf.summary.image('input', imgs, 6)
    y = tf.cast(features['y'], tf.int64)
    mask_with_labels = tf.placeholder_with_default(features['mask_with_labels'], shape=(), name="mask_with_labels")
    
    ###################################################
    
    # Primary caps:
    # For computing the outputs, we first apply 2 regular conv. layers:
    # (padding: valid, actually means no padding)
    with tf.name_scope('0_early_conv_layers'):
        conv1 = tf.layers.conv2d(imgs, name="conv1", **conv1_params)
        conv2 = tf.layers.conv2d(conv1, name='conv2', **conv2_params)
#        tf.summary.histogram('1st_conv_layer', conv1)
#        tf.summary.histogram('2nd_conv_layer', conv2)
    
    with tf.name_scope('1st_caps'):
        caps1_output = primary_caps_layer(conv2, myparams)
        
        # display a histogram of primary capsule norms
#        caps1_output_norms = safe_norm(caps1_output, axis=-1, keepdims=True, name="caps1_output_norms")
#        tf.summary.histogram('Primary capsule norms', caps1_output_norms)
        
        # Create second array by repeating the output of the 1st layer 10 times:
        caps1_output_expanded = tf.expand_dims(caps1_output, -1, name='caps1_output_expanded')
        caps1_output_tile = tf.expand_dims(caps1_output_expanded, 2, name='caps1_output_tile')
        caps1_output_tiled = tf.tile(caps1_output_tile, [1, 1, myparams.caps2_ncaps, 1, 1], name='caps1_output_tiled')
    
    ###################################################
    
    with tf.name_scope('2nd_caps'):
        # Since we have 8D and need 16D, W must be 16x8 (overall: 1x1152x10x16x8)
        W_init = tf.random_normal(
                shape=(1, myparams.caps1_ncaps, myparams.caps2_ncaps, myparams.caps2_ndims, myparams.caps1_ndims),
                stddev=myparams.init_sigma, dtype=tf.float32, name='W_init')
        W = tf.Variable(W_init, name='W')
    
        # Create first array by repeating W once per instance:
        batch_size_tensor = tf.shape(imgs)[0]
        W_tiled = tf.tile(W, [batch_size_tensor, 1, 1, 1, 1], name='W_tilted')
        
        # Now we multiply these matrices (matrix multiplication):
        caps2_predicted = tf.matmul(W_tiled, caps1_output_tiled, name='caps2_predicted')
    
        # Routing by agreement:
        caps2_output = routing_by_agreement(caps2_predicted, batch_size_tensor, myparams)
        
        # Compute the norm of the output for each output caps and each instance:
        caps2_output_norm = safe_norm(caps2_output, axis=-2, keepdims=True, name='caps2_output_norm')
#        tf.summary.histogram('Output capsule norms', caps2_output_norm)
    
        # Estimated class probabilities
        # Since the lengths of the output vectors represent probabilties:
        y_proba = safe_norm(caps2_output, axis=-2, name='y_proba')
    
        # Prediction:
        y_proba_argmax = tf.argmax(y_proba, axis=2, name='y_proba_argmax')
        y_pred = tf.squeeze(y_proba_argmax, axis=[1, 2], name='y_pred')
    
    
    ###################################################

    with tf.name_scope('losses'):
        # Reconstruction: add a decoder network on top of the caps network
        # Compute reconstruction (two dense fully connected ReLU layers followed by a dense output sigmoid layer):
#        decoder_output = compute_reconstruction(mask_with_labels, y, y_pred, caps2_output, myparams)
#        decoder_output_images = tf.reshape(decoder_output, [-1, myparams.im_size[0], myparams.im_size[1], myparams.im_depth])
#        tf.summary.image('decoder_output', decoder_output_images, 6)
        
        # Compute losses:
        loss = compute_margin_loss(caps2_output_norm, y, myparams)
#        tf.summary.scalar('margin_loss', margin_loss)
#        reconstruction_loss = compute_reconstruction_loss(imgs, decoder_output, myparams)
#        tf.summary.scalar('reconstruction_loss', reconstruction_loss)
        
#        loss = tf.add_n([margin_loss,
#                            myparams.alpha_rec*reconstruction_loss], name='final_loss')
#        tf.summary.scalar('final_loss', loss)

    ###################################################

#    with tf.name_scope('accuracy'):
#        # Compute accuracy:
#        correct = tf.equal(y, y_pred, name='correct')
#        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')
#        tf.summary.scalar('accuracy', accuracy)

    ###################################################
    
    # to write summaries during evluation and prediction too
#    eval_summary_hook = tf.train.SummarySaverHook(save_steps=25,
#                                                  output_dir=myparams.logdir + '/eval',
#                                                  summary_op=tf.summary.merge_all())
#    pred_summary_hook = tf.train.SummarySaverHook(save_steps=1,
#                                                  output_dir=myparams.logdir + '/pred-' + str(time()),
#                                                  summary_op=tf.summary.merge_all())


    # Wrap all of this in an EstimatorSpec (cf. tf.Estimator tutorials, etc).
    if mode == tf.estimator.ModeKeys.PREDICT:
        # the following line is
        predictions = y_pred
        spec = tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions)
        print('Wrapping in EstimatorSpec worked\n')

    else:
        # Training operations:
        optimizer = tf.train.AdamOptimizer(learning_rate=myparams.learning_rate)
        training_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step(), name="training_op")
#        metrics = {'accuracy': tf.metrics.accuracy(y, y_pred)}

        spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=training_op,
            eval_metric_ops={})  # to write summaries during evaluatino too
        print('Training operations created\n')

    return spec