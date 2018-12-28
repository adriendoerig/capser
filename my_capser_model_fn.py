# -*- coding: utf-8 -*-
"""
My capsnet: model_fn needed for tf train_and_evaluate API
@author: Lynn
All functions that are called in this script are described in more detail in
my_capser_functions.py

Last update on 19.12.2018
-> introduction of nshapes and location loss
-> reconstruction loss now optional
-> added some summaries
-> added alphas for each coordinate type
-> network can be run with 2 or 3 conv layers now
-> you can choose now between xentropy of squared_diff as location or nshapes loss
-> it is possible now to use batch normalization for every type of loss, this involved some major changes in the code!
-> small change for reconstruction decoder
-> clip values of added images too!
-> small change for reconstruction decoder with conv layers
"""

import tensorflow as tf

from my_parameters import parameters
from my_capser_functions import \
conv_layers, primary_caps_layer, secondary_caps_layer, \
predict_shapelabels, create_masked_decoder_input, compute_margin_loss, \
compute_accuracy, compute_reconstruction, compute_reconstruction_loss, \
compute_vernieroffset_loss, compute_nshapes_loss, compute_location_loss


def model_fn(features, labels, mode, params):
    # features: This is the x-arg from the input_fn.
    # labels:   This is the y-arg from the input_fn.
    # mode:     Either TRAIN, EVAL, or PREDICT
    # params:   Optional parameters; here not needed because of parameter-file
    
    ##########################################
    #      Prepararing input variables:      #
    ##########################################
    vernier_images = features['vernier_images']
    shape_images = features['shape_images']
    shapelabels = labels
    nshapeslabels = features['nshapeslabels']
    vernierlabels = features['vernier_offsets']
    x_shape = features['x_shape']
    y_shape = features['y_shape']
    x_vernier = features['x_vernier']
    y_vernier = features['y_vernier']
    mask_with_labels = tf.placeholder_with_default(features['mask_with_labels'], shape=(), name='mask_with_labels')
    is_training = tf.placeholder_with_default(features['is_training'], shape=(), name='is_training')
    
    tf.summary.histogram('input_vernier_images', vernier_images)
    tf.summary.histogram('input_shape_images', shape_images)
    
    X = tf.add(vernier_images, shape_images, name='X')
    X = tf.clip_by_value(X, parameters.clip_values[0], parameters.clip_values[1], name='X_clipped')
    tf.summary.image('input_images', X, 6)


    ##########################################
    #          Build the capsnet:            #
    ##########################################
    # Create convolutional layers and their output:
    conv_output, conv_output_sizes = conv_layers(X, parameters, is_training)
    
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
    #            Prediction mode:            #
    ##########################################
    if mode == tf.estimator.ModeKeys.PREDICT:
        # If in prediction-mode use (one of) the following for predictions:
        # Since accuracy is calculated over whole batch, we have to repeat it
        # batch_size times (coz all prediction vectors must be same length)
        predictions = {'vernier_accuracy': tf.ones(shape=parameters.batch_size) * vernieroffset_accuracy}
        spec = tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)


    ##########################################
    #       Train or Evaluation mode:        #
    ##########################################
    else:
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
        with tf.name_scope('4_Reconstruction_loss'):
            if parameters.decode_reconstruction:
                # Create decoder outputs for vernier and shape images batch
                vernier_reconstructed_output, vernier_reconstructed_output_img = compute_reconstruction(
                        vernier_decoder_input, parameters, is_training, conv_output_sizes)
                shape_reconstructed_output, shape_reconstructed_output_img = compute_reconstruction(
                        shape_decoder_input, parameters, is_training, conv_output_sizes)

                decoder_output_img = vernier_reconstructed_output_img + shape_reconstructed_output_img
    
                tf.summary.image('decoder_output_img', decoder_output_img, 6)
                tf.summary.image('vernier_reconstructed_output_img', vernier_reconstructed_output_img, 6)
                tf.summary.image('shape_reconstructed_output_img', shape_reconstructed_output_img, 6)
                
                # Calculate reconstruction loss for vernier and shapes images batch
                vernier_reconstruction_loss = compute_reconstruction_loss(vernier_images, vernier_reconstructed_output, parameters)
                shape_reconstruction_loss = compute_reconstruction_loss(shape_images, shape_reconstructed_output, parameters)

            else:
                vernier_reconstruction_loss = 0
                shape_reconstruction_loss = 0
            
            reconstruction_loss = vernier_reconstruction_loss + shape_reconstruction_loss
    
            tf.summary.scalar('vernier_reconstruction_loss', parameters.alpha_vernier_reconstruction * vernier_reconstruction_loss)
            tf.summary.scalar('shape_reconstruction_loss', parameters.alpha_shape_reconstruction * shape_reconstruction_loss)
            tf.summary.scalar('reconstruction_loss', reconstruction_loss)


    ##########################################
    #            Decode nshapes              #
    ##########################################
        with tf.name_scope('5_Nshapes_loss'):
            if parameters.decode_nshapes:
                nshapes_loss, nshapes_accuracy = compute_nshapes_loss(shape_decoder_input, nshapeslabels, parameters, is_training)
            else:
                nshapes_loss = 0
                nshapes_accuracy = 0
                
            tf.summary.scalar('nshapes_loss', parameters.alpha_nshapes * nshapes_loss)
            tf.summary.scalar('nshapes_accuracy', nshapes_accuracy)


    ##########################################
    #       Decode x and y coordinates       #
    ##########################################
        with tf.name_scope('6_Location_loss'):
            if parameters.decode_location:
                x_shapeloss, y_shapeloss = compute_location_loss(shape_decoder_input, x_shape, y_shape, parameters, 'shape', is_training)
                x_vernierloss, y_vernierloss = compute_location_loss(vernier_decoder_input, x_vernier, y_vernier, parameters, 'vernier', is_training)

                location_loss = parameters.alpha_x_shapeloss * x_shapeloss + \
                    parameters.alpha_y_shapeloss * y_shapeloss + \
                    parameters.alpha_x_vernierloss * x_vernierloss + \
                    parameters.alpha_y_vernierloss * y_vernierloss

            else:
                x_shapeloss = 0
                y_shapeloss = 0
                x_vernierloss = 0
                y_vernierloss = 0
                location_loss = 0
            
            tf.summary.scalar('x_shapeloss', parameters.alpha_x_shapeloss * x_shapeloss)
            tf.summary.scalar('y_shapeloss', parameters.alpha_y_shapeloss * y_shapeloss)
            tf.summary.scalar('x_vernierloss', parameters.alpha_x_vernierloss * x_vernierloss)
            tf.summary.scalar('y_vernierloss', parameters.alpha_y_vernierloss * y_vernierloss)
            tf.summary.scalar('location_loss', location_loss)


    ##########################################
    #              Final loss                #
    ##########################################
        final_loss = tf.add_n([parameters.alpha_margin * margin_loss,
                               parameters.alpha_vernier_reconstruction * vernier_reconstruction_loss,
                               parameters.alpha_shape_reconstruction * shape_reconstruction_loss,
                               parameters.alpha_vernieroffset * vernieroffset_loss,
                               parameters.alpha_nshapes * nshapes_loss,
                               parameters.alpha_x_shapeloss * x_shapeloss,
                               parameters.alpha_y_shapeloss * y_shapeloss, 
                               parameters.alpha_x_vernierloss * x_vernierloss,
                               parameters.alpha_y_vernierloss * y_vernierloss],
                              name='final_loss')


    ##########################################
    #        Training operations             #
    ##########################################
        # The following is needed due to how tf.layers.batch_normalzation works:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
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


