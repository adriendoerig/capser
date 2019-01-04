# -*- coding: utf-8 -*-
"""
My capsnet: let's decode the reconstructions seperately
@author: Lynn

Last update on 04.01.19
-> first draft of the reconstruction code
-> unfortunately, the code is not as flexible as I would like it to be
-> decide whether to train whole model or only the reconstruction decoder
-> small change in secondary_caps_layer() with regards to reconstruction decoder script
-> loss is in the if clause now
-> finally working (thanks to Adrien)
-> combined reconstruction_main and reconstruction_model_fn
-> clip values of added images too!
-> small change for reconstruction decoder with conv layers
-> use train_procedures 'vernier_shape', 'random_random' or 'random'
"""

import logging
import tensorflow as tf
from tensorboard.plugins.beholder import Beholder
from tensorboard.plugins.beholder import BeholderHook

from my_parameters import parameters
from my_capser_functions import \
conv_layers, primary_caps_layer, secondary_caps_layer, \
predict_shapelabels, create_masked_decoder_input, compute_margin_loss, \
compute_accuracy, compute_reconstruction, compute_reconstruction_loss
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
    shape_1_images = features['shape_1_images']
    shape_2_images = features['shape_2_images']
    shapelabels = labels

    mask_with_labels = tf.placeholder_with_default(features['mask_with_labels'], shape=(), name='mask_with_labels')
    is_training = tf.placeholder_with_default(features['is_training'], shape=(), name='is_training')

    if parameters.train_procedure=='vernier_shape' or parameters.train_procedure=='random_random':
        n_shapes = shapelabels.shape[1]
        input_images = tf.add(shape_1_images, shape_2_images, name='input_images')
        input_images = tf.clip_by_value(input_images, parameters.clip_values[0], parameters.clip_values[1], name='input_images_clipped')
        tf.summary.image('full_input_images', input_images, 6)

    elif parameters.train_procedure=='random':
        n_shapes = 1
        shapelabels = shapelabels[:, 0]
        input_images = shape_1_images
        input_images = tf.clip_by_value(input_images, parameters.clip_values[0], parameters.clip_values[1], name='input_images_clipped')
        tf.summary.image('full_input_images', input_images, 6)
    
    else:
        raise SystemExit('\nThe chosen train_procedure is unknown!\n')


    ##########################################
    #          Build the capsnet:            #
    ##########################################
    # Create convolutional layers and their output:
    conv_output, conv_output_sizes = conv_layers(input_images, parameters, is_training)
    
    # Create primary caps and their output:
    caps1_output = primary_caps_layer(conv_output, parameters)
    
    # Create secondary caps and their output and also divide vernier caps activation and shape caps activation:
    caps2_output, caps2_output_norm = secondary_caps_layer(caps1_output, parameters)
    shape_1_caps_activation = caps2_output[:, :, 0, :, :]
    shape_1_caps_activation = tf.expand_dims(shape_1_caps_activation, 2)

    
    # Just to have a little check whether it reproduced correctly:
    with tf.name_scope('2_margin'):
        shapelabels_pred = predict_shapelabels(caps2_output, n_shapes)

        # Compute accuracy:
        accuracy = compute_accuracy(shapelabels, shapelabels_pred)
        tf.summary.scalar('margin_accuracy', accuracy)

        # Define the loss-function to be optimized
        margin_loss = compute_margin_loss(caps2_output_norm, shapelabels, parameters)
        margin_loss = parameters.alpha_margin * margin_loss
        tf.summary.scalar('margin_loss', margin_loss)


    ##########################################
    #     Create masked decoder input        #
    ##########################################
    with tf.name_scope('3_Masked_decoder_input'):
        if parameters.train_procedure=='vernier_shape' or parameters.train_procedure=='random_random':
            shape_1_decoder_input = create_masked_decoder_input(
                    mask_with_labels, shapelabels[:, 0], shapelabels_pred[:, 0], caps2_output, parameters)
            shape_2_decoder_input = create_masked_decoder_input(
                    mask_with_labels, shapelabels[:, 1], shapelabels_pred[:, 1], caps2_output, parameters)
            
        elif parameters.train_procedure=='random':
            shape_1_decoder_input = create_masked_decoder_input(
                    mask_with_labels, shapelabels, shapelabels_pred, caps2_output, parameters)


    ##########################################
    #         Decode reconstruction          #
    ##########################################
    with tf.name_scope('4_Reconstruction_loss'):
        if n_shapes==2:
            # Create decoder outputs for shape_1 and shape_2 images batch
            shape_1_output_reconstructed = compute_reconstruction(shape_1_decoder_input, parameters, is_training, conv_output_sizes)
            shape_2_output_reconstructed = compute_reconstruction(shape_2_decoder_input, parameters, is_training, conv_output_sizes)
            
            shape_1_img_reconstructed = tf.reshape(
                    shape_1_output_reconstructed,
                    [parameters.batch_size, parameters.im_size[0], parameters.im_size[1], parameters.im_depth],
                    name='shape_1_img_reconstructed')
            shape_2_img_reconstructed = tf.reshape(
                    shape_2_output_reconstructed,
                    [parameters.batch_size, parameters.im_size[0], parameters.im_size[1], parameters.im_depth],
                    name='shape_2_img_reconstructed')

            decoder_output_img = shape_1_img_reconstructed + shape_2_img_reconstructed

            tf.summary.image('decoder_output_img', decoder_output_img, 6)
            tf.summary.image('shape_1_img_rec', shape_1_img_reconstructed, 6)
            tf.summary.image('shape_2_img_rec', shape_2_img_reconstructed, 6)
            
            # Calculate reconstruction loss for shape_1 and shape_2 images batch
            shape_1_reconstruction_loss = compute_reconstruction_loss(shape_1_images, shape_1_output_reconstructed, parameters)
            shape_2_reconstruction_loss = compute_reconstruction_loss(shape_2_images, shape_2_output_reconstructed, parameters)
            
            shape_1_reconstruction_loss = parameters.alpha_shape_1_reconstruction * shape_1_reconstruction_loss
            shape_2_reconstruction_loss = parameters.alpha_shape_2_reconstruction * shape_2_reconstruction_loss


        elif n_shapes==1:
            # Create decoder outputs for shape_1 images batch
            shape_1_output_reconstructed = compute_reconstruction(shape_1_decoder_input, parameters, is_training, conv_output_sizes)
            
            decoder_output_img = tf.reshape(
                    shape_1_output_reconstructed,
                    [parameters.batch_size, parameters.im_size[0], parameters.im_size[1], parameters.im_depth],
                    name='shape_1_img_reconstructed')

            tf.summary.image('decoder_output_img', decoder_output_img, 6)
            
            # Calculate reconstruction loss for shape_1 images batch
            shape_1_reconstruction_loss = compute_reconstruction_loss(shape_1_images, shape_1_output_reconstructed, parameters)
            
            shape_1_reconstruction_loss = parameters.alpha_shape_1_reconstruction * shape_1_reconstruction_loss
            shape_2_reconstruction_loss = 0.
        
        tf.summary.scalar('shape_1_reconstruction_loss', shape_1_reconstruction_loss)
        tf.summary.scalar('shape_2_reconstruction_loss', shape_2_reconstruction_loss)
    
    
    ##########################################
    #        Training operations             #
    ##########################################
    train_vars = [var for var in tf.trainable_variables() if 'reconstruct' in var.name]

    final_loss = tf.add_n([shape_1_reconstruction_loss, shape_2_reconstruction_loss], name='final_loss')

    # The following is needed due to how tf.layers.batch_normalzation works:
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        learning_rate = tf.train.cosine_decay_restarts(parameters.learning_rate, tf.train.get_global_step(),
                                                       parameters.learning_rate_decay_steps, name='learning_rate')
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
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