# -*- coding: utf-8 -*-
"""
My capsnet: let's decode the reconstructions seperately
@author: Lynn

Last update on 06.12.18
-> first draft of the reconstruction code
"""

restoration_file = '\model.ckpt-60000.meta'

import tensorflow as tf

from my_parameters import parameters
from my_capser_functions import safe_norm, routing_by_agreement, \
primary_caps_layer, compute_vernieroffset_loss, predict_shapelabels, \
compute_accuracy, compute_margin_loss, create_masked_decoder_input, \
compute_reconstruction, compute_reconstruction_loss

print('--------------------------------------')
print('TF version:', tf.__version__)
print('Starting reconstruction script...')
print('--------------------------------------')


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


    #########################################
    #    Restore all relevant parameters:   #
    #########################################
    tf.reset_default_graph()  
    imported_meta = tf.train.import_meta_graph(parameters.logdir + parameters.restoration_file)
    
    with tf.Session() as sess:
        # accessing the restored default graph and all operations:
        imported_meta.restore(sess, tf.train.latest_checkpoint(parameters.logdir))
        graph = tf.get_default_graph()
        
        # get kernels and biases for all conv layers:
        conv1_kernel_restored = graph.get_tensor_by_name('conv1/kernel:0')
        conv1_bias_restored = graph.get_tensor_by_name('conv1/bias:0')
        
        conv2_kernel_restored = graph.get_tensor_by_name('conv2/kernel:0')
        conv2_bias_restored = graph.get_tensor_by_name('conv2/bias:0')
        
        conv3_kernel_restored = graph.get_tensor_by_name('conv3/kernel:0')
        conv3_bias_restored = graph.get_tensor_by_name('conv3/bias:0')
        
        # restore weights between first and second caps layer
        W_restored = graph.get_tensor_by_name('3_secondary_caps_layer/W:0')
        
    
    ###################################
    #      Convolutional layers:      #
    ###################################
    # Set up the conv layers the same as before but initialize them with our restored values
    with tf.name_scope('1_convolutional_layers'):
        conv1 = tf.layers.conv2d(X, name='conv1', activation=None,
                                 kernel_initializer=conv1_kernel_restored,
                                 bias_initializer=conv1_bias_restored,
                                 **parameters.conv_params[0])
        if parameters.batch_norm_conv:
            conv1 = tf.layers.batch_normalization(conv1, training=is_training, name='conv1_bn')
        conv1 = tf.nn.relu(conv1)
        tf.summary.histogram('conv1_output', conv1)

        conv2 = tf.layers.conv2d(conv1, name='conv2', activation=None,
                                 kernel_initializer=conv2_kernel_restored,
                                 bias_initializer=conv2_bias_restored,
                                 **parameters.conv_params[1])
        if parameters.batch_norm_conv:
            conv2 = tf.layers.batch_normalization(conv2, training=is_training, name='conv2_bn')
        conv2 = tf.nn.relu(conv2)
        tf.summary.histogram('conv2_output', conv2)

        conv3 = tf.layers.conv2d(conv2, name='conv3', activation=None,
                                 kernel_initializer=conv3_kernel_restored,
                                 bias_initializer=conv3_bias_restored,
                                 **parameters.conv_params[2])
        conv_output = tf.nn.relu(conv3)
        tf.summary.histogram('conv3_output', conv_output)
    
    
    ###################################
    #     Primary capsule layer:      #
    ###################################
    caps1_output = primary_caps_layer(conv_output, parameters)


    ###################################
    #     Secondary capsule layer:    #
    ###################################
    # Set up the 2nd caps layer the same as before but initialize W with our restored values
    with tf.name_scope('3_secondary_caps_layer'):
        # Initialize and repeat weights for further calculations:
        W_init = W_restored
        W = tf.Variable(W_init, dtype=tf.float32, name='W')
        W_tiled = tf.tile(W, [parameters.batch_size, 1, 1, 1, 1], name='W_tiled')
    
        # Create second array by repeating the output of the 1st layer 10 times:
        caps1_output_expanded = tf.expand_dims(caps1_output, -1, name='caps1_output_expanded')
        caps1_output_tile = tf.expand_dims(caps1_output_expanded, 2, name='caps1_output_tile')
        caps1_output_tiled = tf.tile(caps1_output_tile, [1, 1, parameters.caps2_ncaps, 1, 1], name='caps1_output_tiled')
    
        # Now we multiply these matrices (matrix multiplication):
        caps2_predicted = tf.matmul(W_tiled, caps1_output_tiled, name='caps2_predicted')
        
        # Routing by agreement:
        caps2_output = routing_by_agreement(caps2_predicted, parameters.batch_size, parameters)
        
        # Compute the norm of the output for each output caps and each instance:
        caps2_output_norm = safe_norm(caps2_output, axis=-2, keepdims=True, name='caps2_output_norm')
        tf.summary.histogram('caps2_output_norm', caps2_output_norm[0, :, :, :])


    ###################################
    #    Calculate vernier acuity:    #
    ###################################
    # Give me vernier acuity based on what the vernier capsule is doing:
    with tf.name_scope('1_vernier_acuity'):
        vernier_caps_activation = caps2_output[:, :, 0, :, :]
        vernier_caps_activation = tf.expand_dims(vernier_caps_activation, 2)
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
        predictions = {'vernier_offsets': pred_vernierlabels,
                       'vernier_labels': tf.squeeze(vernierlabels),
                       'vernier_accuracy': tf.ones(shape=parameters.batch_size) * vernieroffset_accuracy}
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
            # Create decoder outputs for vernier and shape images batch
            vernier_reconstructed_output, vernier_reconstructed_output_img = compute_reconstruction(
                    vernier_decoder_input, parameters, is_training, '_vernier')
            shape_reconstructed_output, shape_reconstructed_output_img = compute_reconstruction(
                    shape_decoder_input, parameters, is_training, '_shape')

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
    #              Final loss                #
    ##########################################
        final_loss = tf.add_n([parameters.alpha_margin * margin_loss,
                               parameters.alpha_vernier_reconstruction * vernier_reconstruction_loss,
                               parameters.alpha_shape_reconstruction * shape_reconstruction_loss,
                               parameters.alpha_vernieroffset * vernieroffset_loss],
                              name='final_loss')
        
        
    ##########################################
    #        Training operations             #
    ##########################################
        if parameters.batch_norm_conv or parameters.batch_norm_reconstruction or parameters.batch_norm_vernieroffset or parameters.batch_norm_nshapes or parameters.batch_norm_location:
            # The following is needed due to how tf.layers.batch_normalzation works:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                optimizer = tf.train.AdamOptimizer(learning_rate=parameters.learning_rate)
                train_op = optimizer.minimize(loss=final_loss, global_step=tf.train.get_global_step(), name='train_op')
        else:
            optimizer = tf.train.AdamOptimizer(learning_rate=parameters.learning_rate)
            train_op = optimizer.minimize(loss=final_loss, global_step=tf.train.get_global_step(), name='train_op')
        
        # write summaries during evaluation
        eval_summary_hook = tf.train.SummarySaverHook(save_steps=100,
                                                      output_dir=parameters.logdir_rec + '/eval',
                                                      summary_op=tf.summary.merge_all())
        
        # Wrap all of this in an EstimatorSpec.
        spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=final_loss,
            train_op=train_op,
            eval_metric_ops={},
            evaluation_hooks=[eval_summary_hook])
    
    return spec


print('--------------------------------------')
print('Finished...')
print('--------------------------------------')
