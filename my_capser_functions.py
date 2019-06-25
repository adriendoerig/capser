# -*- coding: utf-8 -*-
"""
My capsnet: all functions relevant for the capsnet
Including:
    squash, safe_norm, routing_by_agreement,
    conv_layers, primary_caps_layer, secondary_caps_layer,
    predict_shapelabels, create_masked_decoder_input, compute_margin_loss,
    compute_accuracy, compute_reconstruction, compute_reconstruction_loss,
    compute_vernieroffset_loss, compute_nshapes_loss, compute_location_loss
@author: Lynn

Last update on 28.05.2019
-> adaption for new project: different iter_routing for training and testing
-> adaptation of plot function for using only 2 conditions
-> added variable priming input for routing initialization
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os.path
import pdb

################################
#    Small helper function:    #
################################
def save_params(save_path, parameters):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    txt_file = save_path + '/parameters.txt'
    if os.path.exists(txt_file):
        raise SystemExit('\nPROBLEM: %s already exists!' % txt_file)
    else:
        with open(save_path + '/parameters.txt', 'w') as f_txt:
            f_py = open('./my_parameters.py')
            variables = f_py.read()
            f_txt.write(variables)
            print('Parameter values saved.')


################################
#     Helper plot function:    #
################################
def plot_uncrowding_results(results, categories, n_idx, save=None):
    results = np.array(results)*100
    
    # plot bar width & colors
    width = .3333  # the width of the bars
    color7 = (100. / 255, 170. / 255, 180. / 255)

    N = len(results)/n_idx
    ind = np.arange(N)  # the x locations for the groups
    fig, ax = plt.subplots()
    ax.bar(ind - width / n_idx, results[0::n_idx], width/n_idx, color=color7, edgecolor='black')
    ax.bar(ind, results[1::n_idx], width/n_idx, color=color7, edgecolor='black')
    if n_idx==3:
        ax.bar(ind + width / n_idx, results[2::n_idx], width/n_idx, color=color7, edgecolor='black')

    # add some text for labels, title and axes ticks, and save figure
    ax.set_ylabel('Percent correct')
    ax.set_title("Vernier decoder performance detecting vernier offset")
    ax.set_xticks(ind)
    ax.set_xticklabels(categories)
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')

    chance_line, = ax.plot([-2 * width, N], [50, 50], '#8d8f8c')  # chance level dashed line
    chance_line.set_dashes([1, n_idx, 1, n_idx])

    if save is None:
        plt.show()
        plt.close()
    else:
        plt.savefig(save)
        plt.close()


################################
#      Squash function:        #
################################
def squash(s, axis=-1, epsilon=1e-7, name=None):
    '''Squashing function as described in Sabour et al. (2018) but calculating
    the norm in a safe way (e.g. see Aurelion Geron youtube-video)'''
    with tf.name_scope(name, default_name='squash'):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keepdims=True)
        safe_norm = tf.sqrt(squared_norm + epsilon)
        squash_factor = squared_norm / (1. + squared_norm)
        unit_vector = s / safe_norm
        s_squashed = squash_factor * unit_vector
        return s_squashed


##############################
#        Safe norm           #
##############################
def safe_norm(s, axis=-1, epsilon=1e-7, keepdims=False, name=None):
    '''Safe calculation of the norm for estimated class probabilities'''
    with tf.name_scope(name, default_name='safe_norm'):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keepdims=keepdims)
        s_norm = tf.sqrt(squared_norm + epsilon)
        return s_norm


################################
#     Routing by agreement:    #
################################
def routing_by_agreement(caps2_predicted, batch_size_tensor, iter_routing, priming_input, parameters):
    # How often we do the routing:
    def routing_condition(raw_weights, caps2_output, counter, priming_input):
        output = tf.less(counter, iter_routing)
        return output

    # What the routing is:
    def routing_body(raw_weights, caps2_output, counter, priming_input):
        # use priming values if non zero for the first iteration:
        prime_cond = tf.cast(tf.reduce_sum(priming_input), tf.int32)
        routing_weights = tf.cond(tf.equal(counter, tf.constant(0)),
                                  true_fn=lambda: tf.cond(tf.equal(prime_cond, tf.constant(0)),
                                                          true_fn=lambda: tf.nn.softmax(raw_weights, axis=2, name='routing_weights'),
                                                          false_fn=lambda: tf.nn.softmax(priming_input, axis=2, name='routing_weights')),
                                  false_fn=lambda: tf.nn.softmax(raw_weights, axis=2, name='routing_weights'))
#        routing_weights = tf.nn.softmax(raw_weights, axis=2, name='routing_weights')
        weighted_predictions = tf.multiply(routing_weights, caps2_predicted, name='weighted_predictions')
        weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keepdims=True, name='weighted_sum')
        # use priming values if non zero for the first iteration:
#        prime_cond = tf.cast(tf.reduce_sum(priming_input), tf.int32)
#        caps2_output = tf.cond(tf.equal(counter, tf.constant(0)),
#                               true_fn=lambda: tf.cond(tf.equal(prime_cond, tf.constant(0)),
#                                                       true_fn=lambda: squash(weighted_sum, axis=-2, name='caps2_output'),
#                                                       false_fn=lambda: priming_input),
#                               false_fn=lambda: squash(weighted_sum, axis=-2, name='caps2_output'))
        caps2_output = squash(weighted_sum, axis=-2, name='caps2_output')
        caps2_output_tiled = tf.tile(caps2_output, [1, parameters.caps1_ncaps, 1, 1, 1], name='caps2_output_tiled')
        agreement = tf.matmul(caps2_predicted, caps2_output_tiled, transpose_a=True, name='agreement')
        raw_weights = tf.add(raw_weights, agreement, name='raw_weights')
        return raw_weights, caps2_output, tf.add(counter, 1), priming_input
    
    # Execution of routing via while-loop:
    with tf.name_scope('Routing_by_agreement'):
        # Initialize weights and caps2-output-array
        raw_weights = tf.zeros([batch_size_tensor, parameters.caps1_ncaps, parameters.caps2_ncaps, 1, 1], dtype=tf.float32, name='raw_weights')
        caps2_output = tf.zeros([batch_size_tensor, 1, parameters.caps2_ncaps, parameters.caps2_ndims, 1], dtype=tf.float32, name='caps2_output_init')
        # Counter for number of routing iterations:
        counter = tf.constant(0)
        raw_weights, caps2_output, counter, priming_input = tf.while_loop(routing_condition, routing_body, [raw_weights, caps2_output, counter, priming_input])
        return caps2_output, raw_weights


###################################
#     Create capser network:      #
###################################
def conv_layers(X, parameters, phase=True):
    with tf.name_scope('1_convolutional_layers'):
        # Conv1:
        if parameters.batch_norm_conv:
            conv1 = tf.layers.conv2d(X, name='conv1', use_bias=False, activation=None, **parameters.conv_params[0])
            conv1 = tf.layers.batch_normalization(conv1, training=phase, name='conv1_bn')
        else:
            conv1 = tf.layers.conv2d(X, name='conv1', activation=None, **parameters.conv_params[0])
        conv1 = tf.nn.elu(conv1)
        if parameters.dropout:
            conv1 = tf.layers.dropout(conv1, rate=.3, training=phase, name='dropout1')

        # Conv2:
        if parameters.batch_norm_conv:
            conv2 = tf.layers.conv2d(conv1, name='conv2', use_bias=False, activation=None, **parameters.conv_params[1])
            conv2 = tf.layers.batch_normalization(conv2, training=phase, name='conv2_bn')
        else:
            conv2 = tf.layers.conv2d(conv1, name='conv2', activation=None, **parameters.conv_params[1])
        conv2 = tf.nn.elu(conv2)
        if parameters.dropout:
            conv2 = tf.layers.dropout(conv2, rate=.3, training=phase, name='dropout2')

        # Conv3:
        conv3 = tf.layers.conv2d(conv2, name='conv3', activation=None, **parameters.conv_params[2])
        conv_output = tf.nn.elu(conv3)
        conv_output_sizes = [conv1.get_shape().as_list(), conv2.get_shape().as_list(), conv3.get_shape().as_list()]
                
        tf.summary.histogram('conv1_output', conv1)
        tf.summary.histogram('conv2_output', conv2)
        tf.summary.histogram('conv3_output', conv3)
        return conv_output, conv_output_sizes


def primary_caps_layer(conv_output, parameters):
    with tf.name_scope('2_primary_capsules'):
        caps1_reshaped = tf.reshape(conv_output, [-1, parameters.caps1_ncaps, parameters.caps1_ndims], name='caps1_reshaped')
        caps1_output = squash(caps1_reshaped, name='caps1_output')
        caps1_output_norm = safe_norm(caps1_output, axis=-1, keepdims=False, name='caps1_output_norm')
        tf.summary.histogram('caps1_output_norm', caps1_output_norm)
        return caps1_output


def secondary_caps_layer(caps1_output, batch_size, iter_routing, parameters, priming_input, W_init=None):
    with tf.name_scope('3_secondary_caps_layer'):
        # Initialize and repeat weights for further calculations:
        if W_init==None:
            W_init = lambda: tf.random_normal(
                shape=(1, parameters.caps1_ncaps, parameters.caps2_ncaps, parameters.caps2_ndims, parameters.caps1_ndims),
                stddev=parameters.init_sigma, dtype=tf.float32, seed=parameters.random_seed, name='W_init')

        W = tf.Variable(W_init, dtype=tf.float32, name='W')
        W_tiled = tf.tile(W, [batch_size, 1, 1, 1, 1], name='W_tiled')

        # Create second array by repeating the output of the 1st layer 10 times:
        caps1_output_expanded = tf.expand_dims(caps1_output, -1, name='caps1_output_expanded')
        caps1_output_tile = tf.expand_dims(caps1_output_expanded, 2, name='caps1_output_tile')
        caps1_output_tiled = tf.tile(caps1_output_tile, [1, 1, parameters.caps2_ncaps, 1, 1], name='caps1_output_tiled')
    
        # Now we multiply these matrices (matrix multiplication):
        caps2_predicted = tf.matmul(W_tiled, caps1_output_tiled, name='caps2_predicted')
        
        # Routing by agreement:
        caps2_output, raw_weights = routing_by_agreement(caps2_predicted, batch_size, iter_routing, priming_input, parameters)
        
        # Compute the norm of the output for each output caps and each instance:
        caps2_output_norm = safe_norm(caps2_output, axis=-2, keepdims=True, name='caps2_output_norm')
        tf.summary.histogram('caps2_output_norm', caps2_output_norm[0, :, :, :])
        return caps2_output, caps2_output_norm, raw_weights


################################
#         Margin loss:         #
################################
def predict_shapelabels(caps2_output, n_labels):
    with tf.name_scope('predict_shape'):
        # Calculate caps norm:
        labels_proba = safe_norm(caps2_output, axis=-2, keepdims=False, name='labels_proba')

        # Predict n_labels largest values:
        labels_pred_proba, labels_pred = tf.nn.top_k(labels_proba[:, 0, :, 0], n_labels, name='y_proba')
        labels_pred = tf.cast(labels_pred, tf.int64)
        labels_pred_sorted = tf.contrib.framework.sort(labels_pred, axis=-1, direction='ASCENDING', name='labels_pred_sorted')
        
        labels_pred_sorted_idx = tf.contrib.framework.argsort(labels_pred, axis=-1, direction='ASCENDING', name='labels_pred_sorted')
        labels_pred_proba_sorted = tf.batch_gather(labels_pred_proba, labels_pred_sorted_idx)
        return labels_pred_sorted, labels_pred_proba_sorted


def compute_margin_loss(caps2_output_norm, labels, parameters):
    with tf.name_scope('compute_margin_loss'):
        # Compute the loss for each instance and shape:
        # trick to get a vector for each image in the batch. Labels [0,2] -> [[1, 0, 1]] and [1,1] -> [[0, 1, 0]]
        T_shapelabels_raw = tf.one_hot(labels, depth=parameters.caps2_ncaps, name='T_shapelabels_raw')
        try:
            labels.shape[1]
            T_shapelabels = tf.reduce_sum(T_shapelabels_raw, axis=1)
        except:
            T_shapelabels = T_shapelabels_raw
        T_shapelabels = tf.minimum(T_shapelabels, 1)
        present_error_raw = tf.square(tf.maximum(0., parameters.m_plus - caps2_output_norm), name='present_error_raw')
        present_error = tf.reshape(present_error_raw, shape=(-1, parameters.caps2_ncaps), name='present_error')
        absent_error_raw = tf.square(tf.maximum(0., caps2_output_norm - parameters.m_minus), name='absent_error_raw')
        absent_error = tf.reshape(absent_error_raw, shape=(-1, parameters.caps2_ncaps), name='absent_error')
        L = tf.add(T_shapelabels * present_error, parameters.lambda_val * (1.0 - T_shapelabels) * absent_error, name='L')
        
        # Sum digit losses for each instance and compute mean:
        margin_loss = tf.reduce_mean(tf.reduce_sum(L, axis=1), name='margin_loss')
        return margin_loss


################################
#          Accuracy:           #
################################
def compute_accuracy(labels, labels_pred):
    with tf.name_scope('accuracy'):
        labels_reshaped = tf.reshape(labels, shape=[-1, 1])
        labels_pred_reshaped = tf.reshape(labels_pred, shape=[-1, 1])
        correct = tf.equal(labels_reshaped, labels_pred_reshaped, name='correct')
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')
        return accuracy


################################
#    Create decoder input:     #
################################
def create_masked_decoder_input(mask_with_labels, labels, labels_pred, caps2_output, parameters):
    with tf.name_scope('compute_decoder_input'):
        reconstruction_targets = tf.cond(mask_with_labels, lambda: labels, lambda: labels_pred, name='reconstruction_targets')
        reconstruction_mask = tf.one_hot(reconstruction_targets, depth=parameters.caps2_ncaps, name='reconstruction_mask')
        reconstruction_mask = tf.reshape(reconstruction_mask, [-1, 1, parameters.caps2_ncaps, 1, 1], name='reconstruction_mask')
        caps2_output_masked = tf.multiply(caps2_output, reconstruction_mask, name='caps2_output_masked')
        
        # Flatten decoder inputs:
        decoder_input = tf.reshape(caps2_output_masked, [-1, parameters.caps2_ndims*parameters.caps2_ncaps], name='decoder_input')
        return decoder_input


################################
#     Create reconstruction:   #
################################
def compute_reconstruction(decoder_input,  parameters, phase=True, conv_output_size=None):
    # Finally comes the decoder (two dense fully connected ELU layers followed by a dense output sigmoid layer):
    with tf.name_scope('decoder_reconstruction'):
        # Lets do it as in the original paper:
        if parameters.rec_decoder_type=='fc':
            # Hidden 1:
            if parameters.batch_norm_reconstruction:
                hidden_reconstruction_1 = tf.layers.dense(decoder_input, parameters.n_hidden_reconstruction_1, use_bias=False, 
                                                          reuse=tf.AUTO_REUSE, activation=None, name='hidden_reconstruction_1')
                hidden_reconstruction_1 = tf.layers.batch_normalization(hidden_reconstruction_1, training=phase,
                                                                        reuse=tf.AUTO_REUSE, name='hidden_reconstruction_1_bn')
            else:
                hidden_reconstruction_1 = tf.layers.dense(decoder_input, parameters.n_hidden_reconstruction_1, reuse=tf.AUTO_REUSE,
                                                          activation=None, name='hidden_reconstruction_1')
            hidden_reconstruction_1 = tf.nn.elu(hidden_reconstruction_1, name='hidden_reconstruction_1_activation')
            
            # Hidden 2:
            if parameters.batch_norm_reconstruction:
                hidden_reconstruction_2 = tf.layers.dense(hidden_reconstruction_1, parameters.n_hidden_reconstruction_2, use_bias=False,
                                                          reuse=tf.AUTO_REUSE, activation=None, name='hidden_reconstruction_2')
                hidden_reconstruction_2 = tf.layers.batch_normalization(hidden_reconstruction_2, training=phase,
                                                                        reuse=tf.AUTO_REUSE, name='hidden_reconstruction_2_bn')
            else:
                hidden_reconstruction_2 = tf.layers.dense(hidden_reconstruction_1, parameters.n_hidden_reconstruction_2, reuse=tf.AUTO_REUSE,
                                                          activation=None, name='hidden_reconstruction_2')
            hidden_reconstruction_2 = tf.nn.elu(hidden_reconstruction_2, name='hidden_reconstruction_2_activation')
    
            reconstructed_output = tf.layers.dense(hidden_reconstruction_2, parameters.n_output, reuse=tf.AUTO_REUSE, activation=tf.nn.sigmoid, name='reconstructed_output')

        # Lets deconvolute:
        elif parameters.rec_decoder_type=='conv':            
            # Redo step from primary caps (=conv3) to secondary caps (using fewer parameters):
            bottleneck_units = parameters.caps2_ncaps*parameters.caps2_ndims
#            bottleneck_units = parameters.caps2_ncaps
            upsample_size1 = [conv_output_size[-1][1], conv_output_size[-1][2]]
            if parameters.batch_norm_reconstruction:
                upsample1 = tf.layers.dense(decoder_input, upsample_size1[0] * upsample_size1[1] * bottleneck_units, use_bias=False,
                                            activation=None, reuse=tf.AUTO_REUSE, name='upsample1_reconstruction')
                upsample1 = tf.layers.batch_normalization(upsample1, training=phase, reuse=tf.AUTO_REUSE, name='upsample1_reconstruction_bn')
            else:
                upsample1 = tf.layers.dense(decoder_input, upsample_size1[0] * upsample_size1[1] * bottleneck_units,
                                            activation=None, reuse=tf.AUTO_REUSE, name='upsample1_reconstruction')
            upsample1 = tf.reshape(upsample1, [-1, upsample_size1[0], upsample_size1[1], bottleneck_units],
                                            name='reshaped_upsample_reconstruction')

            # Redo step from conv2 to primary caps (=conv3):
            upsample_size2 = [conv_output_size[-2][1], conv_output_size[-2][2]]
            upsample2 = tf.image.resize_images(upsample1, size=upsample_size2, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            if parameters.batch_norm_reconstruction:
                conv1_upsampled = tf.layers.conv2d(inputs=upsample2, filters=parameters.conv_params[-1]['filters'],
                                               kernel_size=parameters.conv_params[-1]['kernel_size'],
                                               use_bias=False, reuse=tf.AUTO_REUSE, padding='same', activation=None, name='conv1_upsampled_reconstruction')
                conv1_upsampled = tf.layers.batch_normalization(conv1_upsampled, training=phase, reuse=tf.AUTO_REUSE, name='conv1_upsampled_reconstruction_bn')
            else:
                conv1_upsampled = tf.layers.conv2d(inputs=upsample2, filters=parameters.conv_params[-1]['filters'],
                                               kernel_size=parameters.conv_params[-1]['kernel_size'],
                                               reuse=tf.AUTO_REUSE, padding='same', activation=None, name='conv1_upsampled_reconstruction')
            conv1_upsampled = tf.nn.elu(conv1_upsampled, name='conv1_upsampled_reconstruction_activation')

            # Redo step from conv1 to conv2:
            upsample_size3 = [conv_output_size[-3][1], conv_output_size[-3][2]]
            upsample3 = tf.image.resize_images(conv1_upsampled, size=upsample_size3, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            if parameters.batch_norm_reconstruction:
                conv2_upsampled = tf.layers.conv2d(inputs=upsample3, filters=parameters.conv_params[-2]['filters'],
                                               kernel_size=parameters.conv_params[-2]['kernel_size'],
                                               use_bias=False, reuse=tf.AUTO_REUSE, padding='same', activation=None, name='conv2_upsampled_reconstruction')
                conv2_upsampled = tf.layers.batch_normalization(conv2_upsampled, training=phase, reuse=tf.AUTO_REUSE, name='conv2_upsampled_reconstruction_bn')
            else:
                conv2_upsampled = tf.layers.conv2d(inputs=upsample3, filters=parameters.conv_params[-2]['filters'],
                                               kernel_size=parameters.conv_params[-2]['kernel_size'],
                                               reuse=tf.AUTO_REUSE, padding='same', activation=None, name='conv2_upsampled_reconstruction')
            conv2_upsampled = tf.nn.elu(conv2_upsampled, name='conv2_upsampled_reconstruction_activation')

            # Redo step from input image to conv1:
            upsample_size4 = parameters.im_size
            upsample4 = tf.image.resize_images(conv2_upsampled, size=(upsample_size4), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            if parameters.batch_norm_reconstruction:
                conv3_upsampled = tf.layers.conv2d(inputs=upsample4, filters=parameters.conv_params[-3]['filters'],
                                               kernel_size=parameters.conv_params[-3]['kernel_size'],
                                               use_bias=False, reuse=tf.AUTO_REUSE, padding='same', activation=None, name='conv3_upsampled_reconstruction')
                conv3_upsampled = tf.layers.batch_normalization(conv3_upsampled, training=phase, reuse=tf.AUTO_REUSE, name='conv3_upsampled_reconstruction_bn')
            else:
                conv3_upsampled = tf.layers.conv2d(inputs=upsample4, filters=parameters.conv_params[-3]['filters'],
                                               kernel_size=parameters.conv_params[-3]['kernel_size'],
                                               reuse=tf.AUTO_REUSE, padding='same', activation=None, name='conv3_upsampled_reconstruction')
            conv3_upsampled = tf.nn.elu(conv3_upsampled, name='conv3_upsampled_reconstruction_activation')

            # Get back to greyscale
            reconstructed_output = tf.layers.conv2d(inputs=conv3_upsampled, filters=parameters.im_depth, kernel_size=1, padding='same',
                                                    use_bias=True, reuse=tf.AUTO_REUSE, activation=tf.nn.sigmoid, name='reconstructed_output')
            
            # Flatten the output to make it equal to the output of the fc
            reconstructed_output = tf.reshape(reconstructed_output, [-1, parameters.n_output], name='reconstructed_output_flat')
        
        else:
            raise SystemExit('\nPROBLEM: Your reconstruction decoder does not know what to do!\nCheck rec_decoder_type')
        
        return reconstructed_output


################################
#     Reconstruction loss:     #
################################
def compute_reconstruction_loss(X, reconstructed_output, parameters):
    with tf.name_scope('compute_reconstruction_loss'):
        imgs_flat = tf.reshape(X, [-1, parameters.n_output], name='imgs_flat')
        imgs_flat = tf.cast(imgs_flat, tf.float32)
        squared_difference = tf.square(imgs_flat - reconstructed_output, name='squared_difference')
        # reconstruction_loss = tf.reduce_mean(squared_difference, name='reconstruction_loss')
        reconstruction_loss = tf.reduce_sum(squared_difference, name='reconstruction_loss')
        return reconstruction_loss


################################
#     Vernieroffset loss:      #
################################
def compute_vernieroffset_loss(shape_1_caps_activation, vernierlabels, parameters, phase=True):
    with tf.name_scope('compute_vernieroffset_loss'):
        if parameters.train_procedure=='vernier_shape':
            depth = 2
        else:
            depth = 3
        
        shape_1_caps_activation = tf.squeeze(shape_1_caps_activation, [1, 2, -1])
        vernierlabels = tf.squeeze(vernierlabels)
        T_vernierlabels = tf.one_hot(tf.cast(vernierlabels, tf.int64), depth, name='T_vernierlabels')
        
        if parameters.batch_norm_vernieroffset:
            hidden_vernieroffset = tf.layers.dense(shape_1_caps_activation, depth, use_bias=False, activation=None, name='hidden_vernieroffset')
            hidden_vernieroffset = tf.layers.batch_normalization(hidden_vernieroffset, training=phase, name='hidden_vernieroffset_bn')
        else:
            hidden_vernieroffset = tf.layers.dense(shape_1_caps_activation, depth, activation=None, name='hidden_vernieroffset')
            
        logits_vernierlabels = tf.nn.relu(hidden_vernieroffset, name='logits_vernierlabels')
        xent_vernierlabels = tf.losses.softmax_cross_entropy(T_vernierlabels, logits_vernierlabels)
        
        pred_vernierlabels = tf.argmax(logits_vernierlabels, axis=1, name='pred_vernierlabels', output_type=tf.int64)
        correct_vernierlabels = tf.equal(vernierlabels, pred_vernierlabels, name='correct_vernierlabels')
        accuracy_vernierlabels = tf.reduce_mean(tf.cast(correct_vernierlabels, tf.float32), name='accuracy_vernierlabels')
        return pred_vernierlabels, xent_vernierlabels, accuracy_vernierlabels


################################
#     nshapeslabels loss:      #
################################
def compute_nshapes_loss(decoder_input, nshapeslabels, parameters, phase=True):
    with tf.name_scope('compute_nshapes_loss'):
        caps_activation = tf.squeeze(decoder_input)
        nshapeslabels = tf.squeeze(tf.cast(nshapeslabels, tf.int64))
        
        depth = len(parameters.n_shapes)  # or rather len/max?
        T_nshapes = tf.one_hot(tf.cast(nshapeslabels, tf.int64), depth, name='T_nshapes')
        
        if parameters.batch_norm_nshapes:
            hidden_nshapes = tf.layers.dense(caps_activation, depth, use_bias=False, activation=None, reuse=tf.AUTO_REUSE, name='hidden_nshapes')
            hidden_nshapes = tf.layers.batch_normalization(hidden_nshapes, training=phase, reuse=tf.AUTO_REUSE, name='hidden_nshapes_bn')
        else:
            hidden_nshapes = tf.layers.dense(caps_activation, depth, activation=None, reuse=tf.AUTO_REUSE, name='hidden_nshapes')

        logits_nshapes = tf.nn.relu(hidden_nshapes, name='logits_nshapes')
        pred_nshapes = tf.argmax(logits_nshapes, axis=1, name='predicted_nshapes', output_type=tf.int64)
        squared_diff_nshapes = tf.square(tf.cast(nshapeslabels, tf.float32) - tf.cast(pred_nshapes, tf.float32), name='squared_diff_nshapes')
        tf.summary.histogram('nshapes_real', nshapeslabels)
        tf.summary.histogram('nshapes_pred', pred_nshapes)
        tf.summary.histogram('nshapes_distance', tf.sqrt(squared_diff_nshapes))
        correct_nshapes = tf.equal(nshapeslabels, pred_nshapes, name='correct_nshapes')
        accuracy_nshapes = tf.reduce_mean(tf.cast(correct_nshapes, tf.float32), name='accuracy_nshapes')
        
        if parameters.nshapes_loss == 'xentropy':
            loss_nshapes = tf.losses.softmax_cross_entropy(T_nshapes, logits_nshapes)
        elif parameters.nshapes_loss == 'squared_diff':
            loss_nshapes = tf.reduce_sum(squared_diff_nshapes, name='squared_diff_loss_nshapes')
        return loss_nshapes, accuracy_nshapes


################################
#       Location loss:         #
################################
def compute_location_loss(decoder_input, x_label, y_label, parameters, name_extra=None, phase=True):
    with tf.name_scope('compute_' + name_extra + '_location_loss'):
        caps_activation = tf.squeeze(decoder_input)
        
        # Loss for x-coordinates:
        x_label = tf.squeeze(x_label)
        x_depth = parameters.im_size[1]-parameters.shape_size
        T_x = tf.one_hot(tf.cast(x_label, tf.int64), x_depth, dtype=tf.float32, name='T_x_'+name_extra)
        
        if parameters.batch_norm_location:
            hidden_x = tf.layers.dense(caps_activation, x_depth, use_bias=False, activation=None, name='hidden_x'+name_extra)
            hidden_x = tf.layers.batch_normalization(hidden_x, training=phase, name='hidden_x_bn'+name_extra)
        else:
            hidden_x = tf.layers.dense(caps_activation, x_depth, activation=None, name='hidden_x'+name_extra)
        
        x_logits = tf.nn.relu(hidden_x, name='x_logits'+name_extra)
        pred_x = tf.argmax(x_logits, axis=1, name='pred_x'+name_extra, output_type=tf.int64)
        x_squared_diff = tf.square(tf.cast(x_label, tf.float32) - tf.cast(pred_x, tf.float32), name='x_squared_difference_'+name_extra)
        tf.summary.histogram('x_real_'+name_extra, x_label)
        tf.summary.histogram('x_pred_'+name_extra, pred_x)
        tf.summary.histogram('x_distance'+name_extra, tf.sqrt(x_squared_diff))
        
        if parameters.location_loss == 'xentropy':
            x_loss = tf.losses.softmax_cross_entropy(T_x, x_logits)
        elif parameters.location_loss == 'squared_diff':
            x_loss = tf.reduce_sum(x_squared_diff, name='x_squared_difference_loss_'+name_extra)


        # Loss for y-coordinates:
        y_label = tf.squeeze(y_label)
        y_depth = parameters.im_size[0]-parameters.shape_size
        T_y = tf.one_hot(tf.cast(y_label, tf.int64), y_depth, dtype=tf.float32, name='T_y_'+name_extra)
        
        if parameters.batch_norm_location:
            hidden_y = tf.layers.dense(caps_activation, y_depth, use_bias=False, activation=None, name='hidden_y'+name_extra)
            hidden_y = tf.layers.batch_normalization(hidden_y, training=phase, name='hidden_y_bn'+name_extra)
        else:
            hidden_y = tf.layers.dense(caps_activation, y_depth, activation=None, name='hidden_y'+name_extra)
        
        y_logits = tf.nn.relu(hidden_y, name='y_logits'+name_extra)
        pred_y = tf.argmax(y_logits, axis=1, name='pred_y_'+name_extra, output_type=tf.int64)
        y_squared_diff = tf.square(tf.cast(y_label, tf.float32) - tf.cast(pred_y, tf.float32), name='y_squared_difference_'+name_extra)
        tf.summary.histogram('y_real_'+name_extra, y_label)
        tf.summary.histogram('y_pred_'+name_extra, pred_y)
        tf.summary.histogram('y_distance'+name_extra, tf.sqrt(y_squared_diff))
        
        if parameters.location_loss == 'xentropy':
            y_loss = tf.losses.softmax_cross_entropy(T_y, y_logits)
        elif parameters.location_loss == 'squared_diff':
            y_loss = tf.reduce_sum(y_squared_diff, name='y_squared_difference_loss_'+name_extra)

        return x_loss, y_loss

