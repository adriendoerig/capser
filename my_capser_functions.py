# -*- coding: utf-8 -*-
"""
My capsnet: capsule functions
Version 2
Including:
    squash, safe_norm, routing_by_agreement,
    compute_margin_loss, compute_reconstruction, compute_reconstruction_loss,
    conv_layers, primary_caps_layer, secondary_caps_layer, caps_prediction
Last update on 23.10.2018
@author: Lynn
"""

#import ipdb
import tensorflow as tf

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
def routing_by_agreement(caps2_predicted, batch_size_tensor, parameters):
    # How often we do the routing:
    def routing_condition(raw_weights, caps2_output, counter):
        output = tf.less(counter, parameters.iter_routing)
        return output
    
    # What the routing is:
    def routing_body(raw_weights, caps2_output, counter):
        routing_weights = tf.nn.softmax(raw_weights, axis=2, name='routing_weights')
        weighted_predictions = tf.multiply(routing_weights, caps2_predicted, name='weighted_predictions')
        weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keepdims=True, name='weighted_sum')
        caps2_output = squash(weighted_sum, axis=-2, name='caps2_output')
        caps2_output_tiled = tf.tile(caps2_output, [1, parameters.caps1_ncaps, 1, 1, 1], name='caps2_output_tiled')
        agreement = tf.matmul(caps2_predicted, caps2_output_tiled, transpose_a=True, name='agreement')
        raw_weights = tf.add(raw_weights, agreement, name='raw_weights')
        return raw_weights, caps2_output, tf.add(counter, 1)
    
    # Execution of routing via while-loop:
    with tf.name_scope('Routing_by_agreement'):
        # Initialize weights and caps2-output-array
        raw_weights = tf.zeros([batch_size_tensor, parameters.caps1_ncaps, parameters.caps2_ncaps, 1, 1],
                               dtype=tf.float32, name='raw_weights')
        caps2_output_init = tf.zeros([batch_size_tensor, 1, parameters.caps2_ncaps, parameters.caps2_ndims, 1],
                                     dtype=tf.float32, name='caps2_output_init')
        # Counter for number of routing iterations:
        counter = tf.constant(0)
        raw_weights, caps2_output, counter = tf.while_loop(routing_condition, routing_body,
                                                           [raw_weights, caps2_output_init, counter])
        return caps2_output


###################################
#     Create capser network:      #
###################################
def conv_layers(X, conv1_params, conv2_params):
    with tf.name_scope('convolutional_layers'):
        conv1 = tf.layers.conv2d(X, name="conv1", **conv1_params)
        conv2 = tf.layers.conv2d(conv1, name='conv2', **conv2_params)
        return conv2


def primary_caps_layer(conv_output, parameters):
    with tf.name_scope('primary_capsules'):
        caps1_reshaped = tf.reshape(conv_output, [-1, parameters.caps1_ncaps, parameters.caps1_ndims], name='caps1_reshaped')
        caps1_output = squash(caps1_reshaped, name='caps1_output')
        caps1_output_norm = safe_norm(caps1_output, axis=-1, keepdims=True, name='caps1_output_norm')
        return caps1_output, caps1_output_norm


def secondary_caps_layer(caps1_output, parameters):
    with tf.name_scope('secondary_caps_layer'):
        # Initialize and repeat weights for further calculations:
        W_init = lambda: tf.random_normal(
            shape=(1, parameters.caps1_ncaps, parameters.caps2_ncaps, parameters.caps2_ndims, parameters.caps1_ndims),
            stddev=parameters.init_sigma, dtype=tf.float32, name='W_init')
        W = tf.Variable(W_init, dtype=tf.float32, name="W")
        W_tiled = tf.tile(W, [parameters.batch_size, 1, 1, 1, 1], name="W_tiled")

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
        return caps2_output, caps2_output_norm


def predict_shapelabels(caps2_output):
    with tf.name_scope('predict_shape'):
        # Calculate caps norm:
        labels_proba = safe_norm(caps2_output, axis=-2, name='labels_proba')
    
        # Prediction:
        labels_proba_argmax = tf.argmax(labels_proba, axis=2, name='labels_proba_argmax')
        labels_pred = tf.squeeze(labels_proba_argmax, axis=[1, 2], name='labels_pred')
        return labels_pred


################################
#         Margin loss:         #
################################
def compute_margin_loss(caps2_output_norm, labels, parameters):
    with tf.name_scope('compute_margin_loss'):
        # Compute the loss for each instance and digit:
        T_shapelabels = tf.one_hot(labels, depth=parameters.caps2_ncaps, name='T_shapelabels')
        present_error_raw = tf.square(tf.maximum(0., parameters.m_plus - caps2_output_norm), name='present_error_raw')
        present_error = tf.reshape(present_error_raw, shape=(parameters.batch_size, parameters.caps2_ncaps), name='present_error')
        absent_error_raw = tf.square(tf.maximum(0., caps2_output_norm - parameters.m_minus), name='absent_error_raw')
        absent_error = tf.reshape(absent_error_raw, shape=(parameters.batch_size, parameters.caps2_ncaps), name='absent_error')
        L = tf.add(T_shapelabels * present_error, parameters.lambda_val * (1.0 - T_shapelabels) * absent_error, name='L')
        
        # Sum digit losses for each instance and compute mean:
        margin_loss = tf.reduce_mean(tf.reduce_sum(L, axis=1), name='margin_loss')
        return margin_loss


################################
#          Accuracy:           #
################################
def compute_accuracy(labels, labels_pred):
    with tf.name_scope('accuracy'):
        correct = tf.equal(labels, labels_pred, name='correct')
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')
        return accuracy


################################
#       Reconstruction:        #
################################
def compute_reconstruction(mask_with_labels, labels, labels_pred, caps2_output, parameters):
    with tf.name_scope('compute_decoder_input'):
        reconstruction_targets = tf.cond(mask_with_labels, lambda: labels, lambda: labels_pred, name='reconstruction_targets')
        reconstruction_mask = tf.one_hot(reconstruction_targets, depth=parameters.caps2_ncaps, name='reconstruction_mask')
        reconstruction_mask = tf.reshape(reconstruction_mask, [-1, 1, parameters.caps2_ncaps, 1, 1], name='reconstruction_mask')
        caps2_output_masked = tf.multiply(caps2_output, reconstruction_mask, name='caps2_output_masked')
        
        # Flatten decoder inputs:
        decoder_input = tf.reshape(caps2_output_masked, [-1, parameters.caps2_ndims*parameters.caps2_ncaps], name='decoder_input')

    # Finally comes the decoder (two dense fully connected ReLU layers followed by a dense output sigmoid layer):
    with tf.name_scope('decoder'):
        hidden1 = tf.layers.dense(decoder_input, parameters.n_hidden1, activation=tf.nn.relu, name='hidden1')
        hidden2 = tf.layers.dense(hidden1, parameters.n_hidden2, activation=tf.nn.relu, name='hidden2')
        decoder_output = tf.layers.dense(hidden2, parameters.n_output, activation=tf.nn.sigmoid, name='decoder_output')
        decoder_output_img = tf.reshape(decoder_output, [-1, parameters.im_size[0], parameters.im_size[1], parameters.im_depth],
                                        name='decoder_output_img')
        return decoder_output, decoder_output_img


################################
#     Reconstruction loss:     #
################################
def compute_reconstruction_loss(imgs, decoder_output, parameters):
    with tf.name_scope('compute_reconstruction_loss'):
        imgs_flat = tf.reshape(imgs, [-1, parameters.n_output], name='imgs_flat')
        squared_difference = tf.square(imgs_flat - decoder_output, name='squared_difference')
        reconstruction_loss = tf.reduce_mean(squared_difference, name='reconstruction_loss')
        return reconstruction_loss


################################
#     vernieroffset loss:      #
################################
def compute_vernieroffset_loss(vernier_caps_activation, vernierlabels, depth=2):
    with tf.name_scope('compute_vernieroffset_loss'):
        vernier_caps_activation = tf.squeeze(vernier_caps_activation)
        vernierlabels = tf.squeeze(vernierlabels)
        
        T_vernierlabels = tf.one_hot(vernierlabels, depth, name='T_vernierlabels')
        logits_vernierlabels = tf.layers.dense(vernier_caps_activation, depth, tf.nn.relu, name='logits_vernierlabels')
        xent_vernierlabels = tf.losses.softmax_cross_entropy(T_vernierlabels, logits_vernierlabels)
        
        pred_vernierlabels = tf.argmax(logits_vernierlabels, axis=1)
        correct_vernierlabels = tf.equal(vernierlabels, pred_vernierlabels, name='correct_vernierlabels')
        accuracy_vernierlabels = tf.reduce_mean(tf.cast(correct_vernierlabels, tf.float32), name='accuracy_vernierlabels')
        return pred_vernierlabels, xent_vernierlabels, accuracy_vernierlabels

