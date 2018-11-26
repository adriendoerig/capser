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

Last update on 26.11.2018
-> added nshapes and location loss
-> network can be run with 2 or 3 conv layers now
-> you can choose now between xentropy of squared_diff as location or nshapes loss
"""

import tensorflow as tf

################################
#    Small helper function:    #
################################
def save_params(parameters):
    with open(parameters.logdir + '/parameters.txt', 'w') as f_txt:
        f_py = open('./my_parameters.py')
        variables = f_py.read()
        f_txt.write(variables)
        print('Parameter values saved.')


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
        caps2_output = tf.zeros([batch_size_tensor, 1, parameters.caps2_ncaps, parameters.caps2_ndims, 1],
                                dtype=tf.float32, name='caps2_output_init')
        # Counter for number of routing iterations:
        counter = tf.constant(1)
        raw_weights, caps2_output, counter = tf.while_loop(routing_condition, routing_body,
                                                           [raw_weights, caps2_output, counter])
        return caps2_output


###################################
#     Create capser network:      #
###################################
def conv_layers(X, parameters, phase=True):
    with tf.name_scope('1_convolutional_layers'):
        if parameters.n_conv_layers==2:
            conv1 = tf.layers.conv2d(X, name='conv1', activation=None, **parameters.conv_params[0])
            if parameters.batch_norm_conv:
                conv1 = tf.layers.batch_normalization(conv1, training=phase, name='conv1_bn')
            conv1 = tf.nn.relu(conv1)
            tf.summary.histogram('conv1_output', conv1)
    
            conv2 = tf.layers.conv2d(conv1, name='conv2', activation=None, **parameters.conv_params[1])
            conv_output = tf.nn.relu(conv2)
            tf.summary.histogram('conv2_output', conv_output)
            tf.summary.histogram('conv3_output', 0)

        elif parameters.n_conv_layers==3:
            conv1 = tf.layers.conv2d(X, name='conv1', activation=None, **parameters.conv_params[0])
            if parameters.batch_norm_conv:
                conv1 = tf.layers.batch_normalization(conv1, training=phase, name='conv1_bn')
            conv1 = tf.nn.relu(conv1)
            tf.summary.histogram('conv1_output', conv1)
    
            conv2 = tf.layers.conv2d(conv1, name='conv2', activation=None, **parameters.conv_params[1])
            if parameters.batch_norm_conv:
                conv2 = tf.layers.batch_normalization(conv2, training=phase, name='conv2_bn')
            conv2 = tf.nn.relu(conv2)
            tf.summary.histogram('conv2_output', conv2)
    
            conv3 = tf.layers.conv2d(conv2, name='conv3', activation=None, **parameters.conv_params[2])
            conv_output = tf.nn.relu(conv3)
            tf.summary.histogram('conv3_output', conv_output)
        return conv_output


def primary_caps_layer(conv_output, parameters):
    with tf.name_scope('2_primary_capsules'):
        caps1_reshaped = tf.reshape(conv_output, [parameters.batch_size, parameters.caps1_ncaps, parameters.caps1_ndims], name='caps1_reshaped')
        caps1_output = squash(caps1_reshaped, name='caps1_output')
        caps1_output_norm = safe_norm(caps1_output, axis=-1, keepdims=False, name='caps1_output_norm')
        tf.summary.histogram('caps1_output_norm', caps1_output_norm)
        return caps1_output


def secondary_caps_layer(caps1_output, parameters):
    with tf.name_scope('3_secondary_caps_layer'):
        # Initialize and repeat weights for further calculations:
        if parameters.random_seed:
            W_init = lambda: tf.random_normal(
                shape=(1, parameters.caps1_ncaps, parameters.caps2_ncaps, parameters.caps2_ndims, parameters.caps1_ndims),
                stddev=parameters.init_sigma, dtype=tf.float32, seed=1, name='W_init')
        else:
            W_init = lambda: tf.random_normal(
                shape=(
                1, parameters.caps1_ncaps, parameters.caps2_ncaps, parameters.caps2_ndims, parameters.caps1_ndims),
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
        tf.summary.histogram('caps2_output_norm', caps2_output_norm[0, :, :, :])
        return caps2_output, caps2_output_norm


################################
#         Margin loss:         #
################################
def predict_shapelabels(caps2_output, n_labels):
    with tf.name_scope('predict_shape'):
        # Calculate caps norm:
        labels_proba = safe_norm(caps2_output, axis=-2, name='labels_proba')

        # Predict n_labels largest values:
        _, labels_pred = tf.nn.top_k(labels_proba[:, 0, :, 0], n_labels, name="y_proba")
        labels_pred = tf.cast(labels_pred, tf.int64)
        labels_pred = tf.contrib.framework.sort(labels_pred, axis=-1, direction='ASCENDING', name='labels_pred_sorted')
        return labels_pred


def compute_margin_loss(caps2_output_norm, labels, parameters):
    with tf.name_scope('compute_margin_loss'):
        # Compute the loss for each instance and shape:
        # trick to get a vector for each image in the batch. Labels [0,2] -> [[1, 0, 1]] and [1,1] -> [[0, 1, 0]]
        T_shapelabels_raw = tf.one_hot(labels, depth=parameters.caps2_ncaps, name='T_shapelabels_raw')
        T_shapelabels = tf.reduce_sum(T_shapelabels_raw, axis=1)
        T_shapelabels = tf.minimum(T_shapelabels, 1)
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
        current_caps2_ncaps = caps2_output.get_shape().as_list()[2]
        reconstruction_targets = tf.cond(mask_with_labels, lambda: labels, lambda: labels_pred, name='reconstruction_targets')
        reconstruction_mask = tf.one_hot(reconstruction_targets, depth=current_caps2_ncaps, name='reconstruction_mask')
        reconstruction_mask = tf.reshape(reconstruction_mask, [parameters.batch_size, 1, current_caps2_ncaps, 1, 1], name='reconstruction_mask')
        caps2_output_masked = tf.multiply(caps2_output, reconstruction_mask, name='caps2_output_masked')
        
        # Flatten decoder inputs:
        decoder_input = tf.reshape(caps2_output_masked, [parameters.batch_size, parameters.caps2_ndims*current_caps2_ncaps], name='decoder_input')
        return decoder_input


################################
#    Create reconstruction:    #
################################
def compute_reconstruction(decoder_input,  parameters, phase=True, name_extra=None):
    # Finally comes the decoder (two dense fully connected ELU layers followed by a dense output sigmoid layer):
    with tf.name_scope('decoder_reconstruction'):
        hidden1 = tf.layers.dense(decoder_input, parameters.n_hidden1, use_bias=False, activation=None, name='hidden1' + name_extra)
        if parameters.batch_norm_decoder:
            hidden1 = tf.layers.batch_normalization(hidden1, training=phase, name='hidden1_bn' + name_extra)
        hidden1 = tf.nn.elu(hidden1, name='hidden1_activation')
        tf.summary.histogram('_hidden1_bn' + name_extra, hidden1)

        hidden2 = tf.layers.dense(hidden1, parameters.n_hidden2, use_bias=False, activation=None, name='hidden2' + name_extra)
        if parameters.batch_norm_decoder:
            hidden2 = tf.layers.batch_normalization(hidden2, training=phase, name='hidden2_bn' + name_extra)
        hidden2 = tf.nn.elu(hidden2, name='hidden2_activation')
        tf.summary.histogram('_hidden2_bn' + name_extra, hidden2)

        reconstructed_output = tf.layers.dense(hidden2, parameters.n_output, activation=tf.nn.sigmoid, name='reconstructed_output' + name_extra)
        reconstructed_output_img = tf.reshape(
                reconstructed_output,[parameters.batch_size, parameters.im_size[0], parameters.im_size[1], parameters.im_depth],
                name='reconstructed_output_img')
        return reconstructed_output, reconstructed_output_img


################################
#     Reconstruction loss:     #
################################
def compute_reconstruction_loss(X, reconstructed_output, parameters):
    with tf.name_scope('compute_reconstruction_loss'):
        imgs_flat = tf.reshape(X, [parameters.batch_size, parameters.n_output], name='imgs_flat')
        imgs_flat = tf.cast(imgs_flat, tf.float32)
        squared_difference = tf.square(imgs_flat - reconstructed_output, name='squared_difference')
        # reconstruction_loss = tf.reduce_mean(squared_difference, name='reconstruction_loss')
        reconstruction_loss = tf.reduce_sum(squared_difference, name='reconstruction_loss')
        return reconstruction_loss


################################
#     Vernieroffset loss:      #
################################
def compute_vernieroffset_loss(vernier_caps_activation, vernierlabels, depth=2):
    with tf.name_scope('compute_vernieroffset_loss'):
        vernier_caps_activation = tf.squeeze(vernier_caps_activation)
        vernierlabels = tf.squeeze(vernierlabels)
        vernierlabels = tf.cast(vernierlabels, tf.float32)
        
        T_vernierlabels = tf.one_hot(tf.cast(vernierlabels, tf.int32), depth, name='T_vernierlabels')
        logits_vernierlabels = tf.layers.dense(vernier_caps_activation, depth, tf.nn.relu, name='logits_vernierlabels')
        xent_vernierlabels = tf.losses.softmax_cross_entropy(T_vernierlabels, logits_vernierlabels)
        
        pred_vernierlabels = tf.argmax(logits_vernierlabels, axis=1)
        correct_vernierlabels = tf.equal(vernierlabels, tf.cast(pred_vernierlabels, tf.float32), name='correct_vernierlabels')
        accuracy_vernierlabels = tf.reduce_mean(tf.cast(correct_vernierlabels, tf.float32), name='accuracy_vernierlabels')
        return pred_vernierlabels, xent_vernierlabels, accuracy_vernierlabels


################################
#     nshapeslabels loss:      #
################################
def compute_nshapes_loss(shape_decoder_input, nshapeslabels, parameters):
    with tf.name_scope('compute_nshapes_loss'):
        depth = len(parameters.n_shapes)  # or rather len/max?
        shape_caps_activation = tf.squeeze(shape_decoder_input)
        nshapeslabels = tf.squeeze(nshapeslabels)
        nshapeslabels = tf.cast(nshapeslabels, tf.float32)

        T_nshapes = tf.one_hot(tf.cast(nshapeslabels, tf.int32), depth, name='T_nshapes')
        logits_nshapes = tf.layers.dense(shape_caps_activation, depth, tf.nn.relu, name='logits_nshapes')
        
        if parameters.nshapes_loss == 'xentropy':
            loss_nshapes = tf.losses.softmax_cross_entropy(T_nshapes, logits_nshapes)
        elif parameters.nshapes_loss == 'squared_diff':
            squared_diff_nshapes = tf.square(T_nshapes - logits_nshapes, name='squared_diff_nshapes')
            loss_nshapes = tf.reduce_sum(squared_diff_nshapes, name='squared_diff_loss_nshapes')

        pred_nshapes = tf.argmax(logits_nshapes, axis=1)
        tf.summary.histogram('pred_nshapes', pred_nshapes)
        correct_nshapes = tf.equal(nshapeslabels, tf.cast(pred_nshapes, tf.float32), name='correct_nshapes')
        accuracy_nshapes = tf.reduce_mean(tf.cast(correct_nshapes, tf.float32), name='accuracy_nshapes')
        return loss_nshapes, accuracy_nshapes


################################
#     location loss:      #
################################
def compute_location_loss(decoder_input, x_label, y_label, parameters, name_extra=None):
    with tf.name_scope('compute_' + name_extra + '_location_loss'):
        # Loss for x-coordinates:
        x_depth = parameters.im_size[1]-parameters.shape_size
        caps_activation = tf.squeeze(decoder_input)
        x_label = tf.squeeze(x_label)
        x_label = tf.cast(x_label, tf.float32)
        T_x = tf.one_hot(tf.cast(x_label, tf.int32), x_depth, name='T_x'+name_extra)
        x_logits = tf.layers.dense(caps_activation, x_depth, tf.nn.relu, name='logits_x'+name_extra)
        
        if parameters.location_loss == 'xentropy':
            x_loss = tf.losses.softmax_cross_entropy(T_x, x_logits)
        elif parameters.location_loss == 'squared_diff':
            x_squared_diff = tf.square(T_x - x_logits, name='x_squared_difference')
            x_loss = tf.reduce_sum(x_squared_diff, name='x_squared_difference_loss')
        pred_x = tf.argmax(x_logits, axis=1)
        tf.summary.histogram('pred_x_'+name_extra, pred_x)
        
        # Loss for y-coordinates:
        y_depth = parameters.im_size[0]-parameters.shape_size
        y_label = tf.squeeze(y_label)
        y_label = tf.cast(y_label, tf.float32)
        T_y = tf.one_hot(tf.cast(y_label, tf.int32), y_depth, name='T_y'+name_extra)
        y_logits = tf.layers.dense(caps_activation, y_depth, tf.nn.relu, name='logits_y'+name_extra)
        
        if parameters.location_loss == 'xentropy':
            y_loss = tf.losses.softmax_cross_entropy(T_y, y_logits)
        elif parameters.location_loss == 'squared_diff':
            y_squared_diff = tf.square(T_y - y_logits, name='y_squared_difference')
            y_loss = tf.reduce_sum(y_squared_diff, name='y_squared_difference_loss')
        
        pred_y = tf.argmax(y_logits, axis=1)
        tf.summary.histogram('pred_y_'+name_extra, pred_y)
        return x_loss, y_loss
