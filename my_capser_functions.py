# -*- coding: utf-8 -*-
"""
My capsnet: capsule functions
Version 1
Created on Thu Oct  4 10:37:15 2018
@author: Lynn
"""

import ipdb
import tensorflow as tf

################################
#      Squash function:        #
################################
def squash(s, axis=-1, epsilon=1e-7, name=None):
    '''Squashing function as described in Sabour et al. (2018) but calculating
    the norm in a safe way (e.g. see Aurelion Geron youtube-video)
    
    Inputs:
        s: matrix;
        axis: axis over which the squash should be computed (default: -1);
        epsilon (default 1e-7);
        name (default squash);
    Output:
        s_squashed'''
    with tf.name_scope(name, default_name='squash'):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keepdims=True)
        safe_norm = tf.sqrt(squared_norm + epsilon)
        squash_factor = squared_norm / (1. + squared_norm)
        unit_vector = s / safe_norm
        s_squashed = squash_factor * unit_vector
        return s_squashed


#####################################
#   Estimated class probabilities   #
#####################################
def safe_norm(s, axis=-1, epsilon=1e-7, keepdims=False, name=None):
    '''Safe calculation of the norm for estimated class probabilities
    
    Inputs:
        s: matrix;
        axis: axis over which the norm should be computed (default: -1);
        epsilon (default: 1e-7);
        keepdims (default: False)
        name (default: squash);
    Output:
        s_norm'''
    with tf.name_scope(name, default_name='safe_norm'):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keepdims=keepdims)
        s_norm = tf.sqrt(squared_norm + epsilon)
        return s_norm


################################
#     Routing by agreement:    #
################################
def routing_by_agreement(caps2_predicted, batch_size_tensor, params):
    # How often we do the routing:
    def routing_condition(raw_weights, caps2_output, counter):
        output = tf.less(counter, params.iter_routing)
        return output
    
    # What the routing is:
    def routing_body(raw_weights, caps2_output, counter):
        routing_weights = tf.nn.softmax(raw_weights, axis=2, name='routing_weights')
        weighted_predictions = tf.multiply(routing_weights, caps2_predicted, name='weighted_predictions')
        weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keepdims=True, name='weighted_sum')
        caps2_output = squash(weighted_sum, axis=-2, name='caps2_output')
        caps2_output_tiled = tf.tile(caps2_output, [1, params.caps1_ncaps, 1, 1, 1], name='caps2_output_tiled')
        agreement = tf.matmul(caps2_predicted, caps2_output_tiled, transpose_a=True, name='agreement')
        raw_weights = tf.add(raw_weights, agreement, name='raw_weights')
        return raw_weights, caps2_output, tf.add(counter, 1)
    
    # Execution of routing via while-loop:
    with tf.name_scope('Routing_by_agreement'):
        # Initialize weights and caps2-output-array
        raw_weights = tf.zeros([batch_size_tensor, params.caps1_ncaps, params.caps2_ncaps, 1, 1],
                               dtype=tf.float32, name='raw_weights')
        caps2_output_init = tf.zeros([batch_size_tensor, 1, params.caps2_ncaps, params.caps2_ndims, 1],
                                     dtype=tf.float32, name='caps2_output_init')
        # Counter for number of routing iterations:
        counter = tf.constant(0)
        raw_weights, caps2_output, counter = tf.while_loop(routing_condition, routing_body,
                                                           [raw_weights, caps2_output_init, counter])
    return caps2_output


################################
#         Margin loss:         #
################################
def compute_margin_loss(caps2_output_norm, labels, params):
    # Compute the loss for each instance and digit:
    if labels.get_shape().as_list()[1] == 1:
        tf.squeeze(labels)
        T = tf.one_hot(labels, depth=params.caps2_ncaps, name="T")
        print('labels shape equal to 1')
    else:
        T_raw = tf.one_hot(labels, depth=params.caps2_ncaps)
        T = tf.reduce_sum(T_raw, axis=1)
        T = tf.minimum(T, 1)
        print('labels shape bigger than 1')
#    T = tf.one_hot(labels, depth=params.caps2_ncaps, name="T")
    present_error_raw = tf.square(tf.maximum(0., params.m_plus - caps2_output_norm), name='present_error_raw')
    present_error = tf.reshape(present_error_raw, shape=(params.batch_size, params.caps2_ncaps), name='present_error')
    absent_error_raw = tf.square(tf.maximum(0., caps2_output_norm - params.m_minus), name='absent_error_raw')
    absent_error = tf.reshape(absent_error_raw, shape=(params.batch_size, params.caps2_ncaps), name='absent_error')
    L = tf.add(T * present_error, params.lambda_val * (1.0 - T) * absent_error, name='L')
    
    # Sum digit losses for each instance and compute mean:
    margin_loss = tf.reduce_mean(tf.reduce_sum(L, axis=1), name='margin_loss')
#    ipdb.set_trace()
    return margin_loss


################################
#       Reconstruction:        #
################################
def compute_reconstruction(mask_with_labels, y, y_pred, caps2_output, params):
    reconstruction_targets = tf.cond(mask_with_labels, lambda: y, lambda: y_pred, name='reconstruction_targets')
    reconstruction_mask = tf.one_hot(reconstruction_targets, depth=params.caps2_ncaps, name='reconstruction_mask')
    reconstruction_mask = tf.reshape(reconstruction_mask, [-1, 1, params.caps2_ncaps, 1, 1], name='reconstruction_mask')
    caps2_output_masked = tf.multiply(caps2_output, reconstruction_mask, name='caps2_output_masked')
    
    # Flatten decoder inputs:
    decoder_input = tf.reshape(caps2_output_masked, [-1, params.caps2_ndims*params.caps2_ncaps], name='decoder_input')
    
    # Finally comes the decoder (two dense fully connected ReLU layers followed by a dense output sigmoid layer):
    with tf.name_scope('decoder'):
        hidden1 = tf.layers.dense(decoder_input, params.n_hidden1, activation=tf.nn.relu, name='hidden1')
        hidden2 = tf.layers.dense(hidden1, params.n_hidden2, activation=tf.nn.relu, name='hidden2')
        decoder_output = tf.layers.dense(hidden2, params.n_output, activation=tf.nn.sigmoid, name='decoder_output')
        return decoder_output


################################
#     Reconstruction loss:     #
################################
def compute_reconstruction_loss(imgs, decoder_output, params):
    imgs_flat = tf.reshape(imgs, [-1, params.n_output], name='imgs_flat')
    squared_difference = tf.square(imgs_flat - decoder_output, name='squared_difference')
    reconstruction_loss = tf.reduce_mean(squared_difference, name='reconstruction_loss')
    return reconstruction_loss


def primary_caps_layer(conv_output, params):
    with tf.name_scope('primary_capsules'):
        # reshape the output to be caps1_n_dims-Dim capsules (since the next layer is FC, we don't need to
        # keep the [batch,xx,xx,n_feature_maps,caps1_n_dims] so we just flatten it to keep it simple)
        caps1_reshaped = tf.reshape(conv_output, [-1, params.caps1_ncaps, params.caps1_ndims], name="caps1_reshaped")
        caps1_output = squash(caps1_reshaped, name="caps1_output")
        return caps1_output



def primary_to_fc_caps_layer(input_batch, caps1_output, caps1_n_caps, caps1_n_dims, caps2_n_caps, caps2_n_dims,
                             rba_rounds=3, print_shapes=False):
    # Now the tricky part: create the predictions of secondary capsules' activities!
    # in essence, this is creating random weights from each dimension of each layer 1 capsule
    # to each dimension of each layer 2 capsule -- initializing random transforms on caps1 output vectors.
    # To make it efficient we use only tensorflow-friendly matrix multiplications. To this end,
    # we use tf.matmul, which performs element wise matrix in multidimensional arrays. To create
    # these arrays we use tf.tile a lot. See ageron github & video for more explanations.

    with tf.name_scope('primary_to_first_fc'):
        # initialise weights
        init_sigma = 0.01  # stdev of weights
        W_init = lambda: tf.random_normal(
            shape=(1, caps1_n_caps, caps2_n_caps, caps2_n_dims, caps1_n_dims),
            stddev=init_sigma, dtype=tf.float32, name="W_init")
        W = tf.Variable(W_init, dtype=tf.float32, name="W")

        # tile weights to [batch_size, caps1_n_caps, caps2_n_caps, caps2_n_dims, caps1_n_dims]
        # i.e. batch_size times a caps2_n_dims*caps1_n_dims array of [caps1_n_caps*caps2_n_caps] weight matrices
        # batch_size = tf.shape(input_batch)[0]  # note: tf.shape(X) is undefined until we fill the placeholder
        W_tiled = tf.tile(W, [batch_size, 1, 1, 1, 1], name="W_tiled")

        # tile caps1_output to [batch_size, caps1_n_caps, caps2_n_caps, caps2_n_dims, caps1_n_dims]
        # to do so, first we need to add the required dimensions with tf.expand_dims
        caps1_output_expanded = tf.expand_dims(caps1_output, -1,
                                               name="caps1_output_expanded")  # expand last dimension
        caps1_output_tile = tf.expand_dims(caps1_output_expanded, 2,
                                           name="caps1_output_tile")  # expand third dimension
        caps1_output_tiled = tf.tile(caps1_output_tile, [1, 1, caps2_n_caps, 1, 1],
                                     name="caps1_output_tiled")  # tile
        # check shapes
        if print_shapes:
            print('shape of tiled W: ' + str(W_tiled))
            print('shape of tiled caps1_output: ' + str(caps1_output_tiled))

        # Thanks to all this hard work, computing the secondary capsules' predicted activities is easy peasy:
        caps2_predicted = tf.matmul(W_tiled, caps1_output_tiled,
                                    name="caps2_predicted")

        # tf.summary.histogram('rba_0', caps2_predicted)

        # check shape
        if print_shapes:
            print('shape of caps2_predicted: ' + str(caps2_predicted))

        ################################################################################################################
        # ROUTING BY AGREEMENT iterative algorithm
        ################################################################################################################

        with tf.name_scope('routing_by_agreement'):

            def do_routing_cond(caps2_predicted, caps2_output, raw_weights, rba_iter, max_iter=rba_rounds):
                return tf.less(rba_iter, max_iter+1, name='do_routing_cond')

            def routing_by_agreement(caps2_predicted, caps2_output, raw_weights, rba_iter, max_iter=rba_rounds):

                # Round 1 of RbA

                # softmax on weights
                routing_weights = tf.nn.softmax(raw_weights, dim=2, name="routing_weights")

                # weighted sum of the lower layer predictions according to the routing weights
                weighted_predictions = tf.multiply(routing_weights, caps2_predicted,
                                                   name="weighted_predictions")
                weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keep_dims=True,
                                             name="weighted_sum")

                # squash
                caps2_rba_output = squash(weighted_sum, axis=-2,
                                          name="caps2_rba_output")

                # check shape
                if print_shapes:
                    print('shape of caps2_output after after RbA round: ' + str(caps2_rba_output))

                # to measure agreement, we just compute dot products between predictions and actual activations.
                # to do so, we will again use tf.matmul and tiling
                caps2_rba_output_tiled = tf.tile(
                    caps2_rba_output, [1, caps1_n_caps, 1, 1, 1],
                    name="caps2_rba_output_tiled")

                # check shape
                if print_shapes:
                    print('shape of TILED caps2_output after RbA round: ' + str(caps2_rba_output_tiled))

                # comput agreement is simple now
                agreement = tf.matmul(caps2_predicted, caps2_rba_output_tiled,
                                      transpose_a=True, name="agreement")

                # update routing weights based on agreement
                raw_weights_new = tf.add(raw_weights, agreement,
                                         name="raw_weights_round_new")

                return caps2_predicted, caps2_rba_output, raw_weights_new, tf.add(rba_iter, 1)

            # initialize routing weights
            raw_weights = tf.zeros([batch_size, caps1_n_caps, caps2_n_caps, 1, 1],
                                   dtype=np.float32, name="raw_weights")

            rba_iter = tf.constant(1, name='rba_iteration_counter')
            caps2_output = tf.zeros(shape=(batch_size, 1, caps2_n_caps, caps2_n_dims, 1), name='caps2_output')
            caps2_predicted, caps2_output, raw_weights, rba_iter = tf.while_loop(do_routing_cond, routing_by_agreement,
                                                                                 [caps2_predicted, caps2_output,
                                                                                  raw_weights, rba_iter])

        # This is the caps2 output!
        # tf.summary.histogram('rba_output', caps2_output)

        if print_shapes:
            print('shape of caps2_output after RbA termination: ' + str(caps2_output))

        return caps2_output


def caps_prediction(caps2_output, n_labels=1, print_shapes=False):
    with tf.name_scope('net_prediction'):

        y_proba = safe_norm(caps2_output, axis=-2, name="y_proba")

        if n_labels == 1:  # there is a single shape to classify
            y_proba_argmax = tf.argmax(y_proba, axis=2, name="y_proba")
            # get predicted class by squeezing out irrelevant dimensions
            y_pred = tf.squeeze(y_proba_argmax, axis=[1, 2], name="y_pred")
        else:  # there are more than one shape to classify
            _, y_pred = tf.nn.top_k(y_proba[:, 0, :, 0], 2, name="y_proba")
            y_pred = tf.cast(y_pred, tf.int64)  # need to cast for type compliance later)

        if print_shapes:
            # check shapes
            print('shape of prediction: '+str(y_pred))

        return y_pred