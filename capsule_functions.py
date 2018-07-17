import tensorflow as tf
import numpy as np
from parameters import *

# define a safe-norm to avoid infinities and zeros
def safe_norm(s, axis=-1, epsilon=1e-7, keep_dims=False, name=None):
    with tf.name_scope(name, default_name="safe_norm"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keep_dims=keep_dims)
        return tf.sqrt(squared_norm + epsilon)


# define the squash function (to apply to capsule vectors)
# a safe-norm is implemented to avoid 0 norms because they
# would fuck up the gradients etc.
def squash(s, axis=-1, epsilon=1e-7, name=None):
    with tf.name_scope(name, default_name="squash"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keep_dims=True)
        safe_norm_squash = tf.sqrt(squared_norm + epsilon)
        squash_factor = squared_norm / (1. + squared_norm)
        unit_vector = s / safe_norm_squash
        return squash_factor * unit_vector


# takes the first regular convolutional layers' output as input and creates the first capsules
# returns the flattened output of the primary capsule layer (only works to feed to a FC caps layer)
def primary_caps_layer(conv_output, caps1_n_maps, caps1_n_caps, caps1_n_dims,
                       conv_kernel_size, conv_strides, conv_padding='valid',
                       conv_activation=tf.nn.relu, print_shapes=False):
    with tf.name_scope('primary_capsules'):
        conv_params = {
            "filters": caps1_n_maps * caps1_n_dims,  # n convolutional filters
            "kernel_size": conv_kernel_size,
            "strides": conv_strides,
            "padding": conv_padding,
            "activation": conv_activation
        }

        # we will reshape this to create the capsules
        conv_for_caps = tf.layers.conv2d(conv_output, name="conv_for_caps", **conv_params)
        if print_shapes:
            print('shape of conv_for_caps: '+str(conv_for_caps))

        # in case we want to force the network to use a certain primary capsule map to represent certain shapes, we must
        # conserve all the map dimensions (we don't care about spatial position)
        caps_per_map = tf.cast(caps1_n_caps / caps1_n_maps, dtype=tf.int32, name='caps_per_map')
        caps1_raw_with_maps = tf.reshape(conv_for_caps, [batch_size_per_shard, caps1_n_maps, caps_per_map, caps1_n_dims], name="caps1_raw_with_maps")

        # reshape the output to be caps1_n_dims-Dim capsules (since the next layer is FC, we don't need to
        # keep the [batch,xx,xx,n_feature_maps,caps1_n_dims] so we just flatten it to keep it simple)
        caps1_raw = tf.reshape(conv_for_caps, [batch_size_per_shard, caps1_n_caps, caps1_n_dims], name="caps1_raw")
        # tf.summary.histogram('caps1_raw', caps1_raw)

        # squash capsule outputs
        caps1_output = squash(caps1_raw, name="caps1_output")
        caps1_output_with_maps = squash(caps1_raw_with_maps, name="caps1_output_with_maps")
        # tf.summary.histogram('caps1_output', caps1_output)
        if print_shapes:
            print('shape of caps1_output: '+str(caps1_output))
            print('shape of caps1_output_with_maps: ' + str(caps1_output_with_maps))

        return caps1_output, caps1_output_with_maps


# takes a (flattened) primary capsule layer caps1 output as input and creates a new fully connected capsule layer caps2
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

        # tile weights to [batch_size_per_shard, caps1_n_caps, caps2_n_caps, caps2_n_dims, caps1_n_dims]
        # i.e. batch_size_per_shard times a caps2_n_dims*caps1_n_dims array of [caps1_n_caps*caps2_n_caps] weight matrices
        # batch_size_per_shard = tf.shape(input_batch)[0]  # note: tf.shape(X) is undefined until we fill the placeholder
        W_tiled = tf.tile(W, [batch_size_per_shard, 1, 1, 1, 1], name="W_tiled")

        # tile caps1_output to [batch_size_per_shard, caps1_n_caps, caps2_n_caps, caps2_n_dims, caps1_n_dims]
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
            raw_weights = tf.zeros([batch_size_per_shard, caps1_n_caps, caps2_n_caps, 1, 1],
                                   dtype=np.float32, name="raw_weights")

            rba_iter = tf.constant(1, name='rba_iteration_counter')
            caps2_output = tf.zeros(shape=(batch_size_per_shard, 1, caps2_n_caps, caps2_n_dims, 1), name='caps2_output')
            caps2_predicted, caps2_output, raw_weights, rba_iter = tf.while_loop(do_routing_cond, routing_by_agreement,
                                                                                 [caps2_predicted, caps2_output,
                                                                                  raw_weights, rba_iter])

        # This is the caps2 output!
        # tf.summary.histogram('rba_output', caps2_output)

        if print_shapes:
            print('shape of caps2_output after RbA termination: ' + str(caps2_output))

        return caps2_output


def primary_to_fc_caps_layer_tpu(input_batch, caps1_output, caps1_n_caps, caps1_n_dims, caps2_n_caps, caps2_n_dims,
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
        W = tf.Variable(W_init, name="W")

        # tile weights to [batch_size_per_shard, caps1_n_caps, caps2_n_caps, caps2_n_dims, caps1_n_dims]
        # i.e. batch_size_per_shard times a caps2_n_dims*caps1_n_dims array of [caps1_n_caps*caps2_n_caps] weight matrices
        # batch_size_per_shard = tf.shape(input_batch)[0]  # note: tf.shape(X) is undefined until we fill the placeholder
        W_tiled = tf.tile(W, [batch_size_per_shard, 1, 1, 1, 1], name="W_tiled")

        # tile caps1_output to [batch_size_per_shard, caps1_n_caps, caps2_n_caps, caps2_n_dims, caps1_n_dims]
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

            # initialize routing weights
            raw_weights = tf.zeros([batch_size_per_shard, caps1_n_caps, caps2_n_caps, 1, 1],
                                   dtype=np.float32, name="raw_weights")

            # softmax on weights
            routing_weights = tf.nn.softmax(raw_weights, dim=2, name="routing_weights")

            # weighted sum of the lower layer predictions according to the routing weights
            weighted_predictions = tf.multiply(routing_weights, caps2_predicted,
                                               name="weighted_predictions")
            weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keep_dims=True,
                                         name="weighted_sum")

            # squash
            caps2_rba1_output = squash(weighted_sum, axis=-2,
                                  name="caps2_rba_output")

            # check shape
            if print_shapes:
                print('shape of caps2_output after after RbA round: ' + str(caps2_rba1_output))

            # to measure agreement, we just compute dot products between predictions and actual activations.
            # to do so, we will again use tf.matmul and tiling
            caps2_rba1_output_tiled = tf.tile(
                caps2_rba1_output, [1, caps1_n_caps, 1, 1, 1],
                name="caps2_rba_output_tiled")

            # check shape
            if print_shapes:
                print('shape of TILED caps2_output after RbA round: ' + str(caps2_rba1_output_tiled))

            # comput agreement is simple now
            agreement = tf.matmul(caps2_predicted, caps2_rba1_output_tiled,
                                  transpose_a=True, name="agreement")

            # update routing weights based on agreement
            raw_weights_round2 = tf.add(raw_weights, agreement,
                                     name="raw_weights_round_new")

            # weighted sum of the lower layer predictions according to the routing weights
            weighted_predictions_round2 = tf.multiply(routing_weights, caps2_predicted,
                                               name="weighted_predictions")
            weighted_sum_round2 = tf.reduce_sum(weighted_predictions, axis=1, keep_dims=True,
                                         name="weighted_sum")

            # squash
            caps2_rba_output_round2 = squash(weighted_sum, axis=-2,
                                       name="caps2_rba_output")

            # check shape
            if print_shapes:
                print('shape of caps2_output after after RbA round: ' + str(caps2_rba_output_round2))

            if print_shapes:
                print('shape of caps2_output after RbA termination: ' + str(caps2_rba_output_round2))

        return caps2_rba_output_round2


# takes a fc capsule layer output (caps1) as input and creates a new fully connected capsule layer (caps2)
# difference with primary_to_fc_caps_layer is that there is no need to tile.
def fc_to_fc_caps_layer(input_batch, caps1_output, caps1_n_caps, caps1_n_dims, caps2_n_caps, caps2_n_dims,
                        rba_rounds=3, print_shapes=False):
    # Now the tricky part: create the predictions of secondary capsules' activities!
    # in essence, this is creating random weights from each dimension of each layer 1 capsule
    # to each dimension of each layer 2 capsule -- initializing random transforms on caps1 output vectors.
    # To make it efficient we use only tensorflow-friendly matrix multiplications. To this end,
    # we use tf.matmul, which performs element wise matrix in multidimensional arrays. To create
    # these arrays we use tf.tile a lot. See ageron github & video for more explanations.
    with tf.name_scope('caps_fc_to_caps_fc'):
        batch_size_per_shard = tf.shape(input_batch)[0]  # note: tf.shape(X) is undefined until we actually fill the placeholder

        # initialise weights
        init_sigma = 0.01  # stdev of weights
        W_init = tf.random_normal(
            shape=(1, caps1_n_caps, caps2_n_caps, caps2_n_dims, caps1_n_dims),
            stddev=init_sigma, dtype=tf.float32, name="W_init")
        W = tf.Variable(W_init, name="W")
        W_tiled = tf.tile(W, [batch_size_per_shard, 1, 1, 1, 1], name="W_tiled")
        caps1_output = tf.transpose(caps1_output,[0, 2, 1, 3, 4])
        caps1_output_tiled = tf.tile(caps1_output, [1, 1, caps2_n_caps, 1, 1],
                                     name="caps1_output_tiled")  # tile

        # check shapes
        if print_shapes:
            print('shape of W_tiled: ' + str(W_tiled))
            print('shape of fc_caps_input_tiled: ' + str(caps1_output_tiled))

        # Thanks to all this hard work, computing the secondary capsules' predicted activities is easy peasy:
        caps2_predicted = tf.matmul(W_tiled, caps1_output_tiled,
                                    name="caps2_predicted")
        # tf.summary.histogram('rba_0', caps2_predicted)

        # check shape
        if print_shapes:
            print('shape of caps2_predicted: ' + str(caps2_predicted))

        with tf.name_scope('routing_by_agreement'):

            def do_routing_cond(caps2_predicted, caps2_output, raw_weights, rba_iter, max_iter=rba_rounds):
                return tf.less(rba_iter, max_iter + 1, name='do_routing_cond')

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
            raw_weights = tf.zeros([batch_size_per_shard, caps1_n_caps, caps2_n_caps, 1, 1],
                                   dtype=np.float32, name="raw_weights")

            rba_iter = tf.constant(1, name='rba_iteration_counter')
            caps2_output = tf.zeros(shape=(batch_size_per_shard, 1, caps2_n_caps, caps2_n_dims, 1), name='caps2_output')
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


def compute_margin_loss(labels, caps2_output, caps2_n_caps, m_plus, m_minus, lambda_, print_shapes=False):
    with tf.name_scope('margin_loss'):

        if len(labels.shape) == 1:  # there is a single shape to classify

            # the T in the margin equation can be computed easily
            T = tf.one_hot(labels, depth=caps2_n_caps, name="T")
            if print_shapes:
                print('Computing output margin loss based on ONE label per image')
                print('shape of output margin loss function -- T: ' + str(T))

        else:  # there are more than one shape to classify

            # trick to get a vector for each image in the batch. Labels [0,2] -> [[1, 0, 1]] and [1,1] -> [[0, 1, 0]]
            T_raw = tf.one_hot(labels, depth=caps2_n_caps)
            T = tf.reduce_sum(T_raw, axis=1)
            T = tf.minimum(T, 1)
            if print_shapes:
                print('Computing output margin loss based on ' + str(len(labels.shape)) + ' labels per image')
                print('shape of output margin loss function -- T: ' + str(T))

        # the norms of the last capsules are taken as output probabilities
        caps2_output_norm = safe_norm(caps2_output, axis=-2, keep_dims=True, name="caps2_output_norm")
        if print_shapes:
            print('shape of output margin loss function -- caps2_output_norm: ' + str(caps2_output_norm))

        # present and absent errors go into the loss
        present_error_raw = tf.square(tf.maximum(0., m_plus - caps2_output_norm), name="present_error_raw")
        if print_shapes:
            print('shape of output margin loss function -- present_error_raw: ' + str(present_error_raw))
        present_error = tf.reshape(present_error_raw, shape=(batch_size_per_shard, caps2_n_caps), name="present_error")  # there is a term for each of the caps2ncaps possible outputs
        if print_shapes:
            print('shape of output margin loss function -- present_error: ' + str(present_error))
        absent_error_raw = tf.square(tf.maximum(0., caps2_output_norm - m_minus), name="absent_error_raw")
        absent_error = tf.reshape(absent_error_raw, shape=(batch_size_per_shard, caps2_n_caps), name="absent_error")    # there is a term for each of the caps2ncaps possible outputs

        # compute the margin loss
        L = tf.add(T * present_error, lambda_ * (1.0 - T) * absent_error, name="L")
        if print_shapes:
            print('shape of output margin loss function -- L: ' + str(L))
        margin_loss = tf.reduce_mean(tf.reduce_sum(L, axis=1), name="margin_loss")
        # tf.summary.scalar('margin_loss_output', margin_loss)

        return margin_loss


def compute_primary_caps_loss(labels, caps1_output_with_maps, caps1_n_maps,  m_plus_primary, m_minus_primary, lambda_primary, print_shapes=False):

    # this loss will penalize capsules activated in the wrong map

    with tf.name_scope('primary_caps_loss'):
        if len(labels.shape) == 1:  # there is a single shape to classify

            # the T in the margin equation can be computed easily
            T = tf.one_hot(labels, depth=caps1_n_maps, name="T")
            if print_shapes:
                print('Computing primary margin loss based on ONE label per image')
                print('shape of primary margin loss function -- T: ' + str(T))

        else:  # there are more than one shape to classify

            # trick to get a vector for each image in the batch. Labels [0,2] -> [[1, 0, 1]] and [1,1] -> [[0, 1, 0]]
            T_raw = tf.one_hot(labels, depth=caps1_n_maps, name='T_raw')
            T = tf.reduce_sum(T_raw, axis=1)
            T = tf.minimum(T, 1)
            if print_shapes:
                print('Computing primarymargin loss based on ' + str(len(labels.shape)) + 'labels per image')
                print('shape of primary margin loss function -- T: ' + str(T))

        # the norms of the capsules
        caps1_output_norm = safe_norm(caps1_output_with_maps, axis=-1, keep_dims=False, name="caps1_output_norm")
        # tf.summary.histogram('primary_capsule_norms_map_0', caps1_output_norm[0, 0, :])
        if print_shapes:
            print('shape of primary caps loss function -- caps1_output_norm: ' + str(caps1_output_norm))
        # we see these norms as one vector per class and squash them + take the norm (i.e., if many capsules are active
        # for a given class, this class has a high norm. We wish the irrelevant classes to have low norms.
        caps1_output_squash = squash(caps1_output_norm, axis=-1, name='caps1_output_squash')
        # tf.summary.histogram('primary_capsule_squash', caps1_output_squash[0, 0, :])
        if print_shapes:
            print('shape of primary caps loss function -- caps1_output_squash: ' + str(caps1_output_squash))
        caps1_output_class_norm = safe_norm(caps1_output_squash, axis=-1, keep_dims=False, name="caps1_output_class_norm")
        # tf.summary.histogram('primary_capsule_class_norms', caps1_output_class_norm[0, :])
        if print_shapes:
            print('shape of primary caps loss function -- caps1_output_class_norm: ' + str(caps1_output_class_norm))

        # present and absent errors go into the loss
        present_error = tf.square(tf.maximum(0., m_plus_primary - caps1_output_class_norm), name="present_error")
        absent_error = tf.square(tf.maximum(0., caps1_output_class_norm - m_minus_primary), name="absent_error")
        if print_shapes:
            print('shape of primary caps loss function -- present_error: ' + str(present_error))
            print('shape of primary caps loss function -- absent_error: ' + str(absent_error))

        # compute the margin loss
        L = tf.add(T * present_error, lambda_primary * (1.0 - T) * absent_error, name="L")
        if print_shapes:
            print('shape of primary caps loss function -- L: ' + str(L))
        margin_loss = tf.reduce_mean(tf.reduce_sum(L, axis=1), name="margin_loss")
        # tf.summary.scalar('loss', margin_loss)

        return margin_loss


def create_masked_decoder_input(labels, labels_pred, caps_output, n_caps, caps_n_dims, mask_with_labels,
                                print_shapes=False):
    # CREATE MASK #

    with tf.name_scope('create_masked_decoder_input'):
        # use the above condition to find out which (label vs. predicted label) to use. returns, for example, 3.
        reconstruction_targets = tf.cond(mask_with_labels,  # condition
                                         lambda: labels,  # if True
                                         lambda: labels_pred,  # if False
                                         name="reconstruction_targets")

        # Let's create the reconstruction mask. It should be equal to 1.0 for the target class, and 0.0 for the
        # other classes, for each instance in the batch.
        reconstruction_mask = tf.one_hot(reconstruction_targets, depth=n_caps, name="reconstruction_mask")
        if len(labels.shape) > 1:  # there are different shapes in the image
            reconstruction_mask = tf.reduce_sum(reconstruction_mask, axis=1)
            reconstruction_mask = tf.minimum(reconstruction_mask, 1)

        # caps2_output shape is (batch size, 1, 10, 16, 1). We want to multiply it by the reconstruction_mask,
        # but the shape of the reconstruction_mask is (batch size, 10). We must reshape it to (batch size, 1, 10, 1, 1)
        # to make multiplication possible:
        reconstruction_mask_reshaped = tf.reshape(reconstruction_mask, [batch_size_per_shard, 1, n_caps, 1, 1], name="reconstruction_mask_reshaped")

        # Apply the mask!
        caps2_output_masked = tf.multiply(caps_output, reconstruction_mask_reshaped, name="caps2_output_masked")

        if print_shapes:
            # check shape
            print('shape of masked output: ' + str(caps2_output_masked))

        # flatten the masked output to feed to the decoder
        decoder_input = tf.reshape(caps2_output_masked, [batch_size_per_shard, n_caps * caps_n_dims], name="decoder_input")

        if print_shapes:
            # check shape
            print('shape of decoder input: ' + str(decoder_input))

        return decoder_input


def compute_n_shapes_loss(masked_input, n_shapes, n_shapes_max, print_shapes=False):

    with tf.name_scope('n_shapes_loss'):
        one_hot_n_shapes = tf.one_hot(tf.cast(n_shapes, tf.int32), n_shapes_max)
        if len(n_shapes.shape) > 1:  # there are different shapes in the image
            one_hot_n_shapes = tf.reduce_sum(one_hot_n_shapes, axis=1)
            one_hot_n_shapes = tf.minimum(one_hot_n_shapes, 1)
        n_shapes_logits = tf.layers.dense(masked_input, n_shapes_max, activation=tf.nn.relu, name="n_shapes_logits")
        n_shapes_xent = tf.losses.softmax_cross_entropy(one_hot_n_shapes, n_shapes_logits)
        # tf.summary.scalar('n_shapes_xentropy', n_shapes_xent)
        correct = tf.equal(n_shapes, tf.cast(tf.argmax(n_shapes_logits, axis=1), tf.float32), name="correct")
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
        # tf.summary.scalar('accuracy', accuracy)

        if print_shapes:
            print('shape of compute_n_shapes_loss -- one_hot_n_shapes: ' + str(one_hot_n_shapes))
            print('shape of compute_n_shapes_loss -- n_shapes_logits: ' + str(n_shapes_logits))
            print('shape of compute_n_shapes_loss -- n_shapes_xent: ' + str(n_shapes_xent))


        return n_shapes_xent


def compute_vernier_offset_loss(vernier_capsule, labels, print_shapes=False):

    with tf.name_scope('vernier_offset_loss'):

        one_hot_offsets = tf.one_hot(tf.cast(labels, tf.int32), 3)
        offset_logits = tf.layers.dense(vernier_capsule, 3, activation=tf.nn.relu, name="offset_logits")
        offset_xent = tf.losses.softmax_cross_entropy(one_hot_offsets, offset_logits)
        # tf.summary.scalar('training_vernier_offset_xentropy', offset_xent)
        correct = tf.equal(labels, tf.cast(tf.argmax(offset_logits, axis=1), tf.float32), name="correct")
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
        # tf.summary.scalar('accuracy', accuracy)

        if print_shapes:
            print('shape of compute_vernier_offset_loss -- input to decoder: ' + str(vernier_capsule))
            print('shape of compute_vernier_offset_loss -- one_hot_offsets: ' + str(one_hot_offsets))
            print('shape of compute_vernier_offset_loss -- offset_logits: ' + str(offset_logits))
            print('shape of compute_vernier_offset_loss -- offset_xent: ' + str(offset_xent))

        return offset_xent, accuracy, offset_logits


def decoder_with_mask(decoder_input, output_width, output_height, n_hidden1=None, n_hidden2=None, n_hidden3=None, print_shapes=False, **deconv_params):

    with tf.name_scope("decoder"):

        n_output = output_width * output_height

        if deconv_params['use_deconvolution_decoder'] is True:

            if deconv_params['fc_width'] is not None:
                hidden1 = tf.layers.dense(decoder_input, deconv_params['fc_width']*deconv_params['fc_height'], activation=tf.nn.relu, name="fc_hidden1")
                hidden1_2d = tf.reshape(hidden1, shape=[batch_size_per_shard, deconv_params['fc_height'], deconv_params['fc_width']])
                hidden1_2d = tf.expand_dims(hidden1_2d, axis=-1)
                # tf.summary.histogram('deconv_decoder_hidden1', hidden1_2d)
                if print_shapes:
                    print('shape of decoder first fc: ' + str(hidden1_2d))

                hidden2 = tf.layers.conv2d_transpose(hidden1_2d, deconv_params['deconv_filters2'], deconv_params['deconv_kernel2'],  deconv_params['deconv_strides2'], activation=tf.nn.relu, name="deconv_hidden2")
                # tf.summary.histogram('deconv_decoder_hidden2', hidden2)
                if print_shapes:
                    print('shape of decoder 1st deconv output: ' + str(hidden2))

                if deconv_params['final_fc'] is True:
                    hidden2_flat = tf.reshape(hidden2, shape=[batch_size_per_shard,n_output*deconv_params['deconv_filters2']])
                    if print_shapes:
                        print('shape of decoder 1st deconv output flat: ' + str(hidden2_flat))

                    decoder_output = tf.layers.dense(hidden2_flat, n_output, activation=tf.nn.sigmoid, name="deconv_decoder_output")
                    if print_shapes:
                        print('shape of decoder output: ' + str(decoder_output))
                else:
                    decoder_output = tf.reduce_sum(hidden2, axis=-1)
                    if print_shapes:
                        print('shape of decoder output: ' + str(decoder_output))
                    decoder_output = tf.reshape(decoder_output, shape=[batch_size_per_shard, n_output])
                    if print_shapes:
                        print('shape of decoder output flat: ' + str(decoder_output))

            return decoder_output

        else:

            if n_hidden1 is not None:
                hidden1 = tf.layers.dense(decoder_input, n_hidden1,
                                          activation=tf.nn.relu,
                                          name="hidden1")
                # tf.summary.histogram('decoder_hidden1', hidden1)
                if n_hidden2 is not None:
                    hidden2 = tf.layers.dense(hidden1, n_hidden2,
                                              activation=tf.nn.relu,
                                              name="hidden2")
                    # tf.summary.histogram('decoder_hidden2', hidden2)
                    if n_hidden3 is not None:
                        hidden3 = tf.layers.dense(hidden2, n_hidden3,
                                                  activation=tf.nn.relu,
                                                  name="hidden2")
                        # tf.summary.histogram('decoder_hidden3', hidden3)
                        decoder_output = tf.layers.dense(hidden3, n_output,
                                                         activation=tf.nn.sigmoid,
                                                         name="decoder_output")
                    else:
                        decoder_output = tf.layers.dense(hidden2, n_output,
                                                         activation=tf.nn.sigmoid,
                                                         name="decoder_output")
                else:
                    decoder_output = tf.layers.dense(hidden1, n_output,
                                                     activation=tf.nn.sigmoid,
                                                     name="decoder_output")
            else:
                decoder_output = tf.layers.dense(decoder_input, n_output,
                                                 activation=tf.nn.sigmoid,
                                                 name="decoder_output")

            # tf.summary.histogram('decoder_output', decoder_output)

            return decoder_output


def batch_norm_fc_layer(x, n_out, phase, name='', activation=None):
    with tf.variable_scope('batch_norm_layer'):
        h1 = tf.layers.dense(x, n_out, activation=None, name=name)
        h2 = tf.contrib.layers.batch_norm(h1,
                                          center=True, scale=True,
                                          is_training=phase,
                                          scope=name+'bn')
    if activation is None:
        return h2
    else:
        return activation(h2)


def decoder_with_mask_batch_norm(decoder_input, n_output, n_hidden1=None, n_hidden2=None, n_hidden3=None, phase=True, name=''):

    with tf.name_scope(name+"decoder"):

        if n_hidden1 is not None:
            hidden1 = batch_norm_fc_layer(decoder_input, n_hidden1, phase, name=name+'hidden1', activation=tf.nn.elu)
            # tf.summary.histogram(name+'_hidden1_bn', hidden1)
            if n_hidden2 is not None:
                hidden2 = batch_norm_fc_layer(hidden1, n_hidden2, phase, name=name+'hidden2', activation=tf.nn.elu)
                # tf.summary.histogram(name+'_hidden2_bn', hidden2)
                if n_hidden3 is not None:
                    hidden3 = batch_norm_fc_layer(hidden2, n_hidden3, phase, name=name + 'hidden2', activation=tf.nn.elu)
                    # tf.summary.histogram(name + '_hidden3_bn', hidden3)
                    decoder_output = tf.layers.dense(hidden3, n_output,
                                                     activation=tf.nn.sigmoid,
                                                     name=name + "_output")
                else:
                    decoder_output = tf.layers.dense(hidden2, n_output,
                                                 activation=tf.nn.sigmoid,
                                                 name=name+"_output")
            else:
                decoder_output = tf.layers.dense(hidden1, n_output,
                                                 activation=tf.nn.sigmoid,
                                                 name=name + "_output")
        else:
            decoder_output = tf.layers.dense(decoder_input, n_output,
                                             activation=tf.nn.sigmoid,
                                             name=name + "_output")

        # tf.summary.histogram(name+'_output', decoder_output)

        return decoder_output


def primary_capsule_reconstruction(shape_patch, labels, caps1_output, caps1_output_norms, primary_caps_decoder_n_hidden1,
                                   primary_caps_decoder_n_hidden2, primary_caps_decoder_n_output,
                                   is_training):

    with tf.variable_scope('primary_capsule_decoder'):

        highest_norm_capsule = tf.argmax(caps1_output_norms, axis=1, output_type=tf.int32)

        decoder_input_primary_caps = [caps1_output[0, highest_norm_capsule[0], :]]
        # this is messy but it's the only way I found to flexibly deal with batch size
        for i in range(1, 15):
            decoder_input_primary_caps = tf.concat([decoder_input_primary_caps,
                                                    [caps1_output[i, highest_norm_capsule[i], :]]], axis=0)

        # add label to capsule inputs (the decoder needs to know which capsule type it is decoding from in order to
        # create different outputs for different capsule types. Otherwise it just ends up with the capsule's activities
        # and has no way to figure out which shape to draw).
        # Let's create the reconstruction labels. It should be equal to 1.0 for the target class, and 0.0 for the
        # other classes, for each instance in the batch.

        decoder_input_primary_caps = tf.concat([tf.transpose([tf.cast(labels, tf.float32)]), decoder_input_primary_caps], axis=1)

        decoder_output_primary_caps = decoder_with_mask_batch_norm(decoder_input_primary_caps,
                                                                   primary_caps_decoder_n_hidden1,
                                                                   primary_caps_decoder_n_hidden2,
                                                                   primary_caps_decoder_n_output,
                                                                   phase=is_training, name='primary_decoder')

        decoder_output_image_primary_caps = tf.reshape(decoder_output_primary_caps,
                                                       [batch_size_per_shard, tf.shape(shape_patch)[1], tf.shape(shape_patch)[2], 1])
        # tf.summary.image('decoder_output', decoder_output_image_primary_caps, 6)

        return decoder_output_primary_caps, highest_norm_capsule[0]

# takes a n*n input and a flat decoder output
def compute_reconstruction_loss(input, reconstruction, loss_type='squared_difference', gain=None, no_tensorboard=False):

    with tf.name_scope('reconstruction_loss'):

        X_flat = tf.reshape(input, [batch_size_per_shard, tf.shape(reconstruction)[1]], name="X_flat")

        if gain is None:
            gain = tf.ones_like(X_flat[:,0])

        squared_difference = tf.square(X_flat - reconstruction, name="squared_difference")
        sparsity_constant = 1
        sparsity_floor = 0  # value of the black background
        rescale_constant = 100000
        threshold = 0
        threshold_constant = 100000
        used_loss = 'sparse'  # which loss to use for training if loss_type is 'plot_all'


        if loss_type is 'squared_difference':
            reconstruction_loss = tf.reduce_sum(squared_difference,  name="reconstruction_loss")
            # if no_tensorboard is False:
                # tf.summary.scalar('reconstruction_loss_squared_diff', reconstruction_loss)
            stimuli_square_differences = tf.reduce_sum(squared_difference, axis=1, name="square_diff_for_each_stimulus")

        elif loss_type is 'sparse':
            reconstruction_loss_sparse = tf.reduce_sum(squared_difference, name="reconstruction_loss")
            # if no_tensorboard is False:
                # tf.summary.scalar('squared_difference_loss', reconstruction_loss_sparse)
            sparsity_loss = sparsity_constant * tf.reduce_sum(tf.square(reconstruction-sparsity_floor))
            # if no_tensorboard is False:
                # tf.summary.scalar('sparsity_loss', sparsity_loss)
            reconstruction_loss_sparse = tf.add(reconstruction_loss_sparse, sparsity_loss, name='sparsity_constraint')
            # if no_tensorboard is False:
                # tf.summary.scalar('reconstruction_loss_sparse', reconstruction_loss_sparse)
            reconstruction_loss = reconstruction_loss_sparse
            stimuli_square_differences = tf.reduce_sum(squared_difference + sparsity_constant * tf.reduce_sum(tf.square(reconstruction)), axis=1,  name="sparse_square_diff_for_each_stimulus")

        elif loss_type is 'rescale':  # rescale to have errors of the same scale for large and small stimuli
            squared_difference_sum = tf.reduce_sum(squared_difference, axis=1, name="squared_difference_sum")
            scales = tf.reduce_sum(X_flat, axis=1, name='scales')
            diff_rescale = rescale_constant * squared_difference_sum / scales
            reconstruction_loss = tf.reduce_sum(diff_rescale, name="reconstruction_loss")

        elif loss_type is 'threshold':
            template = tf.maximum(X_flat, reconstruction, name='template')

            def apply_threshold(input, threshold):
                with tf.name_scope('apply_threshold'):
                    cond = tf.less(input, threshold*tf.ones(tf.shape(input)))
                    out = tf.where(cond, tf.zeros(tf.shape(input)), input)

                    return out

            template = apply_threshold(template, threshold)
            n_above_threshold = tf.count_nonzero(template, dtype=tf.float32)
            thresholded_squared_difference = tf.multiply(squared_difference, template)
            reconstruction_loss = tf.reduce_sum(thresholded_squared_difference, name="reconstruction_loss")/n_above_threshold*threshold_constant

        elif loss_type is 'plot_all':
            reconstruction_loss_square_diff = tf.reduce_sum(squared_difference, name="reconstruction_loss_square_diff")
            # if no_tensorboard is False:
                # tf.summary.scalar('reconstruction_loss_square_diff', reconstruction_loss_square_diff)

            reconstruction_loss_sparse = tf.reduce_sum(squared_difference, name="reconstruction_loss")
            sparsity_loss = sparsity_constant * tf.reduce_sum(tf.square(reconstruction))
            # if no_tensorboard is False:
                # tf.summary.scalar('sparsity_loss', sparsity_loss)
            reconstruction_loss_sparse = tf.add(reconstruction_loss_sparse, sparsity_loss, name='sparsity_constraint')
            # if no_tensorboard is False:
                # tf.summary.scalar('reconstruction_loss_sparse', reconstruction_loss_sparse)

            squared_difference_sum = tf.reduce_sum(squared_difference, axis=1, name="squared_difference_sum")
            scales = tf.reduce_sum(X_flat, axis=1, name='scales')
            diff_rescale = rescale_constant * squared_difference_sum / scales
            reconstruction_loss_rescale = tf.reduce_sum(diff_rescale, name="reconstruction_loss")
            # if no_tensorboard is False:
                # tf.summary.scalar('reconstruction_loss_rescale', reconstruction_loss_rescale)

            threshold = 0
            template = tf.maximum(X_flat, reconstruction, name='template')
            def apply_threshold(input, threshold):
                with tf.name_scope('apply_threshold'):
                    cond = tf.less(input, threshold * tf.ones(tf.shape(input)))
                    out = tf.where(cond, tf.zeros(tf.shape(input)), input)

                    return out
            template = apply_threshold(template, threshold)
            n_above_threshold = tf.count_nonzero(template, dtype=tf.float32)
            thresholded_squared_difference = tf.multiply(squared_difference, template)
            reconstruction_loss_threshold = tf.reduce_sum(thresholded_squared_difference, name="reconstruction_loss") / n_above_threshold * threshold_constant
            # if no_tensorboard is False:
                # tf.summary.scalar('reconstruction_loss_threshold', reconstruction_loss_threshold)

            if used_loss is 'squared_difference':
                reconstruction_loss = reconstruction_loss_square_diff
                stimuli_square_differences = tf.reduce_sum(squared_difference, axis=1, name="square_diff_for_each_stimulus")
            elif used_loss is 'sparse':
                reconstruction_loss = reconstruction_loss_sparse
                stimuli_square_differences = tf.reduce_sum(squared_difference + sparsity_constant * tf.reduce_sum(tf.square(reconstruction)), axis=1, name="sparse_square_diff_for_each_stimulus")
            elif used_loss is 'rescale':
                reconstruction_loss = reconstruction_loss_rescale
                stimuli_square_differences = diff_rescale
            elif used_loss is 'threshold':
                reconstruction_loss = reconstruction_loss_threshold
                stimuli_square_differences = tf.reduce_sum(thresholded_squared_difference, axis=1, name="square_diff_for_each_stimulus")

        return reconstruction_loss, stimuli_square_differences


# decode vernier orientation from an input
def vernier_classifier(input, is_training=True, n_hidden1=None, n_hidden2=None, batch_norm=False, dropout=False, name=''):

    with tf.name_scope(name):
        batch_size_per_shard = tf.shape(input)[0]

        # find how many units are in this layer to flatten it
        items_to_multiply = len(np.shape(input)) - 1
        n_units = 1
        for i in range(1, items_to_multiply + 1):
            n_units = n_units * int(np.shape(input)[i])

        flat_input = tf.reshape(input, [batch_size_per_shard, n_units])
        # tf.summary.histogram('classifier_input_no_bn', flat_input)

        if batch_norm:
            print('Applying batch normalization to vernier decoder input.')
            flat_input = tf.contrib.layers.batch_norm(flat_input, center=True, scale=True, is_training=is_training, scope=name + 'input_bn')
            # tf.summary.histogram('classifier_input_bn', flat_input)

        if n_hidden1 is None:

            print('No hidden layer in vernier classifier.')
            if not batch_norm:
                classifier_fc = tf.layers.dense(flat_input, 2, name='classifier_top_fc')
            else:
                print('Using a single batch norm fc layer as vernier decoder.')
                classifier_fc = batch_norm_fc_layer(flat_input, 2, is_training, name='classifier_top_fc')
            # tf.summary.histogram(name + '_fc', classifier_fc)

        else:

            print('Number of hidden units in first hidden layer: ' + str(n_hidden1))
            print('Number of hidden units in second hidden layer: ' + str(n_hidden2))
            with tf.device('/cpu:0'):

                classifier_hidden1 = tf.layers.dense(flat_input, n_hidden1, activation=tf.nn.elu, name=name + '_hidden_fc')

                if is_training and dropout:
                    print('Using dropout at vernier classifier first hidden layer.')
                    classifier_hidden1 = tf.nn.dropout(classifier_hidden1, keep_prob=0.5, name='vernier_fc_dropout')
                else:
                    classifier_hidden1 = tf.nn.dropout(classifier_hidden1, keep_prob=1.0, name='vernier_fc_dropout')
                    # classifier_hidden = batch_norm_layer(flat_input, n_hidden, is_training,
                    # activation=tf.nn.relu, name='classifier_hidden_fc')
                # tf.summary.histogram(name + '_hidden', classifier_hidden1)

            if n_hidden2 is None:
                if not batch_norm:
                    classifier_fc = tf.layers.dense(classifier_hidden1, 2, activation=tf.nn.elu, name=name + '_top_fc')
                else:
                    print('Applying batch normalization to output layer.')
                    classifier_fc = batch_norm_fc_layer(classifier_hidden1, 2, is_training, activation=tf.nn.elu, name='classifier_top_fc')
                # tf.summary.histogram(name + '_fc', classifier_fc)
            else:
                if not batch_norm:
                    classifier_hidden2 = tf.layers.dense(classifier_hidden1, 2, activation=tf.nn.elu, name=name + '_hidden2')
                else:
                    print('Applying batch normalization to hidden2 layer.')
                    classifier_hidden2 = batch_norm_fc_layer(classifier_hidden1, 2, is_training, activation=tf.nn.elu, name='classifier_hidden2')
                # tf.summary.histogram(name + '_hidden2', classifier_hidden2)

                if not batch_norm:
                    classifier_fc = tf.layers.dense(classifier_hidden2, 2, activation=tf.nn.elu, name=name + '_top_fc')
                else:
                    print('Applying batch normalization to output layer.')
                    classifier_fc = batch_norm_fc_layer(classifier_hidden2, 2, is_training, activation=tf.nn.elu, name='classifier_top_fc')
                # tf.summary.histogram(name + '_fc', classifier_fc)

        classifier_out = tf.nn.softmax(classifier_fc, name='softmax')

        return classifier_out


def vernier_x_entropy(prediction_vector, label):
    with tf.name_scope("x_entropy"):
        xent = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=prediction_vector, labels=tf.one_hot(label, 2)), name="xent")
        # tf.summary.scalar("xent", xent)
        return xent


def vernier_correct_mean(prediction, label):
    with tf.name_scope('correct_mean'):
        correct = tf.equal(prediction, label, name="correct")
        correct_mean = tf.reduce_mean(tf.cast(correct, tf.float32), name="correct_mean")
        # tf.summary.scalar('correct_mean', correct_mean)
        return correct_mean


def run_test_stimuli(test_stimuli, n_stimuli, stim_maker, batch_size_per_shard, noise_level, normalize_images, fixed_stim_position, simultaneous_shapes,
                     capser, X, y, reconstruction_targets, vernier_offsets, mask_with_labels, sess, LOGDIR,
                     label_encoding='lr_01', res_path=None, plot_ID=None, saver=False, checkpoint_path=None,
                     summary_writer=None, global_step=None):

    for category in test_stimuli.keys():

        stim_matrices = test_stimuli[category]

        if saver is not False:
            print('Decoding vernier orientation for : ' + category)
            saver.restore(sess, checkpoint_path)

        # we will collect correct responses here
        correct_responses = np.zeros(shape=(3))

        for this_stim in range(3):

            n_batches = n_stimuli // batch_size_per_shard

            for batch in range(n_batches):
                curr_stim = stim_matrices[this_stim]

                # get a batch of the current stimulus
                batch_data, vernier_labels = stim_maker.makeConfigBatch(batch_size_per_shard, curr_stim, noiseLevel=noise_level,
                                                                        normalize=normalize_images,
                                                                        fixed_position=fixed_stim_position)

                reconstruction_targets_serge = np.repeat(batch_data, 2, axis=-1)

                if label_encoding is 'nothinglr_012':
                    vernier_labels = -vernier_labels+2  # (just because capser has l&r = 1&2 as vernier labels instead of l&r = 1&0

                # we will feed an empty y that will not be used, but needs to have the right shape (called y_serge)
                if simultaneous_shapes == 1:
                    y_serge = np.zeros(shape=(len(vernier_labels)))
                else:
                    y_serge = np.zeros(shape=(len(vernier_labels), 2))

                # run test stimuli through the netwoek and get classifier output:
                correct_in_this_batch_all = sess.run(capser["vernier_accuracy"],
                                                     feed_dict={X: batch_data,
                                                                reconstruction_targets: reconstruction_targets_serge,
                                                                y: y_serge,  # just a trick to make it run, we actually don't care about this
                                                                vernier_offsets: vernier_labels,
                                                                mask_with_labels: False})
                correct_responses[this_stim] += np.array(correct_in_this_batch_all)

        percent_correct = correct_responses * 100 / n_batches

        if saver is False:
            summary = tf.summary()
            summary.value.add(tag='zzz_uncrowding_exp/' + category + '_0_vernier', simple_value=percent_correct[0] / 100)
            summary.value.add(tag='zzz_uncrowding_exp/' + category + '_1_crowded', simple_value=percent_correct[1] / 100)
            summary.value.add(tag='zzz_uncrowding_exp/' + category + '_2_uncrowded', simple_value=percent_correct[2] / 100)
            summary_writer.add_summary(summary, global_step)
            summary_writer.flush()


        if saver is not False:
            print('... testing done.')
            print('Percent correct for vernier decoders with stimuli: ' + category)
            print(percent_correct)
            print('Writing data and plot')
            np.save(LOGDIR + '/' + category + '_percent_correct', percent_correct)

            # PLOT
            x_labels = ['vernier', 'crowded', 'uncrowded']

            ####### PLOT RESULTS #######

            # N = len(x_labels)
            # ind = np.arange(N)  # the x locations for the groups
            # width = 0.25  # the width of the bars
            #
            # fig, ax = plt.subplots()
            # plot_color = (0. / 255, 91. / 255, 150. / 255)
            # rects1 = ax.bar(ind, percent_correct, width, color=plot_color)
            #
            # # add some text for labels, title and axes ticks, and save figure
            # ax.set_ylabel('Percent correct')
            # # ax.set_title('Vernier decoding from alexnet layers')
            # ax.set_xticks(ind + width / 2)
            # ax.set_xticklabels(x_labels)
            # ax.plot([-0.3, N], [50, 50], 'k--')  # chance level cashed line
            # ax.legend(rects1, ('vernier', '1 ' + category[:-1], '7 ' + category))
            # plt.savefig(res_path + '/' + category + '_uncrowding_plot_' + plot_ID + '.png')
            # plt.close()

