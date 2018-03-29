import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import io

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
        # reshape the second layer to be caps1_n_dims-Dim capsules (since the next layer is FC, we don't need to keep
        # the [batch,xx,xx,n_feature_maps,caps1_n_dims] so we just flatten it to keep it simple)
        caps1_raw = tf.reshape(conv_for_caps, [-1, caps1_n_caps, caps1_n_dims],
                               name="caps1_raw")
        tf.summary.histogram('caps1_raw', caps1_raw)

        # squash capsule outputs
        caps1_output = squash(caps1_raw, name="caps1_output")
        tf.summary.histogram('caps1_output', caps1_output)
        if print_shapes:
            print('shape of caps1_output: '+str(caps1_output))

        return caps1_output


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
        W_init = tf.random_normal(
            shape=(1, caps1_n_caps, caps2_n_caps, caps2_n_dims, caps1_n_dims),
            stddev=init_sigma, dtype=tf.float32, name="W_init")
        W = tf.Variable(W_init, name="W")

        # tile weights to [batch_size, caps1_n_caps, caps2_n_caps, caps2_n_dims, caps1_n_dims]
        # i.e. batch_size times a caps2_n_dims*caps1_n_dims array of [caps1_n_caps*caps2_n_caps] weight matrices
        batch_size = tf.shape(input_batch)[0]  # note: tf.shape(X) is undefined until we fill the placeholder
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

        tf.summary.histogram('rba_0', caps2_predicted)

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
        tf.summary.histogram('rba_output', caps2_output)

        if print_shapes:
            print('shape of caps2_output after RbA termination: ' + str(caps2_output))

        return caps2_output


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
        batch_size = tf.shape(input_batch)[0]  # note: tf.shape(X) is undefined until we actually fill the placeholder

        # initialise weights
        init_sigma = 0.01  # stdev of weights
        W_init = tf.random_normal(
            shape=(1, caps1_n_caps, caps2_n_caps, caps2_n_dims, caps1_n_dims),
            stddev=init_sigma, dtype=tf.float32, name="W_init")
        W = tf.Variable(W_init, name="W")
        W_tiled = tf.tile(W, [batch_size, 1, 1, 1, 1], name="W_tiled")
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
        tf.summary.histogram('rba_0', caps2_predicted)

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
            raw_weights = tf.zeros([batch_size, caps1_n_caps, caps2_n_caps, 1, 1],
                                   dtype=np.float32, name="raw_weights")

            rba_iter = tf.constant(1, name='rba_iteration_counter')
            caps2_output = tf.zeros(shape=(batch_size, 1, caps2_n_caps, caps2_n_dims, 1), name='caps2_output')
            caps2_predicted, caps2_output, raw_weights, rba_iter = tf.while_loop(do_routing_cond, routing_by_agreement,
                                                                                 [caps2_predicted, caps2_output,
                                                                                  raw_weights, rba_iter])

            # This is the caps2 output!
            tf.summary.histogram('rba_output', caps2_output)

        if print_shapes:
            print('shape of caps2_output after RbA termination: ' + str(caps2_output))

        return caps2_output


def caps_prediction(caps2_output, print_shapes=False):
    with tf.name_scope('net_prediction'):
        y_proba = safe_norm(caps2_output, axis=-2, name="y_proba")
        y_proba_argmax = tf.argmax(y_proba, axis=2, name="y_proba")
        # get predicted class by squeezing out irrelevant dimensions
        y_pred = tf.squeeze(y_proba_argmax, axis=[1, 2], name="y_pred")

        if print_shapes:
            # check shapes
            print('shape of prediction: '+str(y_pred))

        return y_pred


def compute_margin_loss(labels, caps2_output, caps2_n_caps, m_plus, m_minus, lambda_):
    with tf.name_scope('margin_loss'):
        # the T in the margin equation can be computed easily
        T = tf.one_hot(labels, depth=caps2_n_caps, name="T")

        # the norms of the last capsules are taken as output probabilities
        caps2_output_norm = safe_norm(caps2_output, axis=-2, keep_dims=True,
                                      name="caps2_output_norm")

        # present and absent errors go into the loss
        present_error_raw = tf.square(tf.maximum(0., m_plus - caps2_output_norm),
                                      name="present_error_raw")
        present_error = tf.reshape(present_error_raw, shape=(-1, caps2_n_caps),
                                   name="present_error")  # there is a term for each of the caps2ncaps possible outputs
        absent_error_raw = tf.square(tf.maximum(0., caps2_output_norm - m_minus),
                                     name="absent_error_raw")
        absent_error = tf.reshape(absent_error_raw, shape=(-1, caps2_n_caps),
                                  name="absent_error")    # there is a term for each of the caps2ncaps possible outputs

        # compute the margin loss
        L = tf.add(T * present_error, lambda_ * (1.0 - T) * absent_error,
                   name="L")
        margin_loss = tf.reduce_mean(tf.reduce_sum(L, axis=1), name="margin_loss")

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
        reconstruction_mask = tf.one_hot(reconstruction_targets,
                                         depth=n_caps,
                                         name="reconstruction_mask")

        # caps2_output shape is (batch size, 1, 10, 16, 1). We want to multiply it by the reconstruction_mask,
        # but the shape of the reconstruction_mask is (batch size, 10). We must reshape it to (batch size, 1, 10, 1, 1)
        # to make multiplication possible:
        reconstruction_mask_reshaped = tf.reshape(
            reconstruction_mask, [-1, 1, n_caps, 1, 1],
            name="reconstruction_mask_reshaped")

        # Apply the mask!
        caps2_output_masked = tf.multiply(
            caps_output, reconstruction_mask_reshaped,
            name="caps2_output_masked")

        if print_shapes:
            # check shape
            print('shape of masked output: ' + str(caps2_output_masked))

        # flatten the masked output to feed to the decoder
        decoder_input = tf.reshape(caps2_output_masked,
                                   [-1, n_caps * caps_n_dims],
                                   name="decoder_input")

        if print_shapes:
            # check shape
            print('shape of decoder input: ' + str(decoder_input))

        return decoder_input


def decoder_with_mask(decoder_input, n_hidden1, n_hidden2, n_output):

    with tf.name_scope("decoder"):

        hidden1 = tf.layers.dense(decoder_input, n_hidden1,
                                  activation=tf.nn.relu,
                                  name="hidden1")
        tf.summary.histogram('decoder_hidden1', hidden1)
        hidden2 = tf.layers.dense(hidden1, n_hidden2,
                                  activation=tf.nn.relu,
                                  name="hidden2")
        tf.summary.histogram('decoder_hidden2', hidden2)
        decoder_output = tf.layers.dense(hidden2, n_output,
                                         activation=tf.nn.sigmoid,
                                         name="decoder_output")
        tf.summary.histogram('decoder_output', decoder_output)

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


def decoder_with_mask_batch_norm(decoder_input, n_hidden1, n_hidden2, n_output, phase, name=''):

    with tf.name_scope(name+"decoder"):

        hidden1 = batch_norm_fc_layer(decoder_input, n_hidden1, phase, name=name+'hidden1', activation=tf.nn.elu)
        tf.summary.histogram(name+'_hidden1_bn', hidden1)
        hidden2 = batch_norm_fc_layer(hidden1, n_hidden2, phase, name=name+'hidden2', activation=tf.nn.elu)
        tf.summary.histogram(name+'_hidden2_bn', hidden2)
        decoder_output = tf.layers.dense(hidden2, n_output,
                                         activation=tf.nn.sigmoid,
                                         name=name+"_output")
        tf.summary.histogram(name+'_output', decoder_output)

        return decoder_output


def decoder_with_mask_3layers(decoder_input, n_hidden1, n_hidden2, n_hidden3, n_output):

    with tf.name_scope("decoder"):

        hidden1 = tf.layers.dense(decoder_input, n_hidden1,
                                  activation=tf.nn.relu,
                                  name="hidden1")
        hidden2 = tf.layers.dense(hidden1, n_hidden2,
                                  activation=tf.nn.relu,
                                  name="hidden2")
        hidden3 = tf.layers.dense(hidden2, n_hidden3,
                                  activation=tf.nn.relu,
                                  name="hidden3")
        decoder_output = tf.layers.dense(hidden3, n_output,
                                         activation=tf.nn.sigmoid,
                                         name="decoder_output")
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
                                                       [-1, tf.shape(shape_patch)[1], tf.shape(shape_patch)[2], 1])
        tf.summary.image('decoder_output', decoder_output_image_primary_caps, 6)

        return decoder_output_primary_caps, highest_norm_capsule[0]

# takes a n*n input and a flat decoder output
def compute_reconstruction_loss(input, reconstruction, rescale=False):

    with tf.name_scope('reconstruction_loss'):

        X_flat = tf.reshape(input, [-1, tf.shape(reconstruction)[1]], name="X_flat")
        squared_difference = tf.square(X_flat - reconstruction,
                                       name="squared_difference")
        if rescale:  # rescale to have errors of the same scale for large and small stimuli
            squared_difference_sum = tf.reduce_sum(squared_difference, axis=1, name="squared_difference_sum")
            scales = tf.reduce_sum(X_flat, axis=1, name='scales')
            diff_rescale = 100000 * squared_difference_sum / scales
            reconstruction_loss = tf.reduce_sum(diff_rescale,
                                          name="reconstruction_loss")
        else:
            reconstruction_loss = tf.reduce_sum(squared_difference,
                                            name="reconstruction_loss")

        reconstruction_loss = tf.add(reconstruction_loss, .5*tf.reduce_sum(tf.square(reconstruction)), name='sparsity_constraint')
        print('CAREFUL I ADDED A SPARSITY LOSS TO RECONSTRUCTION ERROR')


        return reconstruction_loss


# decode vernier orientation from an input
def vernier_classifier(input, is_training, n_hidden=1024, name=''):
    with tf.name_scope(name):
        batch_size = tf.shape(input)[0]

        # find how many units are in this layer to flatten it
        items_to_multiply = len(np.shape(input)) - 1
        n_units = 1
        for i in range(1, items_to_multiply + 1):
            n_units = n_units * int(np.shape(input)[i])

        flat_input = tf.reshape(input, [batch_size, n_units])
        tf.summary.histogram('classifier_input_no_bn', flat_input)

        flat_input = tf.contrib.layers.batch_norm(flat_input, center=True, scale=True, is_training=is_training,
                                                  scope=name + 'input_bn')
        tf.summary.histogram('classifier_input_bn', flat_input)

        if n_hidden is None:
            classifier_fc = tf.layers.dense(flat_input, 2, name='classifier_top_fc')
            # classifier_fc = batch_norm_layer(flat_input, 2, is_training, name='classifier_fc')
            tf.summary.histogram(name + '_fc', classifier_fc)
        else:
            with tf.device('/cpu:0'):
                classifier_hidden = tf.layers.dense(flat_input, n_hidden, activation=tf.nn.elu,
                                                    name=name + '_hidden_fc')
                if is_training:
                    classifier_hidden = tf.nn.dropout(classifier_hidden, keep_prob=0.5, name='vernier_fc_dropout')
                else:
                    classifier_hidden = tf.nn.dropout(classifier_hidden, keep_prob=1.0, name='vernier_fc_dropout')
                # classifier_hidden = batch_norm_layer(flat_input, n_hidden, is_training,
                # activation=tf.nn.relu, name='classifier_hidden_fc')
                tf.summary.histogram(name + '_hidden', classifier_hidden)
            classifier_fc = tf.layers.dense(classifier_hidden, 2, activation=tf.nn.elu, name=name + '_top_fc')
            # classifier_fc = batch_norm_layer(classifier_hidden, 2, is_training, name='classifier_top_fc')
            tf.summary.histogram(name + '_fc', classifier_fc)

        classifier_out = tf.nn.softmax(classifier_fc, name='softmax')

        return classifier_out


def vernier_x_entropy(prediction_vector, label):
    with tf.name_scope("x_entropy"):
        xent = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=prediction_vector, labels=tf.one_hot(label, 2)), name="xent")
        tf.summary.scalar("xent", xent)
        return xent


def vernier_correct_mean(prediction, label):
    with tf.name_scope('correct_mean'):
        correct = tf.equal(prediction, label, name="correct")
        correct_mean = tf.reduce_mean(tf.cast(correct, tf.float32), name="correct_mean")
        tf.summary.scalar('correct_mean', correct_mean)
        return correct_mean
