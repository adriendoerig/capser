# a function form of capser to make param changes easy
from __future__ import division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
from capsule_functions import primary_caps_layer, primary_to_fc_caps_layer, \
    caps_prediction, compute_margin_loss, compute_primary_caps_loss, create_masked_decoder_input, \
    decoder_with_mask, decoder_with_mask_batch_norm, primary_capsule_reconstruction, \
    compute_reconstruction_loss, safe_norm, compute_n_shapes_loss, \
    compute_vernier_offset_loss


def batch_norm_conv_layer(x, phase, name='', activation=None, **conv_params):
    with tf.variable_scope('batch_norm_conv_layer'):
        conv = tf.layers.conv2d(x, activation=None, name=name+"conv", **conv_params)
        norm_conv = tf.contrib.layers.batch_norm(conv,
                                                 center=True, scale=True,
                                                 is_training=phase,
                                                 scope=name+'bn')
        tf.summary.histogram(name, conv)
        tf.summary.histogram(name+'_batch_norm', norm_conv)
        if activation is None:
            return norm_conv
        else:
            return activation(norm_conv)


def capser_model(X, y, im_size, conv1_params, conv2_params, conv3_params,
                 caps1_n_maps, caps1_n_dims, conv_caps_params,
                 primary_caps_decoder_n_hidden1, primary_caps_decoder_n_hidden2, primary_caps_decoder_n_hidden3, primary_caps_decoder_n_output,
                 caps2_n_caps, caps2_n_dims, rba_rounds,
                 m_plus, m_minus, lambda_, alpha_margin,
                 m_plus_primary, m_minus_primary, lambda_primary, alpha_primary,
                 output_caps_decoder_n_hidden1, output_caps_decoder_n_hidden2, output_caps_decoder_n_hidden3, reconstruction_loss_type, alpha_reconstruction,
                 is_training, mask_with_labels,
                 primary_caps_decoder=False, do_primary_caps_loss=False, do_n_shapes_loss=False, do_vernier_offset_loss=False,
                 n_shapes_labels=0, n_shapes_max=0, alpha_n_shapes=0,
                 vernier_offset_labels=0, alpha_vernier_offset=0,
                 shape_patch=0, conv_batch_norm=False, decoder_batch_norm=False,
                 **output_decoder_deconv_params
                 ):


    print_shapes = True  # to print the size of each layer during graph construction

    ####################################################################################################################
    # Early conv layers and first capsules
    ####################################################################################################################


    # maybe batch-norm the input?
    # X = tf.contrib.layers.batch_norm(X, center=True, scale=True, is_training=is_training, scope='input_bn')

    with tf.name_scope('0_early_conv_layers'):
        # sizes, etc.
        conv1_width = int((im_size[0] - conv1_params["kernel_size"])/conv1_params["strides"] + 1)
        conv1_height = int((im_size[1] - conv1_params["kernel_size"])/conv1_params["strides"] + 1)
        conv2_width = int((conv1_width - conv2_params["kernel_size"])/conv2_params["strides"] + 1)
        conv2_height = int((conv1_height - conv2_params["kernel_size"])/conv2_params["strides"] + 1)

        if conv3_params is None:
            caps1_n_caps = int((caps1_n_maps *
                                int((conv2_width-conv_caps_params["kernel_size"])/conv_caps_params["strides"] + 1) *
                                int((conv2_height-conv_caps_params["kernel_size"])/conv_caps_params["strides"] + 1)))
        else:
            conv3_width = int((conv2_width - conv3_params["kernel_size"])/conv3_params["strides"] + 1)
            conv3_height = int((conv2_height-conv3_params["kernel_size"])/conv3_params["strides"] + 1)
            caps1_n_caps = int((caps1_n_maps *
                                int((conv3_width - conv_caps_params["kernel_size"]) / conv_caps_params["strides"] + 1) *
                                int((conv3_height - conv_caps_params["kernel_size"]) / conv_caps_params["strides"] + 1)))

        # create early conv layers
        if conv_batch_norm:
            conv1 = batch_norm_conv_layer(X, is_training, name='conv1', **conv1_params)
            conv2 = batch_norm_conv_layer(conv1, is_training, name='conv2', **conv2_params)
        else:
            conv1 = tf.layers.conv2d(X, name="conv1", **conv1_params)
            tf.summary.histogram('1st_conv_layer', conv1)
            conv2 = tf.layers.conv2d(conv1, name="conv2", **conv2_params)
            tf.summary.histogram('2nd_conv_layer', conv2)

        if conv3_params is not None:

            if conv_batch_norm:
                conv3 = batch_norm_conv_layer(conv2, is_training, name='conv3', **conv3_params)
            else:
                conv3 = tf.layers.conv2d(conv2, name="conv3", **conv3_params)
            tf.summary.histogram('3rd_conv_layer', conv3)

    with tf.name_scope('1st_caps'):

        if conv3_params is None:
            # create first capsule layer
            caps1_output, caps1_output_with_maps = primary_caps_layer(conv2, caps1_n_maps, caps1_n_caps, caps1_n_dims,
                                                                      conv_caps_params["kernel_size"],
                                                                      conv_caps_params["strides"],
                                                                      conv_padding=conv_caps_params['padding'],
                                                                      conv_activation=conv_caps_params['activation'],
                                                                      print_shapes=print_shapes)

        else:
            # create first capsule layer
            caps1_output, caps1_output_with_maps = primary_caps_layer(conv3, caps1_n_maps, caps1_n_caps, caps1_n_dims, conv_caps_params["kernel_size"], conv_caps_params["strides"], conv_padding=conv_caps_params['padding'], conv_activation=conv_caps_params['activation'], print_shapes=print_shapes)

        # display a histogram of primary capsule norms
        caps1_output_norms = safe_norm(caps1_output, axis=-1, keep_dims=False, name="primary_capsule_norms")
        tf.summary.histogram('Primary capsule norms', caps1_output_norms)


    ####################################################################################################################
    # Decode from or apply loss to first capsule layer if requested
    ####################################################################################################################


        if primary_caps_decoder:

            decoder_output_primary_caps, highest_norm_capsule = primary_capsule_reconstruction(shape_patch, y, caps1_output, caps1_output_norms,
                                                                                               primary_caps_decoder_n_hidden1,
                                                                                               primary_caps_decoder_n_hidden2,
                                                                                               primary_caps_decoder_n_output, is_training)
            print(decoder_output_primary_caps, shape_patch)

            primary_caps_reconstruction_loss = compute_reconstruction_loss(shape_patch, decoder_output_primary_caps)

            # training operation for this decoder (will not affect total loss)
            train_op_primary_decoder = tf.train.AdamOptimizer().minimize(primary_caps_reconstruction_loss,
                                                                         var_list=tf.get_collection(
                                                                             tf.GraphKeys.GLOBAL_VARIABLES,
                                                                             scope='primary_capsule_decoder'),
                                                                         name="train_op_primary_decoder")

        if do_primary_caps_loss:
            primary_caps_loss = compute_primary_caps_loss(y, caps1_output_with_maps, caps1_n_maps, m_plus_primary, m_minus_primary, lambda_primary, print_shapes=print_shapes)
        else:
            primary_caps_loss = 0

    ####################################################################################################################
    # From caps1 to caps2
    ####################################################################################################################

    with tf.name_scope('2nd_caps'):
        # it is all taken care of by the function
        caps2_output = primary_to_fc_caps_layer(X, caps1_output, caps1_n_caps, caps1_n_dims, caps2_n_caps, caps2_n_dims,
                                                rba_rounds=rba_rounds, print_shapes=print_shapes)

        # get norms of all capsules for the first simulus in the batch to vizualize them
        caps2_output_norm = tf.squeeze(safe_norm(caps2_output[0, :, :, :], axis=-2, keep_dims=False,
                                                 name="caps2_output_norm"))
        tf.summary.histogram('Output capsule norms', caps2_output_norm)


        ####################################################################################################################
        # Estimated class probabilities
        ####################################################################################################################


        y_pred = caps_prediction(caps2_output, n_labels=len(y.shape), print_shapes=print_shapes)  # get index of max probability


        ####################################################################################################################
        # Compute the margin loss
        ####################################################################################################################


        margin_loss = compute_margin_loss(y, caps2_output, caps2_n_caps, m_plus, m_minus, lambda_, print_shapes=print_shapes)


    ####################################################################################################################
    # N_shapes decoder, reconstruction & reconstruction error
    ####################################################################################################################

    with tf.name_scope('decoders'):
        # create the mask
        decoder_input_output_caps = create_masked_decoder_input(y, y_pred, caps2_output, caps2_n_caps, caps2_n_dims,
                                                                mask_with_labels, print_shapes=print_shapes)
        tf.summary.histogram('decoder_input_no_bn', decoder_input_output_caps)

        # compute n_shapes loss
        if do_n_shapes_loss:
            n_shapes_loss = compute_n_shapes_loss(decoder_input_output_caps, n_shapes_labels, n_shapes_max, print_shapes)
        else:
            n_shapes_loss = 0.

        if do_vernier_offset_loss:
            training_vernier_decoder_input = caps2_output[:, 0, 0, :, 0]
            training_vernier_loss, vernier_accuracy, vernier_logits = compute_vernier_offset_loss(training_vernier_decoder_input, vernier_offset_labels, print_shapes)
        else:
            training_vernier_loss = 0.

        # # batch_normalize input to decoder
        # decoder_input = tf.contrib.layers.batch_norm(decoder_input_output_caps, center=True, scale=True,
        #                                              is_training=is_training, scope='output_caps_decoder_input_bn')
        # tf.summary.histogram('decoder_input_bn', decoder_input_output_caps)

        # run decoder
        if decoder_batch_norm:
            decoder_output_output_caps = decoder_with_mask_batch_norm(decoder_input_output_caps, im_size[0]*im_size[1],
                                                                      output_caps_decoder_n_hidden1,
                                                                      output_caps_decoder_n_hidden2,
                                                                      phase=is_training,
                                                                      name='output_decoder')
        else:
            decoder_output_output_caps = decoder_with_mask(decoder_input=decoder_input_output_caps, output_width=im_size[1], output_height=im_size[0],
                                                           n_hidden1=output_caps_decoder_n_hidden1, n_hidden2=output_caps_decoder_n_hidden2,
                                                           n_hidden3=output_caps_decoder_n_hidden3, print_shapes=print_shapes,
                                                           **output_decoder_deconv_params)

        decoder_output_image_output_caps = tf.reshape(decoder_output_output_caps, [-1, im_size[0], im_size[1], 1])
        tf.summary.image('decoder_output', decoder_output_image_output_caps, 6)

        # reconstruction loss
        output_caps_reconstruction_loss, squared_differences = compute_reconstruction_loss(X, decoder_output_output_caps, loss_type=reconstruction_loss_type)


    ####################################################################################################################
    # Final loss, accuracy, training operations, init & saver
    ####################################################################################################################


    with tf.name_scope('total_loss'):
        loss = tf.add_n([alpha_margin * margin_loss,
                         alpha_reconstruction * output_caps_reconstruction_loss,
                         alpha_primary * primary_caps_loss,
                         alpha_n_shapes * n_shapes_loss,
                         alpha_vernier_offset * training_vernier_loss],
                         name="loss")

        tf.summary.scalar('total_loss', loss)

    with tf.name_scope('accuracy'):
        correct = tf.equal(y, y_pred, name="correct")
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
        tf.summary.scalar('accuracy', accuracy)

    # TRAINING OPERATIONS #

    optimizer = tf.train.AdamOptimizer(learning_rate=0.0005 )
    update_batch_norm_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # for batch norm
    loss_training_op = optimizer.minimize(loss, name="training_op")
    training_op = [loss_training_op, update_batch_norm_ops]

    return locals()
