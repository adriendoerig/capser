# a function form of capser to make param changes easy
from __future__ import division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
from capsule_functions import primary_caps_layer, primary_to_fc_caps_layer, \
    caps_prediction, compute_margin_loss, create_masked_decoder_input, \
    decoder_with_mask_batch_norm, primary_capsule_reconstruction, compute_reconstruction_loss, safe_norm


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


def capser_batch_norm_2_caps_layers(X, y, im_size, conv1_params, conv2_params, conv3_params,
                                    caps1_n_maps, caps1_n_dims, conv_caps_params,
                                    primary_caps_decoder_n_hidden1, primary_caps_decoder_n_hidden2, primary_caps_decoder_n_hidden3, primary_caps_decoder_n_output,
                                    caps2_n_caps, caps2_n_dims,
                                    m_plus, m_minus, lambda_, alpha,
                                    output_caps_decoder_n_hidden1, output_caps_decoder_n_hidden2, output_caps_decoder_n_hidden3, output_caps_n_output,
                                    is_training, mask_with_labels,
                                    primary_caps_decoder=False, shape_patch=0
                                    ):

    ####################################################################################################################
    # Early conv layers and first capsules
    ####################################################################################################################

    # batch_normalize input
    X = tf.contrib.layers.batch_norm(X, center=True, scale=True, is_training=is_training, scope='input_bn')
    if primary_caps_decoder:
        shape_patch = tf.contrib.layers.batch_norm(shape_patch, center=True, scale=True, is_training=is_training,
                                                   scope='shape_patch_bn')

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
    conv1 = batch_norm_conv_layer(X, is_training, name='conv1', **conv1_params)
    conv2 = batch_norm_conv_layer(conv1, is_training, name='conv2', **conv2_params)
    if conv3_params is None:
        # create first capsule layer
        caps1_output = primary_caps_layer(conv2, caps1_n_maps, caps1_n_caps, caps1_n_dims,
                                          conv_caps_params["kernel_size"], conv_caps_params["strides"],
                                          conv_padding='valid', conv_activation=tf.nn.elu, print_shapes=False)
    else:
        conv3 = batch_norm_conv_layer(conv2, is_training, name='conv3', activation=tf.nn.elu, **conv3_params)
        # create first capsule layer
        caps1_output = primary_caps_layer(conv3, caps1_n_maps, caps1_n_caps, caps1_n_dims,
                                          conv_caps_params["kernel_size"], conv_caps_params["stride"],
                                          conv_padding='valid',
                                          conv_activation=tf.nn.elu, print_shapes=False)
        caps1_output = tf.contrib.layers.batch_norm(caps1_output, center=True, scale=True, is_training=is_training,
                                                    scope='caps1_output_bn')
        tf.summary.histogram('caps_1_output_bn', caps1_output)

    # display a histogram of primary capsule norms
    caps1_output_norms = safe_norm(caps1_output, axis=-1, keep_dims=False, name="primary_capsule_norms")
    tf.summary.histogram('Primary capsule norms', caps1_output_norms[0, :])


    ####################################################################################################################
    # Decode from 1st capsule layer if requested
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



    ####################################################################################################################
    # From caps1 to caps2
    ####################################################################################################################

    # it is all taken care of by the function
    caps2_output = primary_to_fc_caps_layer(X, caps1_output, caps1_n_caps, caps1_n_dims, caps2_n_caps, caps2_n_dims,
                                            rba_rounds=2, print_shapes=False)

    # get norms to vizualize them
    caps2_output_norm = tf.squeeze(safe_norm(caps2_output[1, :, :, :], axis=-2, keep_dims=False,
                                            name="caps2_output_norm"))
    tf.summary.histogram('Output capsule norms', caps2_output_norm)

    ####################################################################################################################
    # Estimated class probabilities
    ####################################################################################################################

    y_pred = caps_prediction(caps2_output, print_shapes=False)  # get index of max probability

    ####################################################################################################################
    # Compute the margin loss
    ####################################################################################################################

    margin_loss = compute_margin_loss(y, caps2_output, caps2_n_caps, m_plus, m_minus, lambda_)

    ####################################################################################################################
    # Reconstruction & reconstruction error
    ####################################################################################################################

    # create the mask
    decoder_input_output_caps = create_masked_decoder_input(y, y_pred, caps2_output, caps2_n_caps, caps2_n_dims,
                                                            mask_with_labels, print_shapes=False)
    tf.summary.histogram('decoder_input_no_bn', decoder_input_output_caps)
    # # batch_normalize input to decoder
    # decoder_input = tf.contrib.layers.batch_norm(decoder_input_output_caps, center=True, scale=True,
    #                                              is_training=is_training, scope='output_caps_decoder_input_bn')
    # tf.summary.histogram('decoder_input_bn', decoder_input_output_caps)

    # run decoder
    if output_caps_decoder_n_hidden3 is None:
        decoder_output_output_caps = decoder_with_mask_batch_norm(decoder_input_output_caps,
                                                                  output_caps_decoder_n_hidden1,
                                                                  output_caps_decoder_n_hidden2,
                                                                  output_caps_n_output, phase=is_training,
                                                                  name='output_decoder')
    else:
        decoder_output_output_caps = decoder_with_mask_3layers_batch_norm(decoder_input_output_caps,
                                                                          output_caps_decoder_n_hidden1,
                                                                          output_caps_decoder_n_hidden2,
                                                                          output_caps_decoder_n_hidden3,
                                                                          output_caps_n_output, phase=is_training,
                                                                          name='output_decoder')

    decoder_output_image_output_caps = tf.reshape(decoder_output_output_caps, [-1, im_size[0], im_size[1], 1])
    tf.summary.image('decoder_output', decoder_output_image_output_caps, 6)

    # reconstruction loss
    output_caps_reconstruction_loss = compute_reconstruction_loss(X, decoder_output_output_caps)

    ####################################################################################################################
    # Final loss, accuracy, training operations, init & saver
    ####################################################################################################################

    # alpha = 0.0005  # * (60 * 128) / (im_size[0] * im_size[1])  # 0.0005 was good for 60*128 images

    with tf.name_scope('total_loss'):
        loss = tf.add(margin_loss, alpha * output_caps_reconstruction_loss, name="loss")
        tf.summary.scalar('total_loss', loss)

    with tf.name_scope('accuracy'):
        correct = tf.equal(y, y_pred, name="correct")
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
        tf.summary.scalar('accuracy', accuracy)

    # TRAINING OPERATIONS #

    optimizer = tf.train.AdamOptimizer()
    update_batch_norm_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # for batch norm
    loss_training_op = optimizer.minimize(loss, name="training_op")
    training_op = [loss_training_op, update_batch_norm_ops]

    return locals()
