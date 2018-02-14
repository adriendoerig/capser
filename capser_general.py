# a function form of capser to make param changes easy
from __future__ import division, print_function, unicode_literals
import tensorflow as tf
from capsule_functions import primary_caps_layer, primary_to_fc_caps_layer, \
    caps_prediction, compute_margin_loss, create_masked_decoder_input, \
    decoder_with_mask, decoder_with_mask_3layers, compute_reconstruction_loss, safe_norm


def capser_general_2_caps_layers(X, y, im_size, conv1_params, conv2_params, conv3_params,
                                 caps1_n_maps, caps1_n_dims, conv_caps_params,
                                 caps2_n_caps, caps2_n_dims,
                                 m_plus, m_minus, lambda_, alpha,
                                 n_hidden1, n_hidden2, n_hidden3, n_output,
                                 mask_with_labels, MODEL_NAME=''
                                ):


    ####################################################################################################################
    # Early conv layers and first capsules
    ####################################################################################################################


    # sizes, etc.
    conv1_width  = int((im_size[0]  -conv1_params["kernel_size"])/conv1_params["strides"] + 1)
    conv1_height = int((im_size[1]  -conv1_params["kernel_size"])/conv1_params["strides"] + 1)
    conv2_width  = int((conv1_width -conv2_params["kernel_size"])/conv2_params["strides"] + 1)
    conv2_height = int((conv1_height-conv2_params["kernel_size"])/conv2_params["strides"] + 1)

    if conv3_params == None:
        caps1_n_caps = int((caps1_n_maps * int((conv2_width-conv_caps_params["kernel_size"])/conv_caps_params["strides"] + 1) *
                                           int((conv2_height-conv_caps_params["kernel_size"])/conv_caps_params["strides"]+ 1)))
    else:
        conv3_width = int((conv2_width -conv3_params["kernel_size"])/conv3_params["strides"] + 1)
        conv3_height = int((conv2_height-conv3_params["kernel_size"])/conv3_params["strides"] + 1)
        caps1_n_caps = int((caps1_n_maps * int((conv3_width - conv_caps_params["kernel_size"]) / conv_caps_params["strides"] + 1) *
                                           int((conv3_height - conv_caps_params["kernel_size"]) / conv_caps_params["strides"] + 1)))

    # create early conv layers
    conv1 = tf.layers.conv2d(X, name="conv1", **conv1_params)  # ** means that conv1_params is a dict {param_name:param_value}
    tf.summary.histogram('1st_conv_layer', conv1)
    conv2 = tf.layers.conv2d(conv1, name="conv2", **conv2_params)  # ** means that conv1_params is a dict {param_name:param_value}
    tf.summary.histogram('2nd_conv_layer', conv2)
    if conv3_params == None:
        # create first capsule layer
        caps1_output = primary_caps_layer(conv2, caps1_n_maps, caps1_n_caps, caps1_n_dims,
                                          conv_caps_params["kernel_size"], conv_caps_params["strides"], conv_padding='valid',
                                          conv_activation=tf.nn.relu, print_shapes=False)
    else:
        conv3 = tf.layers.conv2d(conv2, name="conv3", **conv3_params)  # ** means that conv1_params is a dict {param_name:param_value}
        tf.summary.histogram('3rd_conv_layer', conv3)
        # create first capsule layer
        caps1_output = primary_caps_layer(conv3, caps1_n_maps, caps1_n_caps, caps1_n_dims,
                                          conv_caps_params["kernel_size"], conv_caps_params["stride"],
                                          conv_padding='valid',
                                          conv_activation=tf.nn.relu, print_shapes=False)


    ########################################################################################################################
    # From caps1 to caps2
    ########################################################################################################################


    # it is all taken care of by the function
    caps2_output = primary_to_fc_caps_layer(X, caps1_output, caps1_n_caps, caps1_n_dims, caps2_n_caps, caps2_n_dims,
                                            rba_rounds=3, print_shapes=False)


    ########################################################################################################################
    # Estimated class probabilities
    ########################################################################################################################


    y_pred = caps_prediction(caps2_output, print_shapes=False)# get index of max probability

    # get norms to vizualize them
    caps_output_norm = tf.squeeze(safe_norm(caps2_output[1, :, :, :], axis=-2, keep_dims=False,
                                            name="caps2_output_norm"))
    tf.summary.histogram('Output capsule norms', caps_output_norm)

    ########################################################################################################################
    # Compute the margin loss
    ########################################################################################################################


    # parameters for the margin loss
    margin_loss = compute_margin_loss(y, caps2_output, caps2_n_caps, m_plus, m_minus, lambda_)


    ########################################################################################################################
    # Reconstruction & reconstruction error
    ########################################################################################################################

    # create the mask
    decoder_input = create_masked_decoder_input(y, y_pred, caps2_output, caps2_n_caps, caps2_n_dims,
                                                mask_with_labels, print_shapes=False)
    # run decoder
    if n_hidden3 == None:
        decoder_output = decoder_with_mask(decoder_input, n_hidden1, n_hidden2, n_output)
    else:
        decoder_output = decoder_with_mask_3layers(decoder_input, n_hidden1, n_hidden2, n_hidden3, n_output)

    decoder_output_image = tf.reshape(decoder_output,[-1, im_size[0], im_size[1],1])
    tf.summary.image('decoder_output',decoder_output_image,6)

    # reconstructon loss
    reconstruction_loss = compute_reconstruction_loss(X,decoder_output)

    ####################################################################################################################
    # Final loss, accuracy, training operations, init & saver
    ####################################################################################################################


    # alpha = 0.0005 #* (60 * 128) / (im_size[0] * im_size[1])  # 0.0005 was good for 60*128 images

    with tf.name_scope('total_loss'):
        loss = tf.add(margin_loss, alpha * reconstruction_loss, name="loss")
        tf.summary.scalar('total_loss', loss)

    with tf.name_scope('accuracy'):
        correct = tf.equal(y, y_pred, name="correct")
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
        tf.summary.scalar('accuracy', accuracy)

    ### TRAINING OPERATIONS ###

    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss, name="training_op")

    all_variables = locals()

    return all_variables