from capser_model import capser_model
from parameters import *
from time import time
from capsule_functions import *

def model_fn_temp(features, labels, mode, params):

    print('USING MODEL_FN_TEMP')
    using_TPUEstimator = False
    simultaneous_shapes = 2
    n_shapes_loss = False

    X = features['X']
    x_image = tf.reshape(X, [params['model_batch_size'], im_size[0], im_size[1], 1])
    tf.summary.image('input', x_image, 6)
    reconstruction_targets = features['reconstruction_targets']
    vernier_offsets = features['vernier_offsets']

    if simultaneous_shapes > 1:
        y = tf.cast(features['y'], tf.int64)
        n_shapes = tf.placeholder_with_default(tf.zeros(shape=(params['model_batch_size'], simultaneous_shapes)), shape=(params['model_batch_size'], simultaneous_shapes))
    else:
        y = tf.cast(features['y'], tf.int64)
        n_shapes = tf.placeholder_with_default(tf.zeros(shape=(params['model_batch_size'])), shape=[None], name="n_shapes_labels")

    # tell the program whether to use the true or the predicted labels (the placeholder is needed to have a bool in tf).
    mask_with_labels = tf.placeholder_with_default(features['mask_with_labels'], shape=(), name="mask_with_labels")
    # boolean specifying if training or not (for batch normalization)
    is_training = tf.placeholder_with_default(features['is_training'], shape=(), name='is_training')
    is_training = True

    print_shapes = False  # to print the size of each layer during graph construction

    ####################################################################################################################
    # Early conv layers and first capsules
    ####################################################################################################################

    # maybe batch-norm the input?
    # X = tf.contrib.layers.batch_norm(X, center=True, scale=True, is_training=is_training, scope='input_bn')

    with tf.name_scope('0_early_conv_layers'):
        # sizes, etc.
        conv1_width = int((im_size[0] - conv1_params["kernel_size"]) / conv1_params["strides"] + 1)
        conv1_height = int((im_size[1] - conv1_params["kernel_size"]) / conv1_params["strides"] + 1)

        if conv2_params is not None:
            conv2_width = int((conv1_width - conv2_params["kernel_size"]) / conv2_params["strides"] + 1)
            conv2_height = int((conv1_height - conv2_params["kernel_size"]) / conv2_params["strides"] + 1)

            if conv3_params is None:
                caps1_n_caps = int((caps1_n_maps *
                                    int((conv2_width - conv_caps_params["kernel_size"]) / conv_caps_params[
                                        "strides"] + 1) *
                                    int((conv2_height - conv_caps_params["kernel_size"]) / conv_caps_params[
                                        "strides"] + 1)))
            else:
                conv3_width = int((conv2_width - conv3_params["kernel_size"]) / conv3_params["strides"] + 1)
                conv3_height = int((conv2_height - conv3_params["kernel_size"]) / conv3_params["strides"] + 1)
                caps1_n_caps = int((caps1_n_maps *
                                    int((conv3_width - conv_caps_params["kernel_size"]) / conv_caps_params[
                                        "strides"] + 1) *
                                    int((conv3_height - conv_caps_params["kernel_size"]) / conv_caps_params[
                                        "strides"] + 1)))
        else:
            caps1_n_caps = int((caps1_n_maps *
                                int((conv1_width - conv_caps_params["kernel_size"]) / conv_caps_params["strides"] + 1) *
                                int((conv1_height - conv_caps_params["kernel_size"]) / conv_caps_params[
                                    "strides"] + 1)))

        # create early conv layers
        if conv_batch_norm:
            print('Using conv_batch_norm: layer 1')
            conv1 = tf.layers.conv2d(X, name="conv1", use_bias=False, **conv1_params)
            conv1 = tf.layers.batch_normalization(conv1, training=is_training, name='conv1_bn')
            conv1 = tf.nn.elu(conv1, name='conv1_activation')
            if conv2_params is not None:
                print('Using conv_batch_norm: layer 2')
                conv2 = tf.layers.conv2d(conv1, name="conv2", use_bias=False, **conv2_params)
                conv2 = tf.layers.batch_normalization(conv2, training=is_training, name='conv2_bn')
                conv2 = tf.nn.elu(conv2, name='conv2_activation')
        else:
            conv1 = tf.layers.conv2d(X, name="conv1", **conv1_params)
            if not using_TPUEstimator:
                tf.summary.histogram('1st_conv_layer', conv1)
            if conv2_params is not None:
                conv2 = tf.layers.conv2d(conv1, name="conv2", **conv2_params)
                if not using_TPUEstimator:
                    tf.summary.histogram('2nd_conv_layer', conv2)

        if conv3_params is not None:

            if conv_batch_norm:
                conv3 = batch_norm_conv_layer(conv2, is_training, name='conv3', **conv3_params)
            else:
                conv3 = tf.layers.conv2d(conv2, name="conv3", **conv3_params)
                if not using_TPUEstimator:
                    tf.summary.histogram('3rd_conv_layer', conv3)

    with tf.name_scope('1st_caps'):

        if conv2_params is None:
            caps1_output, caps1_output_with_maps = primary_caps_layer(conv1, caps1_n_maps, caps1_n_caps, caps1_n_dims,
                                                                      conv_caps_params["kernel_size"],
                                                                      conv_caps_params["strides"],
                                                                      conv_padding=conv_caps_params['padding'],
                                                                      conv_activation=conv_caps_params['activation'],
                                                                      print_shapes=print_shapes)

        else:
            if conv3_params is None:
                # create first capsule layer
                caps1_output, caps1_output_with_maps = primary_caps_layer(conv2, caps1_n_maps, caps1_n_caps,
                                                                          caps1_n_dims,
                                                                          conv_caps_params["kernel_size"],
                                                                          conv_caps_params["strides"],
                                                                          conv_padding=conv_caps_params['padding'],
                                                                          conv_activation=conv_caps_params[
                                                                              'activation'],
                                                                          print_shapes=print_shapes)

            else:
                # create first capsule layer
                caps1_output, caps1_output_with_maps = primary_caps_layer(conv3, caps1_n_maps, caps1_n_caps,
                                                                          caps1_n_dims,
                                                                          conv_caps_params["kernel_size"],
                                                                          conv_caps_params["strides"],
                                                                          conv_padding=conv_caps_params['padding'],
                                                                          conv_activation=conv_caps_params[
                                                                              'activation'],
                                                                          print_shapes=print_shapes)

        # display a histogram of primary capsule norms
        caps1_output_norms = safe_norm(caps1_output, axis=-1, keep_dims=False, name="primary_capsule_norms")
        if not using_TPUEstimator:
            tf.summary.histogram('Primary capsule norms', caps1_output_norms)



    ####################################################################################################################
    # From caps1 to caps2
    ####################################################################################################################

    with tf.name_scope('2nd_caps'):
        # it is all taken care of by the function
        if using_TPUEstimator:
            caps2_output = primary_to_fc_caps_layer_tpu(X, caps1_output, caps1_n_caps, caps1_n_dims, caps2_n_caps,
                                                        caps2_n_dims,
                                                        rba_rounds=rba_rounds, print_shapes=print_shapes)
        else:
            caps2_output = primary_to_fc_caps_layer(X, caps1_output, caps1_n_caps, caps1_n_dims, caps2_n_caps,
                                                    caps2_n_dims,
                                                    rba_rounds=rba_rounds, print_shapes=print_shapes)

        # get norms of all capsules for the first simulus in the batch to vizualize them
        caps2_output_norm = tf.squeeze(safe_norm(caps2_output[0, :, :, :], axis=-2, keep_dims=False,
                                                 name="caps2_output_norm"))
        if not using_TPUEstimator:
            tf.summary.histogram('Output capsule norms', caps2_output_norm)

        ####################################################################################################################
        # Estimated class probabilities
        ####################################################################################################################

        y_pred = caps_prediction(caps2_output, n_labels=len(y.shape),
                                 print_shapes=print_shapes)  # get index of max probability

        ####################################################################################################################
        # Compute the margin loss
        ####################################################################################################################

        margin_loss = compute_margin_loss(y, caps2_output, caps2_n_caps, m_plus, m_minus, lambda_,
                                          print_shapes=print_shapes)

    ####################################################################################################################
    # N_shapes decoder, reconstruction & reconstruction error
    ####################################################################################################################

    with tf.name_scope('decoders'):

        # compute n_shapes loss
        with tf.name_scope('n_shape_loss'):
            if n_shapes_loss:
                # create the mask
                decoder_input_output_caps = create_masked_decoder_input(y, y_pred, caps2_output, caps2_n_caps,
                                                                        caps2_n_dims, mask_with_labels,
                                                                        print_shapes=print_shapes)
                if not using_TPUEstimator:
                    tf.summary.histogram('decoder_input_no_bn', decoder_input_output_caps)

                n_shapes_loss = compute_n_shapes_loss(decoder_input_output_caps, n_shapes, max_cols * max_rows,
                                                      print_shapes)
            else:
                n_shapes_loss = 0.

        # vernier offset loss
        with tf.name_scope('vernier_offset_loss'):
            if vernier_offset_loss:
                training_vernier_decoder_input = caps2_output[:, 0, 0, :, 0]  # decode from vernier capsule
                training_vernier_loss, vernier_accuracy, vernier_logits = compute_vernier_offset_loss(
                    training_vernier_decoder_input, vernier_offsets, print_shapes)
            else:
                training_vernier_loss = 0.

        with tf.name_scope('reconstruction_loss'):
            # run decoder to reconstruct full image at once (used when simultaneous shapes == 1)
            simultaneous_shapes = tf.shape(reconstruction_targets)[-1]
            if simultaneous_shapes == 1:
                if decoder_batch_norm:
                    decoder_output_output_caps = decoder_with_mask_batch_norm(decoder_input_output_caps,
                                                                              im_size[0] * im_size[1],
                                                                              output_caps_decoder_n_hidden1,
                                                                              output_caps_decoder_n_hidden2,
                                                                              phase=is_training,
                                                                              name='output_decoder')
                else:
                    decoder_output_output_caps = decoder_with_mask(decoder_input=decoder_input_output_caps,
                                                                   output_width=im_size[1], output_height=im_size[0],
                                                                   n_hidden1=output_caps_decoder_n_hidden1,
                                                                   n_hidden2=output_caps_decoder_n_hidden2,
                                                                   n_hidden3=output_caps_decoder_n_hidden3,
                                                                   print_shapes=print_shapes,
                                                                   **output_decoder_deconv_params)

                decoder_output_image_output_caps = tf.reshape(decoder_output_output_caps,
                                                              [-1, im_size[0], im_size[1], 1])
                if not using_TPUEstimator:
                    tf.summary.image('decoder_output', decoder_output_image_output_caps, 6)

                # reconstruction loss
                output_caps_reconstruction_loss, squared_differences = compute_reconstruction_loss(X,
                                                                                                   decoder_output_output_caps,
                                                                                                   loss_type=reconstruction_loss_type)

            # if we are in simultaneous_shapes > 1 mode, we reconstruct from each capsule separately.
            # NOTE: for now, let's assume we have 2 shapes. It would be better to use range(simultaneous_shapes),
            # but this raises an error: TypeError: 'Tensor' object cannot be interpreted as an integer
            else:

                decoder_outputs = []

                for this_shape in range(2):
                    with tf.variable_scope('shape_' + str(this_shape)):
                        this_masked_output = create_masked_decoder_input(y[:, this_shape], y_pred[:, this_shape],
                                                                         caps2_output, caps2_n_caps, caps2_n_dims,
                                                                         mask_with_labels, print_shapes=print_shapes)

                        if decoder_batch_norm:
                            decoder_outputs.append(
                                decoder_with_mask_batch_norm(this_masked_output, im_size[0] * im_size[1],
                                                             output_caps_decoder_n_hidden1,
                                                             output_caps_decoder_n_hidden2, phase=is_training,
                                                             name='output_decoder'))
                        else:
                            decoder_outputs.append(
                                decoder_with_mask(decoder_input=this_masked_output, output_width=im_size[1],
                                                  output_height=im_size[0], n_hidden1=output_caps_decoder_n_hidden1,
                                                  n_hidden2=output_caps_decoder_n_hidden2,
                                                  n_hidden3=output_caps_decoder_n_hidden3, print_shapes=print_shapes,
                                                  **output_decoder_deconv_params))

                decoder_outputs = tf.stack(decoder_outputs)
                decoder_outputs = tf.transpose(decoder_outputs, [1, 2, 0])
                if print_shapes:
                    print('shape of decoder_outputs: ' + str(decoder_outputs))

                # reconstruction loss
                output_caps_reconstruction_loss, squared_differences = 0, 0
                for this_shape in range(2):
                    with tf.variable_scope('shape_' + str(this_shape)):
                        # set gain > 1 to pump up the reconstruction error when a vernier is present (because there are so few pixels in the vernier)
                        gain = tf.ones_like(y[:, this_shape])
                        mask = tf.cast(tf.equal(y[:, this_shape], tf.zeros_like(y[:, this_shape])),
                                       tf.int64)  # find verniers
                        gain = tf.cast(gain * mask * vernier_gain, tf.float32)

                        this_output_caps_reconstruction_loss, this_squared_differences = compute_reconstruction_loss(
                            reconstruction_targets[:, :, :, this_shape], decoder_outputs[:, :, this_shape],
                            loss_type=reconstruction_loss_type, gain=gain, no_tensorboard=True)
                        output_caps_reconstruction_loss = output_caps_reconstruction_loss + this_output_caps_reconstruction_loss
                        squared_differences = squared_differences + this_squared_differences

                if not using_TPUEstimator:
                    tf.summary.scalar('reconstruction_loss_sum', output_caps_reconstruction_loss)

                # make an rgb tf.summary image. Note: there's sum fucked up dimension tweaking but it works.
                color_masks = np.array([[121, 199, 83],  # 0: vernier, green
                                        [220, 76, 70],  # 1: red
                                        [79, 132, 196]])  # 3: blue
                color_masks = np.expand_dims(color_masks, axis=1)
                color_masks = np.expand_dims(color_masks, axis=1)

                decoder_output_images = tf.reshape(decoder_outputs, [-1, im_size[0], im_size[1], simultaneous_shapes])
                decoder_output_images_rgb_0 = tf.image.grayscale_to_rgb(
                    tf.expand_dims(decoder_output_images[:, :, :, 0], axis=-1)) * color_masks[0, :, :, :]
                decoder_output_images_rgb_1 = tf.image.grayscale_to_rgb(
                    tf.expand_dims(decoder_output_images[:, :, :, 1], axis=-1)) * color_masks[1, :, :, :]

                decoder_output_images_sum = decoder_output_images_rgb_0 + decoder_output_images_rgb_1
                # display the summed output image
                if not using_TPUEstimator:
                    tf.summary.image('decoder_output', decoder_output_images_sum, 6)

    ####################################################################################################################
    # Final loss, accuracy, training operations, init & saver
    ####################################################################################################################

    with tf.name_scope('total_loss'):

        loss = tf.add_n([alpha_margin * margin_loss,
                         alpha_reconstruction * output_caps_reconstruction_loss,
                         alpha_n_shapes * n_shapes_loss,
                         alpha_vernier_offset * training_vernier_loss],
                        name="loss")

        if not using_TPUEstimator:
            tf.summary.scalar('total_loss', loss)

    if not using_TPUEstimator:
        with tf.name_scope('accuracy'):
            # the reshape is in case simulataneous_shapes > 1 (in this case we need to reorder y and y_pred in ascending order to have matching labels for shape 1 & 2)
            y_sorted = tf.contrib.framework.sort(y, axis=-1, direction='ASCENDING', name='y_sorted')
            y_pred_sorted = tf.contrib.framework.sort(y_pred, axis=-1, direction='ASCENDING',
                                                              name='y_pred_sorted')
            correct = tf.equal(y_sorted, y_pred_sorted, name="correct")
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
            tf.summary.scalar('accuracy', accuracy)

    # to write summaries during and prediction too
    eval_summary_hook = tf.train.SummarySaverHook(save_steps=25, output_dir=checkpoint_path + '/eval',
                                                  summary_op=tf.summary.merge_all())
    pred_summary_hook = tf.train.SummarySaverHook(save_steps=1, output_dir=checkpoint_path + '/pred-' + str(time()),
                                                  summary_op=tf.summary.merge_all())


    # Wrap all of this in an EstimatorSpec.
    if mode == tf.estimator.ModeKeys.PREDICT:
        # the following line is
        predictions = {'vernier_accuracy': tf.tile(tf.expand_dims(vernier_accuracy, -1), [batch_size_per_shard]), 'reconstructions': decoder_output_images_sum, 'vernier_offsets': vernier_offsets}
        spec = tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions)#,
                                          # prediction_hooks=[pred_summary_hook])    # to write summaries during prediction too)
        return spec

    else:

        # TRAINING OPERATIONS #

        if using_TPUEstimator:
            optimizer = tpu_optimizer.CrossShardOptimizer(optimizer)
        else:
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        if decoder_batch_norm is True or conv_batch_norm is True:
            update_batch_norm_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            # with tf.control_dependencies(update_batch_norm_ops):
            training_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step(), name="training_op")

            spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=training_op,
                eval_metric_ops={},
                evaluation_hooks=[eval_summary_hook])  # to write summaries during evaluatino too

            return spec
        else:
            training_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step(), name="training_op")

            spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=training_op,
                eval_metric_ops={},
                evaluation_hooks=[eval_summary_hook])    # to write summaries during evaluatino too


            return spec