import numpy as np
from capser_model import capser_model
from make_tf_dataset import train_input_fn, test_input_fn
import logging
from parameters import *

def model_fn(features, labels, mode, params):


    # placeholders for input images and labels, and optional stuff
    # X = tf.placeholder(shape=[None, im_size[0], im_size[1], 1], dtype=tf.float32, name="X")
    X = features['X']
    x_image = tf.reshape(X, [-1, im_size[0], im_size[1], 1])
    tf.summary.image('input', x_image, 6)
    reconstruction_targets = features['reconstruction_targets']
    # reconstruction_targets = tf.placeholder(shape=[None, im_size[0], im_size[1], simultaneous_shapes], dtype=tf.float32,
    #                                         name="reconstruction_targets")
    if simultaneous_shapes > 1:
        y = tf.cast(features['y'], tf.int64)
        n_shapes = features['n_shapes']
        # y = tf.placeholder(shape=[None, simultaneous_shapes], dtype=tf.int64, name="y")
        # n_shapes = tf.placeholder_with_default(tf.zeros(shape=(params['batch_size'], simultaneous_shapes)),
        #                                        shape=[None, simultaneous_shapes], name="n_shapes_labels")
    else:
        # y = tf.placeholder(shape=[None], dtype=tf.int64, name="y")
        y = tf.cast(features['y'], tf.int64)
        n_shapes = tf.placeholder_with_default(tf.zeros(shape=(params['batch_size'])), shape=[None], name="n_shapes_labels")
    vernier_offsets = tf.placeholder_with_default(tf.zeros(shape=(params['batch_size'])), shape=[None],
                                                  name="vernier_offset_labels")

    # create a placeholder that will tell the program whether to use the true or the predicted labels
    mask_with_labels = tf.placeholder_with_default(features['mask_with_labels'], shape=(), name="mask_with_labels")
    # placeholder specifying if training or not (for batch normalization)
    is_training = tf.placeholder_with_default(features['is_training'], shape=(), name='is_training')
    is_training = features['is_training']

    capser = capser_model(X, y, reconstruction_targets, im_size, conv1_params, conv2_params, conv3_params,
                          caps1_n_maps, caps1_n_dims, conv_caps_params,
                          primary_caps_decoder_n_hidden1, primary_caps_decoder_n_hidden2,
                          primary_caps_decoder_n_hidden3, primary_caps_decoder_n_output,
                          caps2_n_caps, caps2_n_dims, rba_rounds,
                          m_plus, m_minus, lambda_, alpha_margin,
                          m_plus_primary, m_minus_primary, lambda_primary, alpha_primary,
                          output_caps_decoder_n_hidden1, output_caps_decoder_n_hidden2, output_caps_decoder_n_hidden3,
                          reconstruction_loss_type, alpha_reconstruction, vernier_gain,
                          is_training, mask_with_labels,
                          primary_caps_decoder, primary_caps_loss, n_shapes_loss, vernier_offset_loss,
                          n_shapes, max_cols * max_rows, alpha_n_shapes,
                          vernier_offsets, alpha_vernier_offset,
                          0, conv_batch_norm, decoder_batch_norm,
                          **output_decoder_deconv_params)

    # op to train all networks
    train_op = capser["loss_training_op"]

    # Define the evaluation metrics,
    # in this case the classification accuracy.
    metrics = \
        {
            "accuracy": tf.metrics.accuracy(labels, capser['y_pred'])
        }

    # Wrap all of this in an EstimatorSpec.
    spec = tf.estimator.EstimatorSpec(
        mode=mode,
        loss=capser["loss"],
        train_op=train_op,
        eval_metric_ops={})

    return spec

