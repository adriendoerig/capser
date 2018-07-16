from capser_model import capser_model
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from parameters import *

def model_fn(features, labels, mode, params):


    X = features['X']
    x_image = tf.reshape(X, [params['model_batch_size'], im_size[0], im_size[1], 1])
    tf.summary.image('input', x_image, 6)
    reconstruction_targets = features['reconstruction_targets']

    if simultaneous_shapes > 1:
        y = tf.cast(features['y'], tf.int64)
        n_shapes = features['n_shapes']
    else:
        y = tf.cast(features['y'], tf.int64)
        n_shapes = tf.placeholder_with_default(tf.zeros(shape=(params['model_batch_size'])), shape=[None], name="n_shapes_labels")
    vernier_offsets = tf.placeholder_with_default(tf.zeros(shape=(params['model_batch_size'])), shape=[None],
                                                  name="vernier_offset_labels")

    # tell the program whether to use the true or the predicted labels (the placeholder is needed to have a bool in tf).
    mask_with_labels = tf.placeholder_with_default(features['mask_with_labels'], shape=(), name="mask_with_labels")
    # boolean specifying if training or not (for batch normalization)
    is_training = tf.placeholder_with_default(features['is_training'], shape=(), name='is_training')

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
    # metrics = \
        # {
        #     "accuracy": tf.metrics.accuracy(labels, capser['y_pred'])
        # }

    # Wrap all of this in an EstimatorSpec.
    spec = tf.estimator.EstimatorSpec(
        mode=mode,
        loss=capser["loss"],
        train_op=train_op,
        eval_metric_ops={})

    return spec


def model_fn_tpu(features, labels, mode, params):


    X = features['X']
    print(X)
    print(X)
    print(X)
    print(X)
    print(X)
    print(X)
    print(X)
    print(X)
    print(X)
    print(X)
    # x_image = tf.reshape(X, [params['model_batch_size'], im_size[0], im_size[1], 1])
    #tf.summary.image('input', x_image, 6)
    reconstruction_targets = features['reconstruction_targets']

    if simultaneous_shapes > 1:
        y = tf.cast(features['y'], tf.int64)
        n_shapes = features['n_shapes']
    else:
        y = tf.cast(features['y'], tf.int64)
        n_shapes = tf.placeholder_with_default(tf.zeros(shape=(params['model_batch_size'])), shape=[params['model_batch_size']], name="n_shapes_labels")
    vernier_offsets = tf.placeholder_with_default(tf.zeros(shape=(params['model_batch_size'])), shape=[params['model_batch_size']],
                                                  name="vernier_offset_labels")

    # tell the program whether to use the true or the predicted labels (the placeholder is needed to have a bool in tf).
    mask_with_labels = tf.placeholder_with_default(True, shape=(), name="mask_with_labels")
    # boolean specifying if training or not (for batch normalization)
    is_training = tf.placeholder_with_default(True, shape=(), name='is_training')

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
                          using_TPUEstimator=True,
                          **output_decoder_deconv_params)

    # op to train all networks
    train_op = capser["loss_training_op"]

    # Define the evaluation metrics,
    # in this case the classification accuracy.
    # def my_metric_fn(labels, predictions):
    #     return {'accuracy': tf.metrics.accuracy(labels, predictions)}

    # Wrap all of this in an EstimatorSpec.
    spec = tpu_estimator.TPUEstimatorSpec(
        mode=mode,
        loss=capser["loss"],
        train_op=train_op)

    return spec

