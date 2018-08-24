import tensorflow as tf
from image_decoder_parameters import im_size, batch_size


model_dir = './image_decoder'

def image_decoder_model_fn(features, labels, mode, params):

    conv1_params = {"filters": 16,
                    "kernel_size": 5,
                    "strides": 1,
                    "padding": "valid",
                    "activation": tf.nn.relu,
                    }
    n_hidden1 = 512

    X = tf.convert_to_tensor(features['X'])
    y = tf.one_hot(tf.cast(features['y'], tf.int64), 2)

    x_image = tf.reshape(X, [batch_size, im_size[0], im_size[1], 3])
    tf.summary.image('input_image', x_image)

    with tf.name_scope('conv1'):
        conv1 = tf.layers.conv2d(X, name="conv1", **conv1_params)
        conv1_flat = tf.layers.flatten(conv1, name='conv1_flat')
    fc1 = tf.layers.dense(conv1_flat, n_hidden1, tf.nn.relu, name='fc1')
    fc2 = tf.layers.dense(fc1, 2, tf.nn.relu, name='fc2')
    logits = tf.nn.softmax(fc2, name='logits')
    y_pred = tf.argmax(logits, axis=-1, name="y_proba_argmax")

    vernier_accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred, tf.argmax(y, axis=-1)), dtype=tf.float32), name='vernier_accuracy')
    tf.summary.scalar('vernier_accuracy', vernier_accuracy)

    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:

        x_ent = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y)
        loss = tf.reduce_mean(x_ent, name='loss')

        optimizer = tf.train.AdamOptimizer()
        training_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step(), name="training_op")

        spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=training_op,
            eval_metric_ops={})

        return spec

    else:

        pred_summary_hook = tf.train.SummarySaverHook(save_steps=1, output_dir=model_dir + '/pred',
                                                      summary_op=tf.summary.merge_all())
        predictions = {'vernier_accuracy': tf.tile(tf.expand_dims(vernier_accuracy, axis=-1), [64])} # inelegant, but we must return a tensor with batch_size entries...
        spec = tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions,
                                          prediction_hooks=[pred_summary_hook])

        return spec