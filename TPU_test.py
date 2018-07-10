import tensorflow as tf
import numpy as np
from batchMaker import StimMaker
import matplotlib.pyplot as plt

# Cloud TPU Cluster Resolver flags
tf.flags.DEFINE_string(
    "tpu", default=None,
    help="The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")
tf.flags.DEFINE_string(
    "tpu_zone", default=None,
    help="[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")
tf.flags.DEFINE_string(
    "gcp_project", default=None,
    help="[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

# Model specific parameters
tf.flags.DEFINE_string("data_dir", "",
                       "Path to directory containing the MNIST dataset")
tf.flags.DEFINE_string("model_dir", None, "Estimator model_dir")
tf.flags.DEFINE_integer("batch_size", 1024,
                        "Mini-batch size for the training. Note that this "
                        "is the global batch size and not the per-shard batch.")
tf.flags.DEFINE_integer("train_steps", 1000, "Total number of training steps.")
tf.flags.DEFINE_integer("eval_steps", 0,
                        "Total number of evaluation steps. If `0`, evaluation "
                        "after training is skipped.")
tf.flags.DEFINE_float("learning_rate", 0.05, "Learning rate.")

tf.flags.DEFINE_bool("use_tpu", True, "Use TPUs rather than plain CPUs")
tf.flags.DEFINE_integer("iterations", 50,
                        "Number of iterations per TPU training loop.")
tf.flags.DEFINE_integer("num_shards", 8, "Number of shards (TPU chips).")

FLAGS = tf.flags.FLAGS


def main(argv):
    del argv  # Unused.
    tf.logging.set_verbosity(tf.logging.INFO)

    im_size = (30, 60)  # IF USING THE DECONVOLUTION DECODER NEED TO BE EVEN NUMBERS (NB. this suddenly changed. before that, odd was needed... that's odd.)
    shape_size = 10  # size of a single shape in pixels
    bar_width = 1  # thickness of elements' bars
    shape_types = [1, 2, 9]  # see batchMaker.drawShape for number-shape correspondences
    group_last_shapes = 1  # attributes the same label to the last n shapeTypes
    batch_size = 10
    stim_maker = StimMaker(im_size, shape_size, bar_width)  # handles data generation

    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu,
        zone=FLAGS.tpu_zone,
        project=FLAGS.gcp_project
    )

    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        model_dir=FLAGS.model_dir,
        session_config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=True),
        tpu_config=tf.contrib.tpu.TPUConfig(FLAGS.iterations, FLAGS.num_shards),
    )

    estimator = tf.contrib.tpu.TPUEstimator(
        model_fn=model_fn,
        use_tpu=FLAGS.use_tpu,
        train_batch_size=FLAGS.batch_size,
        eval_batch_size=FLAGS.batch_size,
        params={"data_dir": FLAGS.data_dir},
        config=run_config)
    # TPUEstimator.train *requires* a max_steps argument.
    estimator.train(input_fn=input_fn(stim_maker, batch_size, shape_types, group_last_shapes), max_steps=FLAGS.train_steps)


def input_fn(stim_maker, batch_size, shape_types, group_last_shapes):

    def _fn():
        batch_data, batch_labels, vernier_offset_labels, n_elements = stim_maker.makeBatch(batch_size, shape_types, group_last_shapes)
        return batch_data, batch_labels
    return _fn()


def model_fn(images, labels, mode, use_tpu, params):
    """A simple CNN."""
    del params  # unused

    input_layer = tf.reshape(images, [-1, 28, 28, 1])
    conv1 = tf.layers.conv2d(
        inputs=input_layer, filters=32, kernel_size=[5, 5], padding="same",
        activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    conv2 = tf.layers.conv2d(
        inputs=pool1, filters=64, kernel_size=[5, 5],
        padding="same", activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=128, activation=tf.nn.relu)
    logits = tf.layers.dense(inputs=dense, units=3)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=3)

    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

    optimizer = tf.train.AdamOptimizer()
    if use_tpu:
        optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.contrib.tpu.TPUEstimatorSpec(mode=mode, loss=loss, train_op=optimizer.minimize(loss, tf.train.get_global_step()))

def metric_fn(labels, logits):
    accuracy = tf.metrics.accuracy(
        labels=labels, predictions=tf.argmax(logits, axis=1))
    return {"accuracy": accuracy}

if __name__ == "__main__":
    tf.app.run()