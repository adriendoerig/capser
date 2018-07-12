import logging
from tensorflow.contrib.tpu.python.tpu import tpu_config, tpu_estimator, tpu_optimizer
from tensorflow.contrib.cluster_resolver import TPUClusterResolver
from capser_7_model_fn import *
from capser_7_input_fn import *
import subprocess


class FLAGS(object):
    use_tpu = True
    tpu_name = None
    # Use a local temporary path for the `model_dir`
    model_dir = LOGDIR
    # saves a checkpoint each save_checkpoints_secs seconds
    save_checkpoints_secs = 1000
    # number of steps which must have run before showing summaries
    save_summary_steps = 100
    # Number of training steps to run on the Cloud TPU before returning control.
    iterations = 1000
    # A single Cloud TPU has 8 shards.
    num_shards = 8

if FLAGS.use_tpu:
    my_project_name = subprocess.check_output(['gcloud', 'config', 'get-value', 'project'])
    my_zone = subprocess.check_output(['gcloud', 'config', 'get-value', 'compute/zone'])
    cluster_resolver = TPUClusterResolver(
        tpu=[FLAGS.tpu_name],
        zone=my_zone,
        project=my_project_name)
    master = TPUClusterResolver.get_master()
else:
    master = ''

my_tpu_run_config = tpu_config.RunConfig(
    master=master,
    evaluation_master=master,
    model_dir=FLAGS.model_dir,
    save_checkpoints_secs=FLAGS.save_checkpoints_secs,
    save_summary_steps=FLAGS.save_summary_steps,
    session_config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True),
    tpu_config=tpu_config.TPUConfig(iterations_per_loop=FLAGS.iterations, num_shards=FLAGS.num_shards),
)


# create estimator for model (the model is described in capser_7_model_fn)
capser = tpu_estimator.TPUEstimator(model_fn=model_fn_tpu,
                                    config=my_tpu_run_config,
                                    use_tpu=FLAGS.use_tpu,
                                    params={'model_batch_size': batch_size})

# train model
logging.getLogger().setLevel(logging.INFO)  # to show info about training progress
capser.train(input_fn=train_input_fn_tpu, steps=n_steps)
