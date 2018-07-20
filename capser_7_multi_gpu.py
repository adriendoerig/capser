from capser_7_model_fn import *
from make_tf_dataset import *
from tensorflow.python import debug as tf_debug
from tensorflow.python.training import basic_session_run_hooks
from tensorboard.plugins.beholder import Beholder
from tensorboard.plugins.beholder import BeholderHook
import logging

# reproducibility
tf.reset_default_graph()
np.random.seed(42)
tf.set_random_seed(42)

# directory management
print('#################### MODEL_NAME_VERSION: ' + MODEL_NAME + '_' + str(version) + ' ####################')

if version_to_restore is None:
    save_params()

# create tfRecord data files if needed, and have a look at
if create_new_train_set:
    make_multi_shape_tfRecords(stim_maker, shape_types, n_train_samples, train_data_path)
if create_new_test_sets:
    for category in test_stimuli.keys():
        make_config_tfRecords(stim_maker, test_stimuli[category], n_test_samples, os.path.join(data_path, "test_" + category + ".tfrecords"))
if check_data is not None:
    show_data(check_data, type='multi_shape')

# create estimator for model (the model is described in capser_7_model_fn)
capser = tf.estimator.Estimator(model_fn=model_fn, params={'model_batch_size': batch_size}, model_dir=LOGDIR)

# train model
logging.getLogger().setLevel(logging.INFO)  # to show info about training progress
metadata_hook = basic_session_run_hooks.ProfilerHook(output_dir=LOGDIR, save_steps=1000)  # to get metadata (e.g. how much time is spent loading data, or processing it on the GPU, etc)
debug_hook = tf_debug.TensorBoardDebugHook('localhost:7000')  # for tensorboard debugger
beholder = Beholder(LOGDIR)
beholder_hook = BeholderHook(LOGDIR)

train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=n_steps, hooks=[metadata_hook, beholder_hook])
eval_spec = tf.estimator.EvalSpec(test_input_fn, steps=100)

tf.estimator.train_and_evaluate(capser, train_spec, eval_spec)
