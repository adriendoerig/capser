from capser_7_model_fn import *
from make_tf_dataset import *
import logging

# reproducibility
tf.reset_default_graph()
np.random.seed(42)
tf.set_random_seed(42)

# directory management
print('#################### MODEL_NAME_VERSION: ' + MODEL_NAME + '_' + str(version) + ' ####################')

if not os.path.exists(image_output_dir):
    os.makedirs(image_output_dir)
if version_to_restore is None:
    save_params()

# create tfRecord data files if needed, and have a look at
if create_new_train_set:
    make_multi_shape_tfRecords(stim_maker, shape_types, n_train_samples, train_data_path)
if create_new_test_sets:
    for category in test_stimuli.keys():
        make_config_tfRecords(stim_maker, test_stimuli[category], n_test_samples, os.path.join(data_path, "test_" + category + ".tfrecords"))
if check_data is not None:
    show_data(check_data, type='config')

# create estimator for model (the model is described in capser_7_model_fn)
capser = tf.estimator.Estimator(model_fn=model_fn_tpu, params={'batch_size': batch_size}, model_dir=LOGDIR)

# train model
logging.getLogger().setLevel(logging.INFO)  # to show info about training progress
capser.train(input_fn=train_input_fn, steps=n_steps)
