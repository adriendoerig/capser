from capser_7_model_fn import *
from capser_7_input_fn import *
import logging
import numpy as np

# reproducibility
tf.reset_default_graph()
np.random.seed(42)
tf.set_random_seed(42)

# directory management
print('#################### MODEL_NAME_VERSION: ' + MODEL_NAME + '_' + str(version) + ' ####################')

# create estimator for model (the model is described in capser_7_model_fn)
capser = tf.estimator.Estimator(model_fn=model_fn, params={'model_batch_size': batch_size}, model_dir=checkpoint_path)

# train model
logging.getLogger().setLevel(logging.INFO)  # to show info about training progress

train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=100)
eval_spec = tf.estimator.EvalSpec(lambda: input_fn_config(data_path+'/test_squares.tfrecords'), steps=100)

tf.estimator.train_and_evaluate(capser, train_spec, eval_spec)


# evaluate on each kind of test stimuli
for this_name in test_filenames:
    capser.evaluate(input_fn=lambda: input_fn_config(this_name))

