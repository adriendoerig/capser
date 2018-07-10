from capser_7_model_fn import *
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

# create estimator for model (the model is described in capser_7_model_fn)
capser = tf.estimator.Estimator(model_fn=model_fn, params={'batch_size': 16}, model_dir=LOGDIR)

# train model
logging.getLogger().setLevel(logging.INFO)  # to show info about training progress
capser.train(input_fn=train_input_fn, steps=n_steps)
