from capser_7_model_fn_temp import *
from capser_7_input_fn import *
import logging
import numpy as np
from tensorflow.python.training import basic_session_run_hooks
from tensorboard.plugins.beholder import Beholder
from tensorboard.plugins.beholder import BeholderHook

# reproducibility
tf.reset_default_graph()
np.random.seed(42)
tf.set_random_seed(42)

# directory management
print('#################### MODEL_NAME_VERSION: ' + MODEL_NAME + ' ####################')

# create estimator for model (the model is described in capser_7_model_fn)
capser = tf.estimator.Estimator(model_fn=model_fn_temp, params={'model_batch_size': batch_size}, model_dir=checkpoint_path)

# train model
logging.getLogger().setLevel(logging.INFO)  # to show info about training progress

metadata_hook = basic_session_run_hooks.ProfilerHook(output_dir=checkpoint_path, save_steps=1000)  # to get metadata (e.g. how much time is spent loading data, or processing it on the GPU, etc)
beholder = Beholder(checkpoint_path)
beholder_hook = BeholderHook(checkpoint_path)

train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=n_steps, hooks=[metadata_hook, beholder_hook])
eval_spec = tf.estimator.EvalSpec(lambda: input_fn_config(data_path+'/test_squares.tfrecords'), steps=100)

tf.estimator.train_and_evaluate(capser, train_spec, eval_spec)

logging.getLogger().setLevel(logging.CRITICAL)  # to show less info in the console

n_expt_batches = 50

for category in test_stimuli.keys():
    print('COMPUTING VERNIER OFFSET FOR ' + category)
    uncrowding_expt_result = np.zeros(shape=(3,))
    for stim in range(3):
        for batch in range(n_expt_batches):
            capser_out = list(capser.predict(input_fn=lambda: input_fn_config(data_path+'/'+category+'/test_'+category+str(stim)+'.tfrecords')))
            vernier_accuracy = [p["vernier_accuracy"] for p in capser_out]
            uncrowding_expt_result[stim] += vernier_accuracy[0]/n_expt_batches
            print('Category: ' + category + '. \nUsing a total of ' + str(n_expt_batches*batch_size) + ' stimuli of each type. \nResult for currently computed stimuli: ' + str(uncrowding_expt_result))
    print('###################################################################')
    print('# Uncrowding experiment result for ' + category + ':')
    print('# ' + str(uncrowding_expt_result))
    print('###################################################################')
    if not os.path.exists(checkpoint_path + '/uncrowding_exp_results_step_'+str(n_steps)+'_noise_'+str(test_noise_level)+'_shape_size_'+str(shape_size)+'.txt'):
        with open(checkpoint_path + '//uncrowding_exp_results_step_'+str(n_steps)+'_noise_'+str(test_noise_level)+'_shape_size_'+str(shape_size)+'.txt', 'w') as f:
            f.write(category + ' : \t' + str(uncrowding_expt_result) + '\n')
    else:
        with open(checkpoint_path + '//uncrowding_exp_results_step_'+str(n_steps)+'_noise_'+str(test_noise_level)+'_shape_size_'+str(shape_size)+'.txt', 'a') as f:
            f.write(category + ' : \t' + str(uncrowding_expt_result) + '\n')