import tensorflow as tf
from image_decoder_input_fn import image_decoder_input_function
from image_decoder_model_fn import image_decoder_model_fn
import numpy as np
import os
from parameters import test_stimuli
from image_decoder_parameters import batch_size, image_decoder_data_path


# reproducibility
tf.reset_default_graph()
np.random.seed(42)
tf.set_random_seed(42)

model_dir = './image_decoder'
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

image_decoder_model = tf.estimator.Estimator(model_fn=image_decoder_model_fn, model_dir=model_dir)

train_spec = tf.estimator.TrainSpec(input_fn=lambda: image_decoder_input_function(filenames=image_decoder_data_path+'/vernier_train_set.tfrecords'), max_steps=10000)
eval_spec = tf.estimator.EvalSpec(input_fn=lambda: image_decoder_input_function(filenames=image_decoder_data_path+'/vernier_train_set.tfrecords'), steps=100)

tf.estimator.train_and_evaluate(image_decoder_model, train_spec, eval_spec)

n_expt_batches = 1

for category in test_stimuli.keys():
    print('COMPUTING VERNIER OFFSET FOR ' + category)
    uncrowding_expt_result = np.zeros(shape=(3,))
    for stim in range(3):
        for batch in range(n_expt_batches):
            print('./image_decoder/np_arrays_'+category+'/'+str(stim))
            image_decoder_out = list(image_decoder_model.predict(input_fn=lambda: image_decoder_input_function(image_decoder_data_path+'/'+category+'/test_'+category+str(stim)+'.tfrecords')))
            vernier_accuracy = [p["vernier_accuracy"] for p in image_decoder_out]
            uncrowding_expt_result[stim] += vernier_accuracy[0]/n_expt_batches
            print('Category: ' + category + '. \nUsing a total of ' + str(n_expt_batches*batch_size) + ' stimuli of each type. \nResult for currently computed stimuli: ' + str(uncrowding_expt_result))
    print('###################################################################')
    print('# Uncrowding experiment result for ' + category + ':')
    print('# ' + str(uncrowding_expt_result))
    print('###################################################################')
    if not os.path.exists(model_dir + '/image_decoder_exp_results_step.txt'):
        with open(model_dir + '/image_decoder_exp_results_step.txt', 'w') as f:
            f.write(category + ' : \t' + str(uncrowding_expt_result) + '\n')
    else:
        with open(model_dir + '/image_decoder_exp_results_step.txt', 'a') as f:
            f.write(category + ' : \t' + str(uncrowding_expt_result) + '\n')