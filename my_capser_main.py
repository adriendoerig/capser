# -*- coding: utf-8 -*-
"""
My capsnet: On my way to an own capsnet!
Main script
Created on 17.10.2018
@author: Lynn
"""

import logging
import numpy as np
import tensorflow as tf
from my_parameters import parameters
from my_capser_model_fn import model_fn
from my_capser_input_fn import train_input_fn, test_input_fn

###################################################

# Disable annoying tf debugging logs:
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
#tf.logging.set_verbosity(tf.logging.ERROR)

print('-------------------------------------------------------')
print('TF version:', tf.__version__)
print('Starting capsnet script...')
print('-------------------------------------------------------')

# For reproducability:
tf.reset_default_graph()
np.random.seed(42)
tf.set_random_seed(42)

# create estimator for model
capser = tf.estimator.Estimator(model_fn=model_fn, model_dir=parameters.logdir)

# to output loss in the terminal every few steps
logging.getLogger().setLevel(logging.INFO)  # to show info about training progress

# tell the estimator where to get training and eval data, and for how long to train
train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=parameters.n_steps)
eval_spec = tf.estimator.EvalSpec(test_input_fn, steps=100)

# train (and evaluate from time to time)!
tf.estimator.train_and_evaluate(capser, train_spec, eval_spec)

print('... Finished capsnet script!')
print('-------------------------------------------------------')
