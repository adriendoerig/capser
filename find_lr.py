# -*- coding: utf-8 -*-
"""
find_lr: Use increasing lr and plot to tensorboard to find the best one
@author: Adrien

Last update on 28.12.2018
-> created script
"""

import logging
import tensorflow as tf

from my_parameters import parameters
from my_capser_model_fn import model_fn
from my_capser_input_fn import train_input_fn, eval_input_fn


##################################
#    Training and evaluation:    #
##################################
# Output the loss in the terminal every few steps:
logging.getLogger().setLevel(logging.INFO)

# Create the estimator:
capser = tf.estimator.Estimator(model_fn=model_fn, model_dir=parameters.logdir, config=tf.estimator.RunConfig(save_summary_steps=10))
train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=parameters.n_steps)
eval_spec = tf.estimator.EvalSpec(lambda: eval_input_fn(parameters.val_data_path), steps=parameters.eval_steps,  throttle_secs=1000000)

# Lets go!
tf.estimator.train_and_evaluate(capser, train_spec, eval_spec)
