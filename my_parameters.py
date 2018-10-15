# -*- coding: utf-8 -*-
"""
My capsnet: all parameters
Version 1
Created on Thu Oct  4 10:43:40 2018
@author: Lynn
"""

import tensorflow as tf

flags = tf.app.flags


###########################
#          Paths          #
###########################
data_path = './data'
flags.DEFINE_string('data_path', data_path, 'path where all data files are located')
flags.DEFINE_string('train_data_path', data_path+'/train.tfrecords', 'path for the tfrecords file involving the training set')
flags.DEFINE_list('test_data_paths', [data_path+'/test_squares.tfrecords',
                                      data_path+'/test_circles.tfrecords',
                                      data_path+'/test_hectagon.tfrecords',
                                      data_path+'/test_4stars.tfrecords',
                                      data_path+'/test_stars.tfrecords',
                                      data_path+'/test_squares_stars.tfrecords'], 'path for the tfrecords file involving the test set')
MODEL_NAME = 'test1'
flags.DEFINE_string('logdir', data_path + '/' + MODEL_NAME + '/', 'save the model results here')

###########################
#   Stimulus parameters   #
###########################
flags.DEFINE_integer('n_train_samples', 20000, 'number of samples in the training set')
flags.DEFINE_integer('n_test_samples', 1000, 'number of samples in the test set')
flags.DEFINE_integer('batch_size', 2, 'batch size')
flags.DEFINE_integer('buffer_size', 1024, 'buffer size')

im_size = [60, 150]
flags.DEFINE_list('im_size', im_size, 'image size of datasets')
flags.DEFINE_integer('im_depth', 1, 'number of colour channels')
flags.DEFINE_integer('shape_size', 20, 'size of the shapes')
flags.DEFINE_integer('bar_width', 1, 'thickness of shape lines')
flags.DEFINE_list('shape_types', [0, 1, 2, 3, 4, 5], 'pool of shapes (see batchmaker)')
flags.DEFINE_list('n_shapes', [1, 3, 5, 7], 'pool of shape repetitions per stimulus')
flags.DEFINE_float('noise', 0.025, 'amount of added random Gaussian noise')


###########################
#   Network parameters    #
###########################

# Primary caps: 32 FMs of 6x6 caps each, each outputs an 8D activation vector
caps1_nmaps = 32
caps1_ndims = 8
kernel1 = 9
kernel2 = 9
stride1 = 1
stride2 = 2
dim1 = int((((im_size[0] - kernel1+1) / stride1) - kernel2+1) / stride2)
dim2 = int((((im_size[1] - kernel1+1) / stride1) - kernel2+1) / stride2)
flags.DEFINE_integer('caps1_nmaps', caps1_nmaps, 'primary caps, number of feature maps')
flags.DEFINE_integer('caps1_ncaps', caps1_nmaps * dim1 * dim2, 'primary caps, number of caps')
flags.DEFINE_integer('caps1_ndims', caps1_ndims, 'primary caps, number of dims')

# Conv1: images shrink by 9-1=8 pix due to no padding (=20x20)
conv1_params = {'filters': caps1_nmaps*caps1_ndims, 'kernel_size': kernel1, 'strides': stride1,
                'padding': 'valid', 'activation': tf.nn.relu}

# Conv2: images shrink by 8 pix (=12x12) and size gets divided by 2 due to
# the stride (=6x6 features maps)
conv2_params = {'filters': caps1_nmaps*caps1_ndims, 'kernel_size': kernel2, 'strides': stride2,
                'padding': 'valid', 'activation': tf.nn.relu}

# Digit caps: 10 capsules (each digit) of 16 dimensions each
flags.DEFINE_integer('caps2_ncaps', 6, 'second caps layer, number of caps')
flags.DEFINE_integer('caps2_ndims', 16, 'second caps layer, number of dims')

# Decoder:
flags.DEFINE_integer('n_hidden1', 512, 'size of hidden layer 1 in decoder')
flags.DEFINE_integer('n_hidden2', 1024, 'size of hidden layer 2 in decoder')
flags.DEFINE_integer('n_output', im_size[0]*im_size[1], 'output size of the decoder')


###########################
#    Hyperparameters      #
###########################

# Margin loss
flags.DEFINE_float('m_plus', 0.9, 'the parameter of m plus')
flags.DEFINE_float('m_minus', 0.1, 'the parameter of m minus')
flags.DEFINE_float('lambda_val', 0.5, 'down weight of the loss for absent digit classes')

# For training
#flags.DEFINE_integer('batch_size', 64, 'batch size')
flags.DEFINE_integer('n_epochs', 1, 'number of epochs')
flags.DEFINE_integer('n_steps', 1000, 'number of steps')
flags.DEFINE_float('learning_rate', 0.0005, 'chosen learning rate for training')
flags.DEFINE_integer('iter_routing', 2, 'number of iterations in routing algorithm')
flags.DEFINE_boolean('mask_with_y', True, 'use the true label to mask out target capsule or not')
flags.DEFINE_boolean('restore_checkpoint', True, 'restore_checkpoint or not')

flags.DEFINE_float('init_sigma', 0.01, 'stddev for W initializer')
flags.DEFINE_float('regularization_scale', 0.392, 'regularization coefficient for reconstruction loss, default to 0.0005*784=0.392')
flags.DEFINE_float('alpha_rec', 0.0005, 'alpha for reconstruction loss')
# which one makes more sense?

############################
#   Environment setting    #
############################
#flags.DEFINE_string('dataset', 'mnist', 'The name of dataset [mnist, fashion-mnist')
flags.DEFINE_boolean('is_training', True, 'train or predict phase')
flags.DEFINE_integer('num_threads', 8, 'number of threads of enqueueing examples')
flags.DEFINE_integer('train_sum_freq', 100, 'the frequency of saving train summary(step)')
flags.DEFINE_integer('val_sum_freq', 500, 'the frequency of saving valuation summary(step)')
flags.DEFINE_integer('save_freq', 3, 'the frequency of saving model(epoch)')
flags.DEFINE_string('results', 'results', 'path for saving results')


params = tf.app.flags.FLAGS
