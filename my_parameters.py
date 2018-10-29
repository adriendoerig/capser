# -*- coding: utf-8 -*-
"""
My capsnet: all parameters
Last update on 24.10.2018
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
flags.DEFINE_list('test_data_paths', [data_path+'/test_squares',
                                      data_path+'/test_circles',
                                      data_path+'/test_hectagon',
                                      data_path+'/test_4stars',
                                      data_path+'/test_stars',
                                      data_path+'/test_squares_stars'], 'path for the tfrecords file involving the test set')
MODEL_NAME = 'test_new19'
flags.DEFINE_string('logdir', data_path + '/' + MODEL_NAME + '/', 'save the model results here')


###########################
#     Reproducability     #
###########################
flags.DEFINE_boolean('random_seed', True,  'if true, set random_seed=42 for the weights initialization')


###########################
#   Stimulus parameters   #
###########################
batch_size = 32
n_steps = 5000
eval_freq = 50
flags.DEFINE_integer('n_train_samples', batch_size*n_steps, 'number of samples in the training set')
flags.DEFINE_integer('n_test_samples', batch_size*eval_freq, 'number of samples in the test set')

im_size = [60, 150]
shape_types = [0, 1, 2, 3, 4, 5]
flags.DEFINE_list('im_size', im_size, 'image size of datasets')
flags.DEFINE_integer('im_depth', 1, 'number of colour channels')
flags.DEFINE_integer('shape_size', 20, 'size of the shapes')
flags.DEFINE_integer('bar_width', 1, 'thickness of shape lines')
flags.DEFINE_list('shape_types', shape_types, 'pool of shapes (see batchmaker)')
flags.DEFINE_list('n_shapes', [1, 3, 5, 7], 'pool of shape repetitions per stimulus')
flags.DEFINE_float('noise', 0.005, 'amount of added random Gaussian noise')


###########################
#   Network parameters    #
###########################
caps1_nmaps = 16
caps1_ndims = 16
kernel1 = 6
kernel2 = 6
kernel3 = 6
stride1 = 1
stride2 = 2
stride3 = 2
# For some reason (rounding/padding?), the following calculation is not always 100% precise, so u might have to add +1:
dim1 = int((((((im_size[0] - kernel1+1) / stride1) - kernel2+1) / stride2) - kernel3+1) / stride3) + 0
dim2 = int((((((im_size[1] - kernel1+1) / stride1) - kernel2+1) / stride2) - kernel3+1) / stride3) + 1

conv1_params = {'filters': caps1_nmaps*caps1_ndims, 'kernel_size': kernel1, 'strides': stride1,
                'padding': 'valid', 'activation': tf.nn.relu}

conv2_params = {'filters': caps1_nmaps*caps1_ndims, 'kernel_size': kernel2, 'strides': stride2,
                'padding': 'valid', 'activation': tf.nn.relu}

conv3_params = {'filters': caps1_nmaps*caps1_ndims, 'kernel_size': kernel3, 'strides': stride3,
                'padding': 'valid', 'activation': tf.nn.relu}

flags.DEFINE_integer('caps1_nmaps', caps1_nmaps, 'primary caps, number of feature maps')
flags.DEFINE_integer('caps1_ncaps', caps1_nmaps * dim1 * dim2, 'primary caps, number of caps')
flags.DEFINE_integer('caps1_ndims', caps1_ndims, 'primary caps, number of dims')

# Shape caps: 10 capsules (each shape) of 16 dimensions each
flags.DEFINE_integer('caps2_ncaps', len(shape_types), 'second caps layer, number of caps')
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
flags.DEFINE_integer('batch_size', batch_size, 'batch size')
flags.DEFINE_integer('buffer_size', 1024, 'buffer size')
flags.DEFINE_integer('eval_freq', eval_freq, 'frequency for eval spec')
flags.DEFINE_integer('n_epochs', 1, 'number of epochs')
flags.DEFINE_integer('n_steps', n_steps, 'number of steps')
flags.DEFINE_float('learning_rate', 0.001, 'chosen learning rate for training')
flags.DEFINE_integer('iter_routing', 3, 'number of iterations in routing algorithm')

flags.DEFINE_float('init_sigma', 0.01, 'stddev for W initializer')
flags.DEFINE_float('regularization_scale', 0.0005*im_size[0]*im_size[1],
                   'regularization coefficient for reconstruction loss, default to 0.0005*im_size[0]*im_size[1] (reduce_mean)')
flags.DEFINE_float('alpha_margin', 1., 'alpha for margin loss')
flags.DEFINE_float('alpha_reconstruction', 0.0005, 'alpha for reconstruction loss (reduce_sum)')
flags.DEFINE_float('alpha_vernieroffset', 1., 'alpha for vernieroffset loss')

flags.DEFINE_boolean('batch_norm_conv', False, 'use batch normalization between every conv layer')  #
flags.DEFINE_boolean('batch_norm_decoder', True, 'use batch normalization for the decoder layers')


parameters = tf.app.flags.FLAGS
