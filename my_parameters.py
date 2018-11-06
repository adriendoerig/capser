# -*- coding: utf-8 -*-
"""
My capsnet: all parameters
Last update on 31.10.2018
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
MODEL_NAME = 'test_new32'
flags.DEFINE_string('logdir', data_path + '/' + MODEL_NAME + '/', 'save the model results here')


###########################
#     Reproducibility     #
###########################
flags.DEFINE_boolean('random_seed', False,  'if true, set random_seed=42 for the weights initialization')
# Note: seed=42 usually leads to very bad results (= no vernier offset discrimination)
# Contrary to that seed=1 leads to good results


###########################
#   Stimulus parameters   #
###########################
batch_size = 32
n_steps = 50000
eval_steps = 50
flags.DEFINE_integer('n_train_samples', 100000, 'number of samples in the training set')
flags.DEFINE_integer('n_test_samples', 3200, 'number of samples in the test set')

im_size = [60, 150]
shape_types = [0, 1, 2, 3, 4, 5]
flags.DEFINE_list('im_size', im_size, 'image size of datasets')
flags.DEFINE_integer('im_depth', 1, 'number of colour channels')
flags.DEFINE_integer('shape_size', 20, 'size of the shapes')
flags.DEFINE_integer('bar_width', 1, 'thickness of shape lines')
flags.DEFINE_list('shape_types', shape_types, 'pool of shapes (see batchmaker)')
flags.DEFINE_list('n_shapes', [1, 3, 5, 7], 'pool of shape repetitions per stimulus')

# Can still be changed after creation of tfrecords file:
flags.DEFINE_float('train_noise', 0.01, 'amount of added random Gaussian noise')
flags.DEFINE_float('test_noise', 0.02, 'amount of added random Gaussian noise')
flags.DEFINE_boolean('only_venier', False, 'percentage of pictures in one batch that only involve a vernier')
flags.DEFINE_float('only_venier_percent', 0.2, 'percentage of pictures in one batch that only involve a vernier')


###########################
#   Network parameters    #
###########################
caps1_nmaps = 18
caps1_ndims = 3
kernel1 = 5
kernel2 = 9
kernel3 = 9
stride1 = 1
stride2 = 1
stride3 = 2
# For some reason (rounding/padding?), the following calculation is not always 100% precise, so u might have to add +1:
dim1 = int((((((im_size[0] - kernel1+1) / stride1) - kernel2+1) / stride2) - kernel3+1) / stride3) + 0
dim2 = int((((((im_size[1] - kernel1+1) / stride1) - kernel2+1) / stride2) - kernel3+1) / stride3) + 0

conv1_params = {'filters': 64, 'kernel_size': kernel1, 'strides': stride1,
                'padding': 'valid'}

conv2_params = {'filters': 64, 'kernel_size': kernel2, 'strides': stride2,
                'padding': 'valid'}

# these will be reshaped to create the primary capsules
conv3_params = {'filters': caps1_nmaps*caps1_ndims, 'kernel_size': kernel3, 'strides': stride3,
                'padding': 'valid'}

flags.DEFINE_integer('caps1_nmaps', caps1_nmaps, 'primary caps, number of feature maps')
flags.DEFINE_integer('caps1_ncaps', caps1_nmaps * dim1 * dim2, 'primary caps, number of caps')
flags.DEFINE_integer('caps1_ndims', caps1_ndims, 'primary caps, number of dims')

# Output capsules: caps2_ncaps capsules (one for each shape type) of caps2_ndims dimensions each
flags.DEFINE_integer('caps2_ncaps', len(shape_types), 'second caps layer, number of caps')
flags.DEFINE_integer('caps2_ndims', 3, 'second caps layer, number of dims')

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
flags.DEFINE_integer('eval_steps', eval_steps, 'number of evaluation steps per evaluation pass')
flags.DEFINE_integer('eval_throttle_secs', 1800, 'minimal seconds between evaluation passes')
flags.DEFINE_integer('n_epochs', 1, 'number of epochs')
flags.DEFINE_integer('n_steps', n_steps, 'number of steps')
flags.DEFINE_integer('iter_routing', 2, 'number of iterations in routing algorithm')
flags.DEFINE_float('init_sigma', 0.01, 'stddev for W initializer')

# Learning rate
flags.DEFINE_boolean('exp_learning_decay', False, 'if true, use the following parameters for exponential learning rate decay')
flags.DEFINE_float('learning_rate', 0.001, 'chosen learning rate for training')
flags.DEFINE_integer('decay_steps', 500, 'number of steps after which the learning rate exp decays')
flags.DEFINE_float('decay_rate', 0.90, 'decay rate for exponential decay of learning rate')

# Losses
# flags.DEFINE_float('regularization_scale', 0.0005*im_size[0]*im_size[1], 'regularization coefficient for reconstruction loss, default to 0.0005*im_size[0]*im_size[1] (reduce_mean)')
flags.DEFINE_float('alpha_margin', 1., 'alpha for margin loss')
flags.DEFINE_float('alpha_vernier_reconstruction', 0.0005, 'alpha for reconstruction loss for vernier image (reduce_sum)')
flags.DEFINE_float('alpha_shape_reconstruction', 0.0001, 'alpha for reconstruction loss for shape image (reduce_sum)')
flags.DEFINE_float('alpha_vernieroffset', 2., 'alpha for vernieroffset loss')

# Regulariztion:
flags.DEFINE_boolean('batch_norm_conv', False, 'use batch normalization between every conv layer')  #
flags.DEFINE_boolean('batch_norm_decoder', True, 'use batch normalization for the decoder layers')


parameters = tf.app.flags.FLAGS
