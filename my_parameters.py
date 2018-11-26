# -*- coding: utf-8 -*-
"""
My capsnet: all parameters
@author: Lynn

Last update on 26.11.2018
-> added nshapes and location loss
-> added alphas for each coordinate type
-> added overlapping_shapes parameter
-> added data augmentation (noise, flipping, contrast, brightness)
-> network can be run with 2 or 3 conv layers now
-> you can choose now between xentropy of squared_diff as location or nshapes loss
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
MODEL_NAME = '_log7'
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
flags.DEFINE_integer('n_train_samples', 200000, 'number of samples in the training set')
flags.DEFINE_integer('n_test_samples', 3200, 'number of samples in the test set')

im_size = [60, 150]
shape_types = [0, 1, 2, 3, 4, 5]
flags.DEFINE_list('im_size', im_size, 'image size of datasets')
flags.DEFINE_integer('im_depth', 1, 'number of colour channels')
flags.DEFINE_integer('shape_size', 20, 'size of the shapes')
flags.DEFINE_integer('bar_width', 1, 'thickness of shape lines')
flags.DEFINE_list('shape_types', shape_types, 'pool of shapes (see batchmaker)')
flags.DEFINE_list('n_shapes', [0, 1, 2, 3, 4, 5, 6, 7], 'pool of shape repetitions per stimulus')
flags.DEFINE_boolean('overlapping_shapes', True,  'if true, shapes and vernier might overlap')


###########################
#    Data augmentation    #
###########################
flags.DEFINE_float('train_noise', 0.08, 'amount of added random Gaussian noise')
flags.DEFINE_float('test_noise', 0.12, 'amount of added random Gaussian noise')
flags.DEFINE_float('max_delta_brightness', 0.5, 'max factor to adjust brightness (+/-), must be non-negative')
flags.DEFINE_float('min_delta_contrast', 0.5, 'min factor to adjust contrast, must be non-negative')
flags.DEFINE_float('max_delta_contrast', 1.5, 'max factor to adjust contrast, must be non-negative')


###########################
#   Network parameters    #
###########################
n_conv_layers = 3
flags.DEFINE_integer('n_conv_layers', n_conv_layers, 'number of conv layers used (currently only 2 or 3)')

# Conv and primary caps:
caps1_nmaps = 6
caps1_ndims = 3

if n_conv_layers==2:
    # Case of 2 conv layers:
    kernel1 = 3
    kernel2 = 6
    stride1 = 1
    stride2 = 2
    # For some reason (rounding/padding?), the following calculation is not always 100% precise, so u might have to add +1:
    dim1 = int((((im_size[0] - kernel1+1) / stride1) - kernel2+1) / stride2) + 1
    dim2 = int((((im_size[1] - kernel1+1) / stride1) - kernel2+1) / stride2) + 1
    conv1_params = {'filters': caps1_nmaps*caps1_ndims, 'kernel_size': kernel1, 'strides': stride1,
                    'padding': 'valid'}
    conv2_params = {'filters': caps1_nmaps*caps1_ndims, 'kernel_size': kernel2, 'strides': stride2,
                    'padding': 'valid'}
    flags.DEFINE_list('conv_params', [conv1_params, conv2_params], 'list with the conv parameters')
    
elif n_conv_layers==3:
    # Case of 3 conv layers:
    kernel1 = 3
    kernel2 = 7
    kernel3 = 7
    stride1 = 1
    stride2 = 2
    stride3 = 2
    # For some reason (rounding/padding?), the following calculation is not always 100% precise, so u might have to add +1:
    dim1 = int((((((im_size[0] - kernel1+1) / stride1) - kernel2+1) / stride2) - kernel3+1) / stride3) + 0
    dim2 = int((((((im_size[1] - kernel1+1) / stride1) - kernel2+1) / stride2) - kernel3+1) / stride3) + 1
    conv1_params = {'filters': caps1_nmaps*caps1_ndims, 'kernel_size': kernel1, 'strides': stride1,
                    'padding': 'valid'}
    conv2_params = {'filters': caps1_nmaps*caps1_ndims, 'kernel_size': kernel2, 'strides': stride2,
                    'padding': 'valid'}
    conv3_params = {'filters': caps1_nmaps*caps1_ndims, 'kernel_size': kernel3, 'strides': stride3,
                    'padding': 'valid'}
    flags.DEFINE_list('conv_params', [conv1_params, conv2_params, conv3_params], 'list with the conv parameters')


flags.DEFINE_integer('caps1_nmaps', caps1_nmaps, 'primary caps, number of feature maps')
flags.DEFINE_integer('caps1_ncaps', caps1_nmaps * dim1 * dim2, 'primary caps, number of caps')
flags.DEFINE_integer('caps1_ndims', caps1_ndims, 'primary caps, number of dims')


# Output caps:
flags.DEFINE_integer('caps2_ncaps', len(shape_types), 'second caps layer, number of caps')
flags.DEFINE_integer('caps2_ndims', 4, 'second caps layer, number of dims')


# Decoder reconstruction:
flags.DEFINE_integer('n_hidden1', 512, 'size of hidden layer 1 in decoder')
flags.DEFINE_integer('n_hidden2', 1024, 'size of hidden layer 2 in decoder')
flags.DEFINE_integer('n_output', im_size[0]*im_size[1], 'output size of the decoder')


###########################
#    Hyperparameters      #
###########################
# For training
flags.DEFINE_integer('batch_size', 48, 'batch size')
flags.DEFINE_float('learning_rate', 0.0005, 'chosen learning rate for training')
flags.DEFINE_integer('iter_routing', 2, 'number of iterations in routing algorithm')

flags.DEFINE_integer('buffer_size', 1024, 'buffer size')
flags.DEFINE_integer('eval_steps', 50, 'frequency for eval spec; u need at least eval_steps*batch_size stimuli in the validation set')
flags.DEFINE_integer('eval_throttle_secs', 900, 'minimal seconds between evaluation passes')
flags.DEFINE_integer('n_epochs', None, 'number of epochs, if None allow for indifinite readings')
flags.DEFINE_integer('n_steps', 30000, 'number of steps')
flags.DEFINE_float('init_sigma', 0.01, 'stddev for W initializer')


###########################
#         Losses          #
###########################
flags.DEFINE_boolean('decode_reconstruction', False, 'decode the reconstruction and use reconstruction loss')

flags.DEFINE_boolean('decode_nshapes', True, 'decode the number of shapes and use nshapes loss')
flags.DEFINE_string('nshapes_loss', 'squared_diff', 'currently either xentropy or squared_diff')

flags.DEFINE_boolean('decode_location', True, 'decode the shapes locations and use location loss')
flags.DEFINE_string('location_loss', 'squared_diff', 'currently either xentropy or squared_diff')


# Control magnitude of losses
flags.DEFINE_float('alpha_vernieroffset', 1., 'alpha for vernieroffset loss')
flags.DEFINE_float('alpha_margin', 0.5, 'alpha for margin loss')
flags.DEFINE_float('alpha_vernier_reconstruction', 0.0005, 'alpha for reconstruction loss for vernier image (reduce_sum)')
flags.DEFINE_float('alpha_shape_reconstruction', 0.0001, 'alpha for reconstruction loss for shape image (reduce_sum)')
flags.DEFINE_float('alpha_nshapes', 0.3, 'alpha for nshapes loss')
flags.DEFINE_float('alpha_x_shapeloss', 0.1, 'alpha for loss of x coordinate of shape')
flags.DEFINE_float('alpha_y_shapeloss', 0.1, 'alpha for loss of y coordinate of shape')
flags.DEFINE_float('alpha_x_vernierloss', 0.1, 'alpha for loss of x coordinate of vernier')
flags.DEFINE_float('alpha_y_vernierloss', 0.1, 'alpha for loss of y coordinate of vernier')

# Margin loss
flags.DEFINE_float('m_plus', 0.9, 'the parameter of m plus')
flags.DEFINE_float('m_minus', 0.1, 'the parameter of m minus')
flags.DEFINE_float('lambda_val', 0.5, 'down weight of the loss for absent digit classes')


###########################
#     Regulariation       #
###########################
flags.DEFINE_boolean('batch_norm_conv', True, 'use batch normalization between every conv layer')
flags.DEFINE_boolean('batch_norm_decoder', True, 'use batch normalization for the decoder layers')


parameters = tf.app.flags.FLAGS
