# -*- coding: utf-8 -*-
"""
My capsnet: all parameters
@author: Lynn

Last update on 28.12.2018
-> added nshapes and location loss
-> added alphas for each coordinate type
-> added overlapping_shapes parameter
-> added data augmentation (noise, flipping, contrast, brightness)
-> network can be run with 2 or 3 conv layers now
-> you can choose now between xentropy of squared_diff as location or nshapes loss
-> it is possible now to use batch normalization for every type of loss, this involved some major changes in the code!
-> added some parameters for the reconstruction script
-> train and test noise is randomly changed now between a lower and upper border
-> change for random_seed with regards to change in secondary_caps_layer()
-> small changes due to finally working reconstruction script
-> some changes in parameter names
-> you can choose between a reconstruction decoder with fc or conv layers
-> now we always have as many primary caps types as train_shape_types
-> new dataset
-> new validation and testing procedures
"""

import tensorflow as tf

flags = tf.app.flags


###########################
#          Paths          #
###########################
# In general:
data_path = './data'
MODEL_NAME = '_log1'
flags.DEFINE_string('data_path', data_path, 'path where all data files are located')

# For training stimuli:
flags.DEFINE_string('train_data_path', data_path+'/train.tfrecords', 'path for tfrecords with training set')
flags.DEFINE_string('val_data_path', data_path+'/val.tfrecords', 'path for tfrecords with validation set')
flags.DEFINE_list('test_data_paths',
                  [data_path+'/test_squares.tfrecords',
                   data_path+'/test_circles.tfrecords',
                   data_path+'/test_hectagon.tfrecords',
                   data_path+'/test_4stars.tfrecords',
                   data_path+'/test_rhombus.tfrecords',
                   data_path+'/test_stuff.tfrecords'], 'path for tfrecords with test set')

# For crowding/uncrowding stimuli:
flags.DEFINE_string('val_crowding_data_path', data_path+'/val_crowding.tfrecords', 'path for tfrecords with validation crowding set')
flags.DEFINE_list('test_crowding_data_paths',
                  [data_path+'/test_crowding_squares',
                   data_path+'/test_crowding_circles',
                   data_path+'/test_crowding_hectagon',
                   data_path+'/test_crowding_4stars',
                   data_path+'/test_crowding_rhombus',
                   data_path+'/test_crowding_squares_rhombus'], 'path for tfrecords with test crowding set')

flags.DEFINE_string('logdir', data_path + '/' + MODEL_NAME + '/', 'save the model results here')
flags.DEFINE_string('logdir_reconstruction', data_path + '/' + MODEL_NAME + '_rec/', 'save results with reconstructed weights here')


###########################
#     Reproducibility     #
###########################
flags.DEFINE_integer('random_seed', None, 'if not None, set seed for weights initialization')


###########################
#   Stimulus parameters   #
###########################
flags.DEFINE_integer('n_train_samples', 200000, 'number of samples in the training set')
flags.DEFINE_integer('n_test_samples', 3200, 'number of samples in the test set')

im_size = [35, 90]
flags.DEFINE_list('im_size', im_size, 'image size of datasets')
flags.DEFINE_integer('im_depth', 1, 'number of colour channels')
flags.DEFINE_integer('shape_size', 16, 'size of the shapes')
flags.DEFINE_integer('bar_width', 1, 'thickness of shape lines')

# shape_types for training have to have a range from 0 to max
# the data_paths for the train and test have to match the chosen shape types 
shape_types = [0, 1, 2, 3, 4, 5, 6]
test_shape_types = [0, 1, 2, 3, 4, 5, 42]
flags.DEFINE_list('shape_types', shape_types, 'pool of shapes (see batchmaker)')
flags.DEFINE_list('test_shape_types', test_shape_types, 'pool of shapes (see batchmaker)')
flags.DEFINE_list('n_shapes', [0, 1, 2, 3, 4, 5], 'pool of shape repetitions per stimulus')
flags.DEFINE_boolean('overlapping_shapes', True,  'if true, shapes and vernier might overlap')


###########################
#    Data augmentation    #
###########################
flags.DEFINE_list('train_noise', [0.001, 0.003], 'amount of added random Gaussian noise')
flags.DEFINE_list('test_noise', [0.2, 0.15], 'amount of added random Gaussian noise')
flags.DEFINE_list('clip_values', [0., 1.], 'min and max pixel value for every image')
flags.DEFINE_float('delta_brightness', 0.1, 'factor to adjust brightness (+/-), must be non-negative')
flags.DEFINE_list('delta_contrast', [0.6, 1.2], 'min and max factor to adjust contrast, must be non-negative')


###########################
#   Network parameters    #
###########################
n_conv_layers = 3
flags.DEFINE_integer('n_conv_layers', n_conv_layers, 'number of conv layers used (currently only 2 or 3)')

# Conv and primary caps:
caps1_nmaps = len(shape_types)*3
caps1_ndims = 8

if n_conv_layers==2:
    # Case of 2 conv layers:
    kernel1 = 3
#    kernel2 = 6
#    stride1 = 1
#    stride2 = 2
#    # For some reason (rounding/padding?), the following calculation is not always 100% precise, so u might have to add +1:
#    dim1 = int((((im_size[0] - kernel1+1) / stride1) - kernel2+1) / stride2) + 1
#    dim2 = int((((im_size[1] - kernel1+1) / stride1) - kernel2+1) / stride2) + 1
#    conv1_params = {'filters': caps1_nmaps*caps1_ndims, 'kernel_size': kernel1, 'strides': stride1,
#                    'padding': 'valid'}
#    conv2_params = {'filters': caps1_nmaps*caps1_ndims, 'kernel_size': kernel2, 'strides': stride2,
#                    'padding': 'valid'}
#    flags.DEFINE_list('conv_params', [conv1_params, conv2_params], 'list with the conv parameters')
    
elif n_conv_layers==3:
    # Case of 3 conv layers:
    kernel1 = 5
    kernel2 = 5
    kernel3 = 5
    stride1 = 1
    stride2 = 2
    stride3 = 2
    # For some reason (rounding/padding?), the following calculation is not always 100% precise, so u might have to add +1:
    dim1 = int((((((im_size[0] - kernel1+1) / stride1) - kernel2+1) / stride2) - kernel3+1) / stride3) + 1
    dim2 = int((((((im_size[1] - kernel1+1) / stride1) - kernel2+1) / stride2) - kernel3+1) / stride3) + 1
    conv1_params = {'filters': 32, 'kernel_size': kernel1, 'strides': stride1, 'padding': 'valid'}
    conv2_params = {'filters': 32, 'kernel_size': kernel2, 'strides': stride2, 'padding': 'valid'}
    conv3_params = {'filters': caps1_nmaps*caps1_ndims, 'kernel_size': kernel3, 'strides': stride3, 'padding': 'valid'}
    flags.DEFINE_list('conv_params', [conv1_params, conv2_params, conv3_params], 'list with the conv parameters')


flags.DEFINE_integer('caps1_nmaps', caps1_nmaps, 'primary caps, number of feature maps')
flags.DEFINE_integer('caps1_ncaps', caps1_nmaps * dim1 * dim2, 'primary caps, number of caps')
flags.DEFINE_integer('caps1_ndims', caps1_ndims, 'primary caps, number of dims')


# Output caps:
flags.DEFINE_integer('caps2_ncaps', len(shape_types), 'second caps layer, number of caps')
flags.DEFINE_integer('caps2_ndims', 8, 'second caps layer, number of dims')


# Decoder reconstruction:
flags.DEFINE_string('rec_decoder_type', 'conv', 'use fc or conv layers for decoding (only with 3 conv layers)')
flags.DEFINE_integer('n_hidden_reconstruction_1', 512, 'size of hidden layer 1 in decoder')
flags.DEFINE_integer('n_hidden_reconstruction_2', 1024, 'size of hidden layer 2 in decoder')
flags.DEFINE_integer('n_output', im_size[0]*im_size[1], 'output size of the decoder')


###########################
#    Hyperparameters      #
###########################
# For training
flags.DEFINE_integer('batch_size', 48, 'batch size')
flags.DEFINE_boolean('find_lr', False, 'if true, uses an exponentially increasing learning rate to find when the loss stops improving')
flags.DEFINE_float('learning_rate', 2e-4, 'chosen learning rate for training')
flags.DEFINE_float('learning_rate_decay_steps', 7000, 'decays for this many steps, then goes back up (only used if find_lr is False')
flags.DEFINE_integer('iter_routing', 2, 'number of iterations in routing algorithm')

flags.DEFINE_integer('buffer_size', 1024, 'buffer size')
flags.DEFINE_integer('eval_steps', 50,
                     'frequency for eval spec; u need at least eval_steps*batch_size stimuli in the validation set')
flags.DEFINE_integer('eval_throttle_secs', 900, 'minimal seconds between evaluation passes')
flags.DEFINE_integer('n_epochs', None, 'number of epochs, if None allow for indifinite readings')
flags.DEFINE_integer('n_steps', 2*49000, 'number of steps')
flags.DEFINE_float('init_sigma', 0.01, 'stddev for W initializer')


###########################
#         Losses          #
###########################
flags.DEFINE_boolean('decode_reconstruction', True, 'decode the reconstruction and use reconstruction loss')

flags.DEFINE_boolean('decode_nshapes', True, 'decode the number of shapes and use nshapes loss')
nshapes_loss = 'xentropy'
flags.DEFINE_string('nshapes_loss', nshapes_loss, 'currently either xentropy or squared_diff')

flags.DEFINE_boolean('decode_location', True, 'decode the shapes locations and use location loss')
location_loss = 'xentropy'
flags.DEFINE_string('location_loss', location_loss, 'currently either xentropy or squared_diff')


# Control magnitude of losses
flags.DEFINE_float('alpha_vernieroffset', 1., 'alpha for vernieroffset loss')
flags.DEFINE_float('alpha_margin', 0.5, 'alpha for margin loss')
flags.DEFINE_float('alpha_vernier_reconstruction', 0.0005, 'alpha for reconstruction loss for vernier image (reduce_sum)')
flags.DEFINE_float('alpha_shape_reconstruction', 0.0001, 'alpha for reconstruction loss for shape image (reduce_sum)')

if nshapes_loss=='xentropy':
    flags.DEFINE_float('alpha_nshapes', 0.4, 'alpha for nshapes loss')
elif nshapes_loss=='squared_diff':
    flags.DEFINE_float('alpha_nshapes', 0.002, 'alpha for nshapes loss')

if location_loss=='xentropy':
    flags.DEFINE_float('alpha_x_shapeloss', 0.1, 'alpha for loss of x coordinate of shape')
    flags.DEFINE_float('alpha_y_shapeloss', 0.1, 'alpha for loss of y coordinate of shape')
    flags.DEFINE_float('alpha_x_vernierloss', 0.1, 'alpha for loss of x coordinate of vernier')
    flags.DEFINE_float('alpha_y_vernierloss', 0.1, 'alpha for loss of y coordinate of vernier')
elif location_loss=='squared_diff':
    flags.DEFINE_float('alpha_x_shapeloss', 0.000004, 'alpha for loss of x coordinate of shape')
    flags.DEFINE_float('alpha_y_shapeloss', 0.00005, 'alpha for loss of y coordinate of shape')
    flags.DEFINE_float('alpha_x_vernierloss', 0.000004, 'alpha for loss of x coordinate of vernier')
    flags.DEFINE_float('alpha_y_vernierloss', 0.00005, 'alpha for loss of y coordinate of vernier')

# Margin loss extras
flags.DEFINE_float('m_plus', 0.9, 'the parameter of m plus')
flags.DEFINE_float('m_minus', 0.1, 'the parameter of m minus')
flags.DEFINE_float('lambda_val', 0.5, 'down weight of the loss for absent digit classes')


###########################
#     Regularization       #
###########################
flags.DEFINE_boolean('batch_norm_conv', False, 'use batch normalization between every conv layer')
flags.DEFINE_boolean('batch_norm_reconstruction', True, 'use batch normalization for the reconstruction decoder layers')
flags.DEFINE_boolean('batch_norm_vernieroffset', True, 'use batch normalization for the vernieroffset loss layer')
flags.DEFINE_boolean('batch_norm_nshapes', True, 'use batch normalization for the nshapes loss layer')
flags.DEFINE_boolean('batch_norm_location', True, 'use batch normalization for the location loss layer')


parameters = tf.app.flags.FLAGS
