# -*- coding: utf-8 -*-
"""
My capsnet: all parameters
@author: Lynn

Last update on 18.02.2019
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
-> use train_procedures 'vernier_shape', 'random_random' or 'random'
-> implemented n_rounds to decide how often we evaluate the test sets
-> implemented a variety of uncrowding stimuli (412+)
-> implemented the possibility to have centralized_shapes only (combine this with random procedure!)
-> decide whether to use data augment or not
-> you can reduce the df for the x position of all shapes to have a fairer comparison
-> all shapes included again
"""

import tensorflow as tf

flags = tf.app.flags


###########################
#          Paths          #
###########################
# In general:
data_path = './data'
MODEL_NAME = '_logs_project1_1'
flags.DEFINE_string('data_path', data_path, 'path where all data files are located')

# For training stimuli:
flags.DEFINE_string('train_data_path', data_path+'/train.tfrecords', 'path for tfrecords with training set')
flags.DEFINE_string('val_data_path', data_path+'/val.tfrecords', 'path for tfrecords with validation set')
flags.DEFINE_list('test_data_paths',
#                  [data_path+'/test_squares.tfrecords',
                   [data_path+'/test_circles.tfrecords',
                   data_path+'/test_rhombus.tfrecords',
                   data_path+'/test_4stars.tfrecords'
#                   data_path+'/test_hexagons.tfrecords',
#                   data_path+'/test_6stars.tfrecords'
                   ], 'path for tfrecords with test set')

# For crowding/uncrowding stimuli:
flags.DEFINE_string('val_crowding_data_path', data_path+'/val_crowding.tfrecords', 'path for tfrecords with validation crowding set')
flags.DEFINE_list('test_crowding_data_paths',
#                  [data_path+'/test_crowding_squares',
                   [data_path+'/test_crowding_circles',
                   data_path+'/test_crowding_rhombus',
                   data_path+'/test_crowding_4stars',
#                   data_path+'/test_crowding_hexagons',
#                   data_path+'/test_crowding_6stars',
#                   data_path+'/test_crowding_squares_circles',
#                   data_path+'/test_crowding_circles_squares',
#                   data_path+'/test_crowding_squares_rhombus',
#                   data_path+'/test_crowding_rhombus_squares',
#                   data_path+'/test_crowding_squares_4stars',
#                   data_path+'/test_crowding_4stars_squares',
#                   data_path+'/test_crowding_squares_hexagons',
#                   data_path+'/test_crowding_hexagons_squares',
#                   data_path+'/test_crowding_squares_6stars',
#                   data_path+'/test_crowding_6stars_squares',
                   data_path+'/test_crowding_circles_rhombus',
                   data_path+'/test_crowding_rhombus_circles',
                   data_path+'/test_crowding_circles_4stars',
                   data_path+'/test_crowding_4stars_circles',
#                   data_path+'/test_crowding_circles_hexagons',
#                   data_path+'/test_crowding_hexagons_circles',
#                   data_path+'/test_crowding_circles_6stars',
#                   data_path+'/test_crowding_6stars_circles',
                   data_path+'/test_crowding_rhombus_4stars',
                   data_path+'/test_crowding_4stars_rhombus'
#                   data_path+'/test_crowding_rhombus_hexagons',
#                   data_path+'/test_crowding_hexagons_rhombus',
#                   data_path+'/test_crowding_rhombus_6stars',
#                   data_path+'/test_crowding_6stars_rhombus',
#                   data_path+'/test_crowding_4stars_hexagons',
#                   data_path+'/test_crowding_hexagons_4stars',
#                   data_path+'/test_crowding_4stars_6stars',
#                   data_path+'/test_crowding_6stars_4stars',
#                   data_path+'/test_crowding_hexagons_6stars',
#                   data_path+'/test_crowding_6stars_hexagons'
                   ], 'path for tfrecords with test crowding set')

flags.DEFINE_string('logdir', data_path + '/' + MODEL_NAME + '/', 'save the model results here')
flags.DEFINE_string('logdir_reconstruction', data_path + '/' + MODEL_NAME + '_rec/', 'save results with reconstructed weights here')


###########################
#     Reproducibility     #
###########################
flags.DEFINE_integer('random_seed', None, 'if not None, set seed for weights initialization')


###########################
#   Stimulus parameters   #
###########################
# IMPORTANT NOTES:
    # 1. If u change any stimulus parameter, keep in mind that u need to create a new training set.
    # 2. Combine centralized_shapes=True with train_procedure='random'!
# EXCEPTIONS:
    # 1. You can use the same training set for the train_procedures 'random_random & 'random'
flags.DEFINE_string('train_procedure', 'random', 'choose between having vernier_shape, random_random and random')
flags.DEFINE_boolean('overlapping_shapes', True,  'if true, shapes and vernier might overlap')
flags.DEFINE_boolean('centralized_shapes', False,  'if true, each shape is in the middle of the image')
flags.DEFINE_boolean('reduce_df', True,  'if true, the degrees of freedom for position on the x axis get adapted')

flags.DEFINE_integer('n_train_samples', 100000, 'number of samples in the training set')
flags.DEFINE_integer('n_test_samples', 2400, 'number of samples in the test set')

im_size = [16, 72]
flags.DEFINE_list('im_size', im_size, 'image size of datasets')
flags.DEFINE_integer('im_depth', 1, 'number of colour channels')
flags.DEFINE_integer('shape_size', 14, 'size of the shapes')
flags.DEFINE_integer('bar_width', 1, 'thickness of shape lines')

# shape_types for training have to have a range from 0 to max
# the data_paths for the train and test have to match the chosen shape types
shape_types = [0, 1, 2, 3]
#test_shape_types = [1, 2, 3, 4, 5, 6,
#                    412, 421, 413, 431, 414, 441, 415, 451, 416, 461,
#                    423, 432, 424, 442, 425, 452, 426, 462,
#                    434, 443, 435, 453, 436, 463,
#                    445, 454, 446, 464,
#                    456, 465]
test_shape_types = [1, 2, 3,
                    412, 421, 413, 431,
                    423, 432]

flags.DEFINE_list('shape_types', shape_types, 'pool of shapes (see batchmaker)')
flags.DEFINE_list('test_shape_types', test_shape_types, 'pool of shapes (see batchmaker)')
flags.DEFINE_list('n_shapes', [1, 3, 5], 'pool of shape repetitions per stimulus')


###########################
#    Data augmentation    #
###########################
flags.DEFINE_list('train_noise', [0.0, 0.00], 'amount of added random Gaussian noise')
flags.DEFINE_list('test_noise', [0.01, 0.01], 'amount of added random Gaussian noise')
flags.DEFINE_list('clip_values', [0., 1.], 'min and max pixel value for every image')
flags.DEFINE_boolean('allow_flip_augmentation', False, 'augment by flipping the image up/down or left/right')
flags.DEFINE_boolean('allow_contrast_augmentation', True, 'augment by changing contrast and brightness')
flags.DEFINE_float('delta_brightness', 0.1, 'factor to adjust brightness (+/-), must be non-negative')
flags.DEFINE_list('delta_contrast', [0.6, 1.2], 'min and max factor to adjust contrast, must be non-negative')


###########################
#   Network parameters    #
###########################
# Conv and primary caps:
caps1_nmaps = len(shape_types)
caps1_ndims = 1


# Case of 3 conv layers:
kernel1 = 5
kernel2 = 5
kernel3 = 6
stride1 = 1
stride2 = 1
stride3 = 2

# For some reason (rounding/padding?), the following calculation is not always 100% precise, so u might have to add +1:
dim1 = int((((((im_size[0] - kernel1+1) / stride1) - kernel2+1) / stride2) - kernel3+1) / stride3) + 1
dim2 = int((((((im_size[1] - kernel1+1) / stride1) - kernel2+1) / stride2) - kernel3+1) / stride3) + 1

conv1_params = {'filters': caps1_nmaps*caps1_ndims, 'kernel_size': kernel1, 'strides': stride1, 'padding': 'valid'}
conv2_params = {'filters': caps1_nmaps*caps1_ndims, 'kernel_size': kernel2, 'strides': stride2, 'padding': 'valid'}
conv3_params = {'filters': caps1_nmaps*caps1_ndims, 'kernel_size': kernel3, 'strides': stride3, 'padding': 'valid'}
flags.DEFINE_list('conv_params', [conv1_params, conv2_params, conv3_params], 'list with the conv parameters')


flags.DEFINE_integer('caps1_nmaps', caps1_nmaps, 'primary caps, number of feature maps')
flags.DEFINE_integer('caps1_ncaps', caps1_nmaps * dim1 * dim2, 'primary caps, number of caps')
flags.DEFINE_integer('caps1_ndims', caps1_ndims, 'primary caps, number of dims')


# Output caps:
flags.DEFINE_integer('caps2_ncaps', len(shape_types), 'second caps layer, number of caps')
flags.DEFINE_integer('caps2_ndims', 4, 'second caps layer, number of dims')


# Decoder reconstruction:
flags.DEFINE_string('rec_decoder_type', 'fc', 'use fc or conv layers for decoding (only with 3 conv layers)')
flags.DEFINE_integer('n_hidden_reconstruction_1', 512, 'size of hidden layer 1 in decoder')
flags.DEFINE_integer('n_hidden_reconstruction_2', 1024, 'size of hidden layer 2 in decoder')
flags.DEFINE_integer('n_output', im_size[0]*im_size[1], 'output size of the decoder')


###########################
#    Hyperparameters      #
###########################
# For training
flags.DEFINE_integer('batch_size', 48, 'batch size')
flags.DEFINE_float('learning_rate', 0.0004, 'chosen learning rate for training')
flags.DEFINE_float('learning_rate_decay_steps', 400, 'decay for cosine decay restart')

flags.DEFINE_integer('n_epochs', None, 'number of epochs, if None allow for indifinite readings')
flags.DEFINE_integer('n_steps', 4800, 'number of steps')
flags.DEFINE_integer('n_rounds', 1, 'number of evaluations; full training steps is equal to n_steps times this number')
flags.DEFINE_integer('n_iterations', 20, 'number of trained networks')

flags.DEFINE_integer('buffer_size', 1024, 'buffer size')
flags.DEFINE_integer('eval_steps', 50, 'frequency for eval spec; u need at least eval_steps*batch_size stimuli in the validation set')
flags.DEFINE_integer('eval_throttle_secs', 150, 'minimal seconds between evaluation passes')
flags.DEFINE_integer('iter_routing', 3, 'number of iterations in routing algorithm')
flags.DEFINE_float('init_sigma', 0.01, 'stddev for W initializer')


###########################
#         Losses          #
###########################
flags.DEFINE_boolean('decode_reconstruction', False, 'decode the reconstruction and use reconstruction loss')

flags.DEFINE_boolean('decode_nshapes', True, 'decode the number of shapes and use nshapes loss')
nshapes_loss = 'xentropy'
flags.DEFINE_string('nshapes_loss', nshapes_loss, 'currently either xentropy or squared_diff')

flags.DEFINE_boolean('decode_location', False, 'decode the shapes locations and use location loss')
location_loss = 'xentropy'
flags.DEFINE_string('location_loss', location_loss, 'currently either xentropy or squared_diff')


# Control magnitude of losses
flags.DEFINE_float('alpha_vernieroffset', 1., 'alpha for vernieroffset loss')
flags.DEFINE_float('alpha_margin', 0.5, 'alpha for margin loss')
flags.DEFINE_float('alpha_shape_1_reconstruction', 0.0005, 'alpha for reconstruction loss for vernier image (reduce_sum)')
flags.DEFINE_float('alpha_shape_2_reconstruction', 0.0001, 'alpha for reconstruction loss for shape image (reduce_sum)')

if nshapes_loss=='xentropy':
    flags.DEFINE_float('alpha_nshapes', 0.4, 'alpha for nshapes loss')
elif nshapes_loss=='squared_diff':
    flags.DEFINE_float('alpha_nshapes', 0.002, 'alpha for nshapes loss')

if location_loss=='xentropy':
    flags.DEFINE_float('alpha_x_shape_1_loss', 0.1, 'alpha for loss of x coordinate of shape')
    flags.DEFINE_float('alpha_y_shape_1_loss', 0.1, 'alpha for loss of y coordinate of shape')
    flags.DEFINE_float('alpha_x_shape_2_loss', 0.1, 'alpha for loss of x coordinate of vernier')
    flags.DEFINE_float('alpha_y_shape_2_loss', 0.1, 'alpha for loss of y coordinate of vernier')
elif location_loss=='squared_diff':
    flags.DEFINE_float('alpha_x_shape_1_loss', 0.000004, 'alpha for loss of x coordinate of shape')
    flags.DEFINE_float('alpha_y_shape_1_loss', 0.00005, 'alpha for loss of y coordinate of shape')
    flags.DEFINE_float('alpha_x_shape_2_loss', 0.000004, 'alpha for loss of x coordinate of vernier')
    flags.DEFINE_float('alpha_y_shape_2_loss', 0.00005, 'alpha for loss of y coordinate of vernier')


# Margin loss extras
flags.DEFINE_float('m_plus', 0.9, 'the parameter of m plus')
flags.DEFINE_float('m_minus', 0.1, 'the parameter of m minus')
flags.DEFINE_float('lambda_val', 0.5, 'down weight of the loss for absent digit classes')


###########################
#     Regularization       #
###########################

flags.DEFINE_boolean('dropout', True, 'use dropout after conv layers 1&2')
flags.DEFINE_boolean('batch_norm_conv', False, 'use batch normalization between every conv layer')
flags.DEFINE_boolean('batch_norm_reconstruction', False, 'use batch normalization for the reconstruction decoder layers')
flags.DEFINE_boolean('batch_norm_vernieroffset', False, 'use batch normalization for the vernieroffset loss layer')
flags.DEFINE_boolean('batch_norm_nshapes', False, 'use batch normalization for the nshapes loss layer')
flags.DEFINE_boolean('batch_norm_location', False, 'use batch normalization for the location loss layer')


parameters = tf.app.flags.FLAGS
