# -*- coding: utf-8 -*-
"""
My capsnet: all parameters
Last update on 31.10.2018
@author: Lynn
"""


###########################
#          Paths          #
###########################
data_path = 'gs://lynns_capser'
train_data_path = data_path + '/train.tfrecords'
test_data_paths = [data_path + '/test_squares',
                   data_path + '/test_circles',
                   data_path + '/test_hectagon',
                   data_path + '/test_4stars',
                   data_path + '/test_stars',
                   data_path + '/test_squares_stars']
MODEL_NAME = 'GCP_log'
logdir_path = data_path + '/' + MODEL_NAME + '/'


###########################
#     Reproducibility     #
###########################
random_seed = False
# Note: seed=42 usually leads to very bad results (= no vernier offset discrimination)
# Contrary to that seed=1 leads to good results


###########################
#   Stimulus parameters   #
###########################
batch_size = 32
n_steps = 20000
eval_freq = 50
n_train_samples = 100000
n_test_samples = 3200

im_size = [60, 150]
shape_types = [0, 1, 2, 3, 4, 5]
im_depth = 1
shape_size = 20
bar_width = 1
n_shapes = [1, 3, 5, 7]

# Can still be changed after creation of tfrecords file:
train_noise = 0.025
test_noise = 0.05
only_venier = False
only_venier_percent = 0.2


###########################
#   Network parameters    #
###########################
caps1_nmaps = 12
caps1_ndims = 10
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
                'padding': 'valid'}

conv2_params = {'filters': caps1_nmaps*caps1_ndims, 'kernel_size': kernel2, 'strides': stride2,
                'padding': 'valid'}

conv3_params = {'filters': caps1_nmaps*caps1_ndims, 'kernel_size': kernel3, 'strides': stride3,
                'padding': 'valid'}

caps1_ncaps = caps1_nmaps * dim1 * dim2

# Shape caps: 10 capsules (each shape) of 16 dimensions each
caps2_ncaps = len(shape_types)
caps2_ndims = 16

# Decoder:
n_hidden1 = 512
n_hidden2 = 1024
n_output = im_size[0]*im_size[1]


###########################
#    Hyperparameters      #
###########################
# Margin loss
m_plus = 0.9
m_minus = 0.1
lambda_val = 0.5

# For training
buffer_size = 1024
n_epochs = 1
iter_routing = 3
init_sigma = 0.01

# Learning rate
exp_learning_decay = True
learning_rate = 0.001
decay_steps = 500
decay_rate = 0.90

# Losses
# flags.DEFINE_float('regularization_scale', 0.0005*im_size[0]*im_size[1], 'regularization coefficient for reconstruction loss, default to 0.0005*im_size[0]*im_size[1] (reduce_mean)')
alpha_margin = 1.
alpha_vernier_reconstruction = 0.0005
alpha_shape_reconstruction = 0.0001
alpha_vernieroffset = 2.

# Regulariztion:
batch_norm_conv = True
batch_norm_decoder = True

