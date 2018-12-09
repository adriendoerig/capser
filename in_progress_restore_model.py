# -*- coding: utf-8 -*-
"""
First attempts to get some visualization of the trained model

@author: Lynn

Last update: 06.12.18
-> some improvements of the code
"""

meta_path = '.\data\_log'
meta_file = '\model.ckpt-60000.meta'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from my_batchmaker import stim_maker_fn
from my_parameters import parameters


plotflag_kernels = 0
plotflag_conv_output = 1
train_flag = 0

test_shape = 3
batch_size = 1

plt.close('all')

print('--------------------------------------')
print('TF version:', tf.__version__)
print('Starting restoration script...')
print('--------------------------------------')


tf.reset_default_graph()  
imported_meta = tf.train.import_meta_graph(meta_path + meta_file)

test = stim_maker_fn(parameters.im_size, parameters.shape_size, parameters.bar_width)

[train_vernier_images, train_shape_images, train_shapelabels, train_nshapeslabels, 
train_vernierlabels, train_x_shape, train_y_shape, train_x_vernier, train_y_vernier] = test.makeTrainBatch(
        parameters.shape_types, parameters.n_shapes, batch_size, parameters.train_noise, 
        overlap=parameters.overlapping_shapes)

[test_vernier_images, test_shape_images, test_shapelabels, test_nshapeslabels, 
test_vernierlabels, test_x_shape, test_y_shape, test_x_vernier, test_y_vernier] = test.makeTestBatch(
 test_shape, parameters.n_shapes, batch_size, None, parameters.test_noise)

if train_flag:
    input_img = train_vernier_images + train_shape_images
else:
    input_img = test_vernier_images + test_shape_images


with tf.Session() as sess:  
    imported_meta.restore(sess, tf.train.latest_checkpoint(meta_path))

    # accessing the default graph which we restored
    graph = tf.get_default_graph()
    
    # get all names of ops
    all_ops = graph.get_operations()
    n_ops = len(all_ops)
    
    # get kernels and biases for all conv layers:
    conv1_kernel_tf = graph.get_tensor_by_name('conv1/kernel:0')
#    conv1_kernel_tf2 = graph.get_tensor_by_name('1_convolutional_layers/conv1/Conv2D:0')
#    conv1_kernel_tf3 = graph.get_tensor_by_name('1_convolutional_layers/conv1_output/tag:0')
    conv1_kernel = conv1_kernel_tf.eval()
#    conv1_kernel2 = conv1_kernel_tf2.eval()
#    conv1_kernel3 = conv1_kernel_tf3.eval()
    conv1_bias_tf = graph.get_tensor_by_name('conv1/bias:0')
    conv1_bias = conv1_bias_tf.eval()
    
    conv2_kernel_tf = graph.get_tensor_by_name('conv2/kernel:0')
    conv2_kernel = conv2_kernel_tf.eval()
    conv2_bias_tf = graph.get_tensor_by_name('conv2/bias:0')
    conv2_bias = conv2_bias_tf.eval()
    
    conv3_kernel_tf = graph.get_tensor_by_name('conv3/kernel:0')
    conv3_kernel = conv3_kernel_tf.eval()
    conv3_bias_tf = graph.get_tensor_by_name('conv3/bias:0')
    conv3_bias = conv3_bias_tf.eval()
    
    input_img_tf = tf.constant(input_img, tf.float32)
    if train_flag:
        input_img_tf = tf.add(input_img_tf, tf.random_normal(
            shape=[parameters.im_size[0], parameters.im_size[1], parameters.im_depth], mean=0.0,
            stddev=parameters.train_noise))
    else:
        input_img_tf = tf.add(input_img_tf, tf.random_normal(
            shape=[parameters.im_size[0], parameters.im_size[1], parameters.im_depth], mean=0.0,
            stddev=parameters.test_noise))
    input_img = input_img_tf.eval()
    
    # convolve input and final filters:
    conv1_output_tf = tf.nn.conv2d(input_img_tf, conv1_kernel_tf, [1, 1, 1, 1], 'VALID')
    conv1_output_tf = tf.nn.relu(conv1_output_tf)
    conv1_output = conv1_output_tf.eval()
    
    conv2_output_tf = tf.nn.conv2d(conv1_output_tf, conv2_kernel_tf, [1, 2, 2, 1], 'VALID')
    conv2_output_tf = tf.nn.relu(conv2_output_tf)
    conv2_output = conv2_output_tf.eval()
    
    conv3_output_tf = tf.nn.conv2d(conv2_output_tf, conv3_kernel_tf, [1, 2, 2, 1], 'VALID')
    conv3_output_tf = tf.nn.relu(conv3_output_tf)
    conv3_output = conv3_output_tf.eval()


# Plot kernels in a big subplot:
def plot_kernels(kernels):
    plt.figure()
    n_maps = kernels.shape[2] * kernels.shape[3]
    kernels = np.reshape(kernels, [kernels.shape[0], kernels.shape[1], n_maps])
    n_rows = np.ceil(np.sqrt(n_maps))
    n_cols = np.floor(np.sqrt(n_maps))
    for i in range(n_maps):
        plt.subplot(n_rows, n_cols, i+1)
        plt.axis('off')
        plt.imshow(np.squeeze(kernels[:, :, i]), interpolation='nearest', cmap='gray')


def plot_conv_output(input_img, conv_output):
    plt.figure()
    n_maps = conv_output.shape[3]
    n_rows = np.ceil(np.sqrt(n_maps+1))
    n_cols = np.floor(np.sqrt(n_maps+1))
    plt.subplot(n_rows, n_cols, 1)
    plt.axis('off')
    plt.imshow(np.squeeze(input_img), interpolation='nearest', cmap='gray')
    for i in range(1, n_maps):
        plt.subplot(n_rows, n_cols, i+1)
        plt.axis('off')
        plt.imshow(np.squeeze(conv_output[0, :, :, i]), interpolation='nearest', cmap='gray')



if plotflag_kernels:
    plot_kernels(conv1_kernel)
    plot_kernels(conv2_kernel)
    plot_kernels(conv3_kernel)

if plotflag_conv_output:
    plot_conv_output(input_img, conv1_output)
    plot_conv_output(input_img, conv2_output)
    plot_conv_output(input_img, conv3_output)


print('--------------------------------------')
print('Finished...')
print('--------------------------------------')