# -*- coding: utf-8 -*-
"""
Test the data augmentation functions
Created on Wed Nov 28 15:05:45 2018
@author: Lynn
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from my_batchmaker import stim_maker_fn
from my_parameters import parameters

add_noise = 0
add_flip_up_down = 1
add_flip_left_right = 1
add_contrast = 0

print('--------------------------------------')
print('TF version:', tf.__version__)
print('Starting script...')
print('--------------------------------------')

tf.reset_default_graph()

imSize = parameters.im_size
shapeSize = parameters.shape_size
barWidth = parameters.bar_width
n_shapes = parameters.n_shapes
train_noise = parameters.train_noise
test_noise = parameters.test_noise
batch_size = 1
shape_types = parameters.shape_types
overlap = parameters.overlapping_shapes
test = stim_maker_fn(imSize, shapeSize, barWidth)


[train_vernier_images, train_shape_images, train_shapelabels, train_nshapeslabels, 
train_vernierlabels, train_x_shape, train_y_shape, train_x_vernier, train_y_vernier] = test.makeTrainBatch(
        shape_types, n_shapes, batch_size, train_noise, overlap=overlap)


##################################
#       Data augmentation:       #
##################################
def data_augmentation(vernier_images, shape_images, nshapeslabels, vernierlabels, x_shape, y_shape, x_vernier, y_vernier):
    vernier_images = tf.constant(vernier_images, tf.float32)
    shape_images = tf.constant(shape_images, tf.float32)
    nshapeslabels = tf.constant(nshapeslabels, tf.float32)
    vernierlabels = tf.constant(vernierlabels, tf.float32)
    x_shape = tf.constant(x_shape, tf.float32)
    y_shape = tf.constant(y_shape, tf.float32)
    x_vernier = tf.constant(x_vernier, tf.float32)
    y_vernier = tf.constant(y_vernier, tf.float32)
    
    # Add some random gaussian TRAINING noise:
    if add_noise:
        vernier_images = tf.add(vernier_images, tf.random_normal(
            shape=[parameters.im_size[0], parameters.im_size[1], parameters.im_depth], mean=0.0,
            stddev=parameters.train_noise))
        shape_images = tf.add(shape_images, tf.random_normal(
            shape=[parameters.im_size[0], parameters.im_size[1], parameters.im_depth], mean=0.0,
            stddev=parameters.train_noise))

    # Adjust brightness and contrast by a random factor
    def bright_contrast():
        vernier_images_augmented = tf.image.random_brightness(vernier_images, parameters.max_delta_brightness)
        shape_images_augmented = tf.image.random_brightness(shape_images, parameters.max_delta_brightness)
        vernier_images_augmented = tf.image.random_contrast(vernier_images_augmented,parameters.min_delta_contrast, parameters.max_delta_contrast)
        shape_images_augmented = tf.image.random_contrast(shape_images_augmented, parameters.min_delta_contrast, parameters.max_delta_contrast)
        return vernier_images_augmented, shape_images_augmented
    
    def contrast_bright():
        vernier_images_augmented = tf.image.random_contrast(vernier_images, parameters.min_delta_contrast, parameters.max_delta_contrast)
        shape_images_augmented = tf.image.random_contrast(shape_images, parameters.min_delta_contrast, parameters.max_delta_contrast)
        vernier_images_augmented = tf.image.random_brightness(vernier_images_augmented, parameters.max_delta_brightness)
        shape_images_augmented = tf.image.random_brightness(shape_images_augmented, parameters.max_delta_brightness)
        return vernier_images_augmented, shape_images_augmented

    # Maybe adjust contrast and brightness:
    if add_contrast:
        pred = tf.placeholder_with_default(False, shape=())
        vernier_images, shape_images = tf.cond(pred, bright_contrast, contrast_bright)

    # Flipping (code is messy since we r using tf cond atm, but this is the idea):
    # - change vernierlabels: abs(vernierlabels - 1)
    # - change shape coordinates / vernier coordinates:
    #       - x: im_size[1] - (x + nshapes*shapesize)
    #       - y: im_size[0] - (y + shapesize)

    # no flipping function:
    def flip0():
        vernier_images_flipped = vernier_images
        shape_images_flipped = shape_images
        vernierlabels_flipped = vernierlabels
        x_shape_flipped = x_shape
        y_shape_flipped = y_shape
        x_vernier_flipped = x_vernier
        y_vernier_flipped = y_vernier
        return [vernier_images_flipped, shape_images_flipped, vernierlabels_flipped,
                x_shape_flipped, y_shape_flipped, x_vernier_flipped, y_vernier_flipped]

    # flip left-right function:
    def flip1():
        vernier_images_flipped = tf.image.flip_left_right(vernier_images)
        shape_images_flipped = tf.image.flip_left_right(shape_images)
        vernierlabels_flipped = tf.abs(tf.subtract(vernierlabels, 1))
        x_shape_flipped = tf.subtract(tf.constant(parameters.im_size[1], tf.float32), tf.add(x_shape,
                              tf.multiply(nshapeslabels, tf.constant(parameters.shape_size, tf.float32))))
        y_shape_flipped = y_shape
        x_vernier_flipped = tf.subtract(tf.constant(parameters.im_size[1], tf.float32), tf.add(x_vernier, parameters.shape_size))
        y_vernier_flipped = y_vernier
        return [vernier_images_flipped, shape_images_flipped, vernierlabels_flipped,
                x_shape_flipped, y_shape_flipped, x_vernier_flipped, y_vernier_flipped]

    # flip up-down function:
    def flip2():
        vernier_images_flipped = tf.image.flip_up_down(vernier_images)
        shape_images_flipped = tf.image.flip_up_down(shape_images)
        vernierlabels_flipped = tf.abs(tf.subtract(vernierlabels, 1))
        x_shape_flipped = x_shape
        y_shape_flipped = tf.subtract(tf.constant(parameters.im_size[0], tf.float32), tf.add(y_shape, parameters.shape_size))
        x_vernier_flipped = x_vernier
        y_vernier_flipped = tf.subtract(tf.constant(parameters.im_size[0], tf.float32), tf.add(y_vernier, parameters.shape_size))
        return [vernier_images_flipped, shape_images_flipped, vernierlabels_flipped,
                x_shape_flipped, y_shape_flipped, x_vernier_flipped, y_vernier_flipped]
    
    if add_flip_left_right:
        # Maybe flip left-right:
        pred_flip1 = tf.placeholder_with_default(False, shape=())
        vernier_images, shape_images, vernierlabels, x_shape, y_shape, x_vernier, y_vernier = tf.cond(pred_flip1, flip0, flip1)
    
    if add_flip_up_down:
        # Maybe flip up-down:
        pred_flip2 = tf.placeholder_with_default(False, shape=())
        vernier_images, shape_images, vernierlabels, x_shape, y_shape, x_vernier, y_vernier = tf.cond(pred_flip2, flip0, flip2)

    return vernier_images, shape_images, nshapeslabels, vernierlabels, x_shape, y_shape, x_vernier, y_vernier


####################################
#      Run the data augment:       #
####################################
with tf.Session() as sess: 
    [vernier_images_flip, shape_images_flip, nshapeslabels_flip, vernierlabels_flip, 
     x_shape_flip, y_shape_flip, x_vernier_flip, y_vernier_flip] = data_augmentation(
     train_vernier_images, train_shape_images, train_nshapeslabels, train_vernierlabels, 
     train_x_shape, train_y_shape, train_x_vernier, train_y_vernier)
    
    vernier_images_flip = vernier_images_flip.eval()
    shape_images_flip = shape_images_flip.eval()
    nshapeslabels_flip = nshapeslabels_flip.eval()
    vernierlabels_flip = vernierlabels_flip.eval()
    x_shape_flip = x_shape_flip.eval()
    y_shape_flip = y_shape_flip.eval()
    x_vernier_flip = x_vernier_flip.eval()
    y_vernier_flip = y_vernier_flip.eval()


####################################
#           Plotting:              #
####################################
train_vernier_images[:, int(train_y_vernier), int(train_x_vernier), :] = 2
vernier_images_flip[:, int(y_vernier_flip), int(x_vernier_flip), :] = 2
train_shape_images[:, int(train_y_shape), int(train_x_shape), :] = 2
shape_images_flip[:, int(y_shape_flip), int(x_shape_flip), :] = 2

plt.figure()
n_rows = 2
n_cols = 2
plt.subplot(n_rows, n_cols, 1)
plt.axis('off')
plt.imshow(np.squeeze(train_vernier_images))

plt.subplot(n_rows, n_cols, 2)
plt.axis('off')
plt.imshow(np.squeeze(vernier_images_flip))

plt.subplot(n_rows, n_cols, 3)
plt.axis('off')
plt.imshow(np.squeeze(train_shape_images))

plt.subplot(n_rows, n_cols, 4)
plt.axis('off')
plt.imshow(np.squeeze(shape_images_flip))

print('--------------------------------------')
print('\n... Script finished!')
print('--------------------------------------')
