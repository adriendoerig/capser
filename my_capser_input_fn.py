# -*- coding: utf-8 -*-
"""
My script for the input fn that is working with tfrecords files
@author: Lynn

Last update on 12.12.2018
-> added requirements for nshapes and location loss
-> added num_repeat to None for training and drop_remainder=True (requires at least tf version 1.10.0)
-> added data augmentation (noise, flipping, contrast, brightness)
-> train and test noise is randomly changed now between a lower and upper border
"""

import tensorflow as tf
import numpy as np
from my_parameters import parameters


########################################
#     Parse tfrecords training set:    #
########################################
def parse_tfrecords_train(serialized_data):
    # Define a dict with the data-names and types we expect to find in the TFRecords file.
    features = {'vernier_images': tf.FixedLenFeature([], tf.string),
                'shape_images': tf.FixedLenFeature([], tf.string),
                'shapelabels': tf.FixedLenFeature([], tf.string),
                'nshapeslabels': tf.FixedLenFeature([], tf.string),
                'vernierlabels': tf.FixedLenFeature([], tf.string),
                'x_shape': tf.FixedLenFeature([], tf.string),
                'y_shape': tf.FixedLenFeature([], tf.string),
                'x_vernier': tf.FixedLenFeature([], tf.string),
                'y_vernier': tf.FixedLenFeature([], tf.string)}

    # Parse the serialized data so we get a dict with our data.
    parsed_data = tf.parse_single_example(serialized=serialized_data, features=features)

    # Get the images as raw bytes and decode afterwards.
    vernier_images = parsed_data['vernier_images']
    vernier_images = tf.decode_raw(vernier_images, tf.float32)
    vernier_images = tf.cast(vernier_images, tf.float32)

    shape_images = parsed_data['shape_images']
    shape_images = tf.decode_raw(shape_images, tf.float32)
    shape_images = tf.cast(shape_images, tf.float32)
    
    # Get the labels associated with the image and decode.
    shapelabels = parsed_data['shapelabels']
    shapelabels = tf.decode_raw(shapelabels, tf.float32)
    shapelabels = tf.cast(shapelabels, tf.int64)
    
    nshapeslabels = parsed_data['nshapeslabels']
    nshapeslabels = tf.decode_raw(nshapeslabels, tf.float32)
    nshapeslabels = tf.cast(nshapeslabels, tf.int64)
    
    vernierlabels = parsed_data['vernierlabels']
    vernierlabels = tf.decode_raw(vernierlabels, tf.float32)
    vernierlabels = tf.cast(vernierlabels, tf.int64)
    
    x_shape = parsed_data['x_shape']
    x_shape = tf.decode_raw(x_shape, tf.float32)
    x_shape = tf.cast(x_shape, tf.int64)
    
    y_shape = parsed_data['y_shape']
    y_shape = tf.decode_raw(y_shape, tf.float32)
    y_shape = tf.cast(y_shape, tf.int64)
    
    x_vernier = parsed_data['x_vernier']
    x_vernier = tf.decode_raw(x_vernier, tf.float32)
    x_vernier = tf.cast(x_vernier, tf.int64)
    
    y_vernier = parsed_data['y_vernier']
    y_vernier = tf.decode_raw(y_vernier, tf.float32)
    y_vernier = tf.cast(y_vernier, tf.int64)

    # Reshaping:
    vernier_images = tf.reshape(vernier_images, [parameters.im_size[0], parameters.im_size[1], parameters.im_depth])
    shape_images = tf.reshape(shape_images, [parameters.im_size[0], parameters.im_size[1], parameters.im_depth])
    shapelabels = tf.reshape(shapelabels, [2])
    nshapeslabels = tf.reshape(nshapeslabels, [1])
    vernierlabels = tf.reshape(vernierlabels, [1])
    x_shape = tf.reshape(x_shape, [1])
    y_shape = tf.reshape(y_shape, [1])
    x_vernier = tf.reshape(x_vernier, [1])
    y_vernier = tf.reshape(y_vernier, [1])


    ##################################
    #       Data augmentation:       #
    ##################################
    # Add some random gaussian TRAINING noise (always):
    noise1 = tf.random_uniform([1], parameters.train_noise[0], parameters.train_noise[1], tf.float32)
    noise2 = tf.random_uniform([1], parameters.train_noise[0], parameters.train_noise[1], tf.float32)
    vernier_images = tf.add(vernier_images, tf.random_normal(
        shape=[parameters.im_size[0], parameters.im_size[1], parameters.im_depth], mean=0.0,
        stddev=noise1))
    shape_images = tf.add(shape_images, tf.random_normal(
        shape=[parameters.im_size[0], parameters.im_size[1], parameters.im_depth], mean=0.0,
        stddev=noise2))


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

    # Maybe change contrast and brightness:
    pred = tf.less(tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32), 0.5)
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
        x_shape_flipped = tf.subtract(tf.constant(parameters.im_size[1], tf.int64), tf.add(x_shape,
                              tf.multiply(nshapeslabels, parameters.shape_size)))
        y_shape_flipped = y_shape
        x_vernier_flipped = tf.subtract(tf.constant(parameters.im_size[1], tf.int64), tf.add(x_vernier, parameters.shape_size))
        y_vernier_flipped = y_vernier
        return [vernier_images_flipped, shape_images_flipped, vernierlabels_flipped,
                x_shape_flipped, y_shape_flipped, x_vernier_flipped, y_vernier_flipped]
    
    # flip up-down function:
    def flip2():
        vernier_images_flipped = tf.image.flip_up_down(vernier_images)
        shape_images_flipped = tf.image.flip_up_down(shape_images)
        vernierlabels_flipped = tf.abs(tf.subtract(vernierlabels, 1))
        x_shape_flipped = x_shape
        y_shape_flipped = tf.subtract(tf.constant(parameters.im_size[0], tf.int64), tf.add(y_shape, parameters.shape_size))
        x_vernier_flipped = x_vernier
        y_vernier_flipped = tf.subtract(tf.constant(parameters.im_size[0], tf.int64), tf.add(y_vernier, parameters.shape_size))
        return [vernier_images_flipped, shape_images_flipped, vernierlabels_flipped,
                x_shape_flipped, y_shape_flipped, x_vernier_flipped, y_vernier_flipped]
    
    # tf flip functions need 4D inputs:
    vernier_images = tf.expand_dims(vernier_images, 0)
    shape_images = tf.expand_dims(shape_images, 0)

    # Maybe flip left-right:
    pred_flip1 = tf.less(tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32), 0.5)
    vernier_images, shape_images, vernierlabels, x_shape, y_shape, x_vernier, y_vernier = tf.cond(pred_flip1, flip0, flip1)
    
    # Maybe flip up-down:
    pred_flip2 = tf.less(tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32), 0.5)
    vernier_images, shape_images, vernierlabels, x_shape, y_shape, x_vernier, y_vernier = tf.cond(pred_flip2, flip0, flip2)
    
    # Get rid of extra-dimension:
    vernier_images = tf.squeeze(vernier_images, axis=0)
    shape_images = tf.squeeze(shape_images, axis=0)

    return vernier_images, shape_images, shapelabels, nshapeslabels, vernierlabels, x_shape, y_shape, x_vernier, y_vernier


########################################
#      Parse tfrecords test set:       #
########################################
def parse_tfrecords_test(serialized_data):
    # Define a dict with the data-names and types we expect to find in the TFRecords file.
    features = {'vernier_images': tf.FixedLenFeature([], tf.string),
                'shape_images': tf.FixedLenFeature([], tf.string),
                'shapelabels': tf.FixedLenFeature([], tf.string),
                'nshapeslabels': tf.FixedLenFeature([], tf.string),
                'vernierlabels': tf.FixedLenFeature([], tf.string),
                'x_shape': tf.FixedLenFeature([], tf.string),
                'y_shape': tf.FixedLenFeature([], tf.string),
                'x_vernier': tf.FixedLenFeature([], tf.string),
                'y_vernier': tf.FixedLenFeature([], tf.string)}

    # Parse the serialized data so we get a dict with our data.
    parsed_data = tf.parse_single_example(serialized=serialized_data, features=features)

    # Get the images as raw bytes and decode afterwards.
    vernier_images = parsed_data['vernier_images']
    vernier_images = tf.decode_raw(vernier_images, tf.float32)
    vernier_images = tf.cast(vernier_images, tf.float32)

    shape_images = parsed_data['shape_images']
    shape_images = tf.decode_raw(shape_images, tf.float32)
    shape_images = tf.cast(shape_images, tf.float32)
    
    # Get the labels associated with the image and decode.
    shapelabels = parsed_data['shapelabels']
    shapelabels = tf.decode_raw(shapelabels, tf.float32)
    shapelabels = tf.cast(shapelabels, tf.int64)
    
    nshapeslabels = parsed_data['nshapeslabels']
    nshapeslabels = tf.decode_raw(nshapeslabels, tf.float32)
    nshapeslabels = tf.cast(nshapeslabels, tf.int64)
    
    vernierlabels = parsed_data['vernierlabels']
    vernierlabels = tf.decode_raw(vernierlabels, tf.float32)
    vernierlabels = tf.cast(vernierlabels, tf.int64)
    
    x_shape = parsed_data['x_shape']
    x_shape = tf.decode_raw(x_shape, tf.float32)
    x_shape = tf.cast(x_shape, tf.int64)
    
    y_shape = parsed_data['y_shape']
    y_shape = tf.decode_raw(y_shape, tf.float32)
    y_shape = tf.cast(y_shape, tf.int64)
    
    x_vernier = parsed_data['x_vernier']
    x_vernier = tf.decode_raw(x_vernier, tf.float32)
    x_vernier = tf.cast(x_vernier, tf.int64)
    
    y_vernier = parsed_data['y_vernier']
    y_vernier = tf.decode_raw(y_vernier, tf.float32)
    y_vernier = tf.cast(y_vernier, tf.int64)
    
    # Reshaping:
    vernier_images = tf.reshape(vernier_images, [parameters.im_size[0], parameters.im_size[1], parameters.im_depth])
    shape_images = tf.reshape(shape_images, [parameters.im_size[0], parameters.im_size[1], parameters.im_depth])
    shapelabels = tf.reshape(shapelabels, [2])
    nshapeslabels = tf.reshape(nshapeslabels, [1])
    vernierlabels = tf.reshape(vernierlabels, [1])
    x_shape = tf.reshape(x_shape, [1])
    y_shape = tf.reshape(y_shape, [1])
    x_vernier = tf.reshape(x_vernier, [1])
    y_vernier = tf.reshape(y_vernier, [1])
    
    # For the test and validation set, we dont really need data augmentation,
    # but we'd still like some TEST noise
    noise1 = tf.random_uniform([1], parameters.test_noise[0], parameters.test_noise[1], tf.float32)
    noise2 = tf.random_uniform([1], parameters.test_noise[0], parameters.test_noise[1], tf.float32)
    vernier_images = tf.add(vernier_images, tf.random_normal(
        shape=[parameters.im_size[0], parameters.im_size[1], parameters.im_depth], mean=0.0,
        stddev=noise1))
    shape_images = tf.add(shape_images, tf.random_normal(
        shape=[parameters.im_size[0], parameters.im_size[1], parameters.im_depth], mean=0.0,
        stddev=noise2))
    
    return vernier_images, shape_images, shapelabels, nshapeslabels, vernierlabels, x_shape, y_shape, x_vernier, y_vernier


###########################
#     Input function:     #
###########################
def input_fn(filenames, train, parameters, buffer_size=1024):
    # Create a TensorFlow Dataset-object:
    dataset = tf.data.TFRecordDataset(filenames=filenames, num_parallel_reads=32)
    
    # Currently, I am using two different functions for parsing the train and 
    # test/eval set due to data augmentation:
    if train:
        dataset = dataset.map(parse_tfrecords_train, num_parallel_calls=64)
        
        # Read a buffer of the given size and randomly shuffle it:
        dataset = dataset.shuffle(buffer_size=buffer_size)
        
        # Allow for infinite reading of data
        num_repeat = parameters.n_epochs

    else:
        dataset = dataset.map(parse_tfrecords_test, num_parallel_calls=64)
        
        # Don't shuffle the data and only go through the it once:
        num_repeat = 1
        
    # Repeat the dataset the given number of times and get a batch of data
    dataset = dataset.repeat(num_repeat)
    dataset = dataset.batch(parameters.batch_size, drop_remainder=True)
    
    # Use pipelining to speed up things (see https://www.youtube.com/watch?v=SxOsJPaxHME)
    dataset = dataset.prefetch(2)
    
    # Create an iterator for the dataset and the above modifications.
    iterator = dataset.make_one_shot_iterator()
    
    # Get the next batch of images and labels.
    [vernier_images, shape_images, shapelabels, nshapeslabels, vernierlabels, 
     x_shape, y_shape, x_vernier, y_vernier] = iterator.get_next()

    if train:
        feed_dict = {'vernier_images': vernier_images,
                     'shape_images': shape_images,
                     'shapelabels': shapelabels,
                     'nshapeslabels': nshapeslabels,
                     'vernier_offsets': vernierlabels,
                     'x_shape': x_shape,
                     'y_shape': y_shape,
                     'x_vernier': x_vernier,
                     'y_vernier': y_vernier,
                     'mask_with_labels': True,
                     'is_training': True}

    else:
        feed_dict = {'vernier_images': vernier_images,
                     'shape_images': shape_images,
                     'shapelabels': shapelabels,
                     'nshapeslabels': nshapeslabels,
                     'vernier_offsets': vernierlabels,
                     'x_shape': x_shape,
                     'y_shape': y_shape,
                     'x_vernier': x_vernier,
                     'y_vernier': y_vernier,
                     'mask_with_labels': False,
                     'is_training': False}
    return feed_dict, shapelabels



##############################
#   Final input functions:   #
##############################
def train_input_fn():
    return input_fn(filenames=parameters.train_data_path, train=True, parameters=parameters)


def eval_input_fn():
    rnd_idx = np.random.randint(0, len(parameters.test_data_paths))
    eval_file = parameters.test_data_paths[rnd_idx] + '.tfrecords'
    return input_fn(filenames=eval_file, train=False, parameters=parameters)


def predict_input_fn(filenames):
    return input_fn(filenames=filenames, train=False, parameters=parameters)
