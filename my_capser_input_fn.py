# -*- coding: utf-8 -*-
"""
My script for the input fn that is working with tfrecords files
@author: Lynn

Last update on 15.11.2018
-> added requirements for nshapes and location loss
-> added some data augmentation (random noise & l/r, u/d flipping)
-> added num_repeat to None for training and drop_remainder=True
"""

import tensorflow as tf
import numpy as np
from my_parameters import parameters


###########################
#     Parse tfrecords:    #
###########################
def parse_tfrecords(serialized_data):
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
    return vernier_images, shape_images, shapelabels, nshapeslabels, vernierlabels, x_shape, y_shape, x_vernier, y_vernier


###########################
#     Input function:     #
###########################
def input_fn(filenames, train, parameters, buffer_size=1024):
    # Create a TensorFlow Dataset-object:
    dataset = tf.data.TFRecordDataset(filenames=filenames, num_parallel_reads=32)
    dataset = dataset.map(parse_tfrecords, num_parallel_calls=64)
    
    if train:
        # Read a buffer of the given size and randomly shuffle it:
        dataset = dataset.shuffle(buffer_size=buffer_size)
        
        # Allow for infinite reading of data
        num_repeat = parameters.n_epochs

    else:
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

    # reshape images (they were flattened when transformed into bytes)
    vernier_images = tf.reshape(vernier_images, [parameters.batch_size, parameters.im_size[0], parameters.im_size[1], parameters.im_depth])
    shape_images = tf.reshape(shape_images, [parameters.batch_size, parameters.im_size[0], parameters.im_size[1], parameters.im_depth])
    shapelabels = tf.reshape(shapelabels, [parameters.batch_size, 2])
    nshapeslabels = tf.reshape(nshapeslabels, [parameters.batch_size, 1])
    vernierlabels = tf.reshape(vernierlabels, [parameters.batch_size, 1])
    x_shape = tf.reshape(x_shape, [parameters.batch_size, 1])
    y_shape = tf.reshape(y_shape, [parameters.batch_size, 1])
    x_vernier = tf.reshape(x_vernier, [parameters.batch_size, 1])
    y_vernier = tf.reshape(y_vernier, [parameters.batch_size, 1])


    if train:
###############################################
#       Lets do some data augmentation:       #
###############################################
        # Add some random gaussian TRAINING noise (always):
        vernier_images = tf.add(vernier_images, tf.random_normal(
            shape=[parameters.batch_size, parameters.im_size[0], parameters.im_size[1], parameters.im_depth], mean=0.0,
            stddev=parameters.train_noise))
        shape_images = tf.add(shape_images, tf.random_normal(
            shape=[parameters.batch_size, parameters.im_size[0], parameters.im_size[1], parameters.im_depth], mean=0.0,
            stddev=parameters.train_noise))
        
        # Flipping (code is somewhat messy, but the following calculations r used):
        # - change vernierlabels: abs(vernierlabels - 1)
        # - change coordinates shape / coordinates vernier (nshapes=1):
        #       - x: im_size[1] - (x + nshapes*shapesize)
        #       - y: im_size[0] - (y + shapesize)

        rnd_idx = np.random.randint(0, 4)

        # flip left-right:
        if rnd_idx==1:
            vernier_images = tf.image.flip_left_right(vernier_images)
            shape_images = tf.image.flip_left_right(shape_images)
            vernierlabels = tf.abs(tf.subtract(vernierlabels, 1))
            x_shape = tf.subtract(tf.constant(parameters.im_size[1], tf.int64), tf.add(x_shape,
                                  tf.multiply(nshapeslabels, parameters.shape_size)))
            x_vernier = tf.subtract(tf.constant(parameters.im_size[1], tf.int64), tf.add(x_vernier, parameters.shape_size))
        # flip up-down:
        elif rnd_idx==2:
            vernier_images = tf.image.flip_up_down(vernier_images)
            shape_images = tf.image.flip_up_down(shape_images)
            vernierlabels = tf.abs(tf.subtract(vernierlabels, 1))
            y_shape = tf.subtract(tf.constant(parameters.im_size[0], tf.int64), tf.add(y_shape, parameters.shape_size))
            y_vernier = tf.subtract(tf.constant(parameters.im_size[0], tf.int64), tf.add(y_vernier, parameters.shape_size))
        # flip left-right and up-down:
        elif rnd_idx==3:
            vernier_images = tf.image.flip_up_down(tf.image.flip_left_right(vernier_images))
            shape_images = tf.image.flip_up_down(tf.image.flip_left_right(shape_images))
            x_shape = tf.subtract(tf.constant(parameters.im_size[1], tf.int64), tf.add(x_shape,
                                  tf.multiply(nshapeslabels, parameters.shape_size)))
            y_shape = tf.subtract(tf.constant(parameters.im_size[0], tf.int64), tf.add(y_shape, parameters.shape_size))
            x_vernier = tf.subtract(tf.constant(parameters.im_size[1], tf.int64), tf.add(x_vernier, parameters.shape_size))
            y_vernier = tf.subtract(tf.constant(parameters.im_size[0], tf.int64), tf.add(y_vernier, parameters.shape_size))


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
        # For the test and validation set, we dont really need data augmentation,
        # but we'd still like some TEST noise
        vernier_images = tf.add(vernier_images, tf.random_normal(
            shape=[parameters.batch_size, parameters.im_size[0], parameters.im_size[1], parameters.im_depth], mean=0.0,
            stddev=parameters.test_noise))
        shape_images = tf.add(shape_images, tf.random_normal(
            shape=[parameters.batch_size, parameters.im_size[0], parameters.im_size[1], parameters.im_depth], mean=0.0,
            stddev=parameters.test_noise))

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
    rnd_idx = np.random.randint(0, len(parameters.shape_types))
    eval_file = parameters.test_data_paths[rnd_idx] + '.tfrecords'
    return input_fn(filenames=eval_file, train=False, parameters=parameters)


def predict_input_fn(filenames):
    return input_fn(filenames=filenames, train=False, parameters=parameters)
