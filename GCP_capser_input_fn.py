# -*- coding: utf-8 -*-
"""
My script for the input fn that is working with tfrecords files
Last update on 31.10.2018
@author: Lynn
"""

import tensorflow as tf
import numpy as np
from GCP_parameters import n_epochs, batch_size, im_size, train_noise, test_noise, \
train_data_path, test_data_paths, shape_types


###########################
#     Parse tfrecords:    #
###########################
def parse_tfrecords(serialized_data):
    # Define a dict with the data-names and types we expect to find in the TFRecords file.
    features = {'vernier_images': tf.FixedLenFeature([], tf.string),
                'shape_images': tf.FixedLenFeature([], tf.string),
                'shapelabels': tf.FixedLenFeature([], tf.string),
                'nshapeslabels': tf.FixedLenFeature([], tf.string),
                'vernierlabels': tf.FixedLenFeature([], tf.string)}

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
    return vernier_images, shape_images, shapelabels, nshapeslabels, vernierlabels


###########################
#     Input function:     #
###########################
def input_fn(filenames, train, n_epochs, batch_size, im_size, buffer_size=1024):
    # Create a TensorFlow Dataset-object:
    dataset = tf.data.TFRecordDataset(filenames=filenames, num_parallel_reads=32)
    dataset = dataset.map(parse_tfrecords, num_parallel_calls=64)
    
    if train:
        # Read a buffer of the given size and randomly shuffle it:
        dataset = dataset.shuffle(buffer_size=buffer_size)
        
        # Allow for infinite reading of data
        num_repeat = n_epochs

    else:
        # Don't shuffle the data and only go through the it once:
        num_repeat = 1
        
    # Repeat the dataset the given number of times and get a batch of data
    dataset = dataset.repeat(num_repeat)
    dataset = dataset.batch(batch_size)
    
    # Use pipelining to speed up things (see https://www.youtube.com/watch?v=SxOsJPaxHME)
    dataset = dataset.prefetch(2)
    
    # Create an iterator for the dataset and the above modifications.
    iterator = dataset.make_one_shot_iterator()
    
    # Get the next batch of images and labels.
    vernier_images, shape_images, shapelabels, nshapeslabels, vernierlabels = iterator.get_next()

    # reshape images (they were flattened when transformed into bytes)
    vernier_images = tf.reshape(vernier_images, [batch_size, im_size[0], im_size[1], 1])
    shape_images = tf.reshape(shape_images, [batch_size, im_size[0], im_size[1], 1])
    shapelabels = tf.reshape(shapelabels, [batch_size, 2])
    nshapeslabels = tf.reshape(nshapeslabels, [batch_size, 1])
    vernierlabels = tf.reshape(vernierlabels, [batch_size, 1])

    # The input-function must return a dict wrapping the images.
    if train:
        # Add some random gaussian noise:
        vernier_images = tf.add(vernier_images, tf.random_normal(
            shape=[batch_size, im_size[0], im_size[1], 1], mean=0.0,
            stddev=train_noise))
        shape_images = tf.add(shape_images, tf.random_normal(
            shape=[batch_size, im_size[0], im_size[1], 1], mean=0.0,
            stddev=train_noise))
        feed_dict = {'vernier_images': vernier_images,
                     'shape_images': shape_images,
                     'shapelabels': shapelabels,
                     'nshapeslabels': nshapeslabels,
                     'vernier_offsets': vernierlabels,
                     'mask_with_labels': True,
                     'is_training': True}
    else:
        # Add some random gaussian noise:
        vernier_images = tf.add(vernier_images, tf.random_normal(
            shape=[batch_size, im_size[0], im_size[1], 1], mean=0.0,
            stddev=test_noise))
        shape_images = tf.add(shape_images, tf.random_normal(
            shape=[batch_size, im_size[0], im_size[1], 1], mean=0.0,
            stddev=test_noise))
        feed_dict = {'vernier_images': vernier_images,
                     'shape_images': shape_images,
                     'shapelabels': shapelabels,
                     'nshapeslabels': nshapeslabels,
                     'vernier_offsets': vernierlabels,
                     'mask_with_labels': False,
                     'is_training': False}
    return feed_dict, shapelabels



##############################
#   Final input functions:   #
##############################
def train_input_fn():
    return input_fn(train_data_path, True, n_epochs, batch_size, im_size)


def eval_input_fn():
    rnd_idx = np.random.randint(0, len(shape_types))
    eval_file = test_data_paths[rnd_idx] + '.tfrecords'
    return input_fn(eval_file, False, n_epochs, batch_size, im_size)


def predict_input_fn(filenames):
    return input_fn(filenames, False, n_epochs, batch_size, im_size)
