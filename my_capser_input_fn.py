# -*- coding: utf-8 -*-
"""
My script for the input fn that is working with tfrecords files
Last update on 24.10.2018
@author: Lynn
"""

import tensorflow as tf
from my_parameters import parameters


###########################
#     Parse tfrecords:    #
###########################
def parse_tfrecords(serialized_data):
    # Define a dict with the data-names and types we expect to
    # find in the TFRecords file.
    features = {'images': tf.FixedLenFeature([], tf.string),
                'shapelabels': tf.FixedLenFeature([], tf.string),
                'nshapeslabels': tf.FixedLenFeature([], tf.string),
                'vernierlabels': tf.FixedLenFeature([], tf.string)}

    # Parse the serialized data so we get a dict with our data.
    parsed_data = tf.parse_single_example(serialized=serialized_data, features=features)

    # Get the image as raw bytes and decode afterwards.
    images = parsed_data['images']
    images = tf.decode_raw(images, tf.float32)
    
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
    return images, shapelabels, nshapeslabels, vernierlabels


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
        num_repeat = None

    else:
        # Don't shuffle the data and only go through the it once:
        num_repeat = 1
        
    # Repeat the dataset the given number of times and get a batch of data
    dataset = dataset.repeat(num_repeat)
    dataset = dataset.batch(parameters.batch_size)
    
    # Use pipelining to speed up things (see https://www.youtube.com/watch?v=SxOsJPaxHME)
    dataset = dataset.prefetch(3)
    
    # Create an iterator for the dataset and the above modifications.
    iterator = dataset.make_one_shot_iterator()
    
    # Get the next batch of images and labels.
    images, shapelabels, nshapeslabels, vernierlabels = iterator.get_next()
    
    # reshape images (they were flattened when transformed into bytes)
    images = tf.reshape(images, [parameters.batch_size, parameters.im_size[0], parameters.im_size[1], parameters.im_depth])
    shapelabels = tf.reshape(shapelabels, [parameters.batch_size, 1])
    nshapeslabels = tf.reshape(nshapeslabels, [parameters.batch_size, 1])
    vernierlabels = tf.reshape(vernierlabels, [parameters.batch_size, 1])
    
    
    # The input-function must return a dict wrapping the images.
    if train:
        feed_dict = {'X': images,
                     'y': shapelabels,
                     'nshapeslabels': nshapeslabels,
                     'vernier_offsets': vernierlabels,
                     'mask_with_labels': True,
                     'dropout_keep_prob': parameters.dropout_keep_prob,
                     'is_training': True}
    else:
        feed_dict = {'X': images,
                     'y': shapelabels,
                     'nshapeslabels': nshapeslabels,
                     'vernier_offsets': vernierlabels,
                     'mask_with_labels': False,
                     'dropout_keep_prob': 1,
                     'is_training': False}
    return feed_dict, shapelabels



##############################
#   Final input functions:   #
##############################
def train_input_fn():
    return input_fn(filenames=parameters.train_data_path, train=True, parameters=parameters)


def eval_input_fn():
    eval_file = parameters.test_data_paths[0] + '.tfrecords'
    return input_fn(filenames=eval_file, train=False, parameters=parameters)


def predict_input_fn(filenames):
    return input_fn(filenames=filenames, train=False, parameters=parameters)
