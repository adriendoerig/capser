# -*- coding: utf-8 -*-
"""
My capsnet: capser input function
Version 1
Created on Wed Oct 10 11:26:04 2018
@author: Lynn
"""

import tensorflow as tf
from my_parameters import params


###########################
#     Parse tfrecords:    #
###########################
def parse_tfrecords(serialized_data):
    features = {'images': tf.FixedLenFeature([], tf.string),
                'labels': tf.FixedLenFeature([], tf.string),
                'nshapes': tf.FixedLenFeature([], tf.string),
                'vernierlabels': tf.FixedLenFeature([], tf.string)}

    # Parse the serialized data so we get a dict with our data.
    parsed_data = tf.parse_single_example(serialized=serialized_data, features=features)

    # Get the image as raw bytes and decode afterwards.
    images_raw = parsed_data['images']
    labels_raw = parsed_data['labels']
    nshapes_raw = parsed_data['nshapes']
    vernierlabels_raw = parsed_data['vernierlabels']
    images = tf.decode_raw(images_raw, tf.float32)
    labels = tf.decode_raw(labels_raw, tf.float32)
    nshapes = tf.decode_raw(nshapes_raw, tf.float32)
    vernierlabels = tf.decode_raw(vernierlabels_raw, tf.float32)
    return images, labels, nshapes, vernierlabels


###########################
#     Input function:     #
###########################
def input_fn(filenames, params, training=True):
    # Create a TensorFlow Dataset-object:
    dataset = tf.data.TFRecordDataset(filenames=filenames)
    dataset = dataset.map(parse_tfrecords)
    
    if training:
        # Read a buffer of the given size andrandomly shuffle it:
        dataset = dataset.shuffle(buffer_size=params.batch_size)
        num_repeat = params.n_epochs
    else:
        # Don't shuffle the data and only go through the it once:
        num_repeat = 1
        
    # Repeat the dataset the given number of times and get a batch of data with the given size.
    dataset = dataset.repeat(num_repeat)
    dataset = dataset.batch(int(params.batch_size/2))
    
    # Use pipelining to speed up things (see https://www.youtube.com/watch?v=SxOsJPaxHME)
#    dataset = dataset.prefetch(2)
    
    # Create an iterator for the dataset and the above modifications.
    iterator = dataset.make_one_shot_iterator()
    
    # Get the next batch of images and labels.
    images, labels, nshapes, vernierlabels = iterator.get_next()
    
    # reshape images (they were flattened when transformed into bytes)
    images = tf.reshape(images, [params.batch_size, params.im_size[0], params.im_size[1], params.im_depth])
    labels = tf.reshape(labels, [params.batch_size, 1])
    
    # The input-function must return a dict wrapping the images.
    if training:
        feed_dict = {'X': images,
                     'y': labels,
                     'nshapes': nshapes,
                     'vernier_offsets': vernierlabels,
                     'mask_with_labels': True,
                     'is_training': True}
    else:
        feed_dict = {'X': images,
                     'y': labels,
                     'nshapes': nshapes,
                     'vernier_offsets': vernierlabels,
                     'mask_with_labels': False,
                     'is_training': False}
    return feed_dict, labels


def train_input_fn():
    return input_fn(params.train_data_path, params)

def test_input_fn():
    return input_fn(params.test_data_path[1], params, training=False)