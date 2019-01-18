# -*- coding: utf-8 -*-
"""
My script for the input fn that is working with tfrecords files
@author: Lynn

Last update on 14.01.2019
-> added requirements for nshapes and location loss
-> added num_repeat to None for training and drop_remainder=True (requires at least tf version 1.10.0)
-> added data augmentation (noise, flipping, contrast, brightness)
-> train and test noise is randomly changed now between a lower and upper border
-> clip the pixel values, so that adding venier and shape images leads to pixel intensities of maximally 1
-> some changes in parameter names
-> new validation and testing procedures
-> use train_procedures 'vernier_shape', 'random_random' or 'random'
-> some name_scope
-> for the random condition, shape_2_image is a pure noise image
-> for the data augmentation, we need the real nshapelabels and not just the idx
-> decide whether to use data augment or not
"""

import tensorflow as tf
from my_parameters import parameters


########################################
#     Parse tfrecords training set:    #
########################################
def parse_tfrecords_train(serialized_data):
    with tf.name_scope('Parsing_trainset'):
        # Define a dict with the data-names and types we expect to find in the TFRecords file.
        features = {'shape_1_images': tf.FixedLenFeature([], tf.string),
                    'shape_2_images': tf.FixedLenFeature([], tf.string),
                    'shapelabels': tf.FixedLenFeature([], tf.string),
                    'nshapeslabels': tf.FixedLenFeature([], tf.string),
                    'nshapeslabels_idx': tf.FixedLenFeature([], tf.string),
                    'vernierlabels': tf.FixedLenFeature([], tf.string),
                    'x_shape_1': tf.FixedLenFeature([], tf.string),
                    'y_shape_1': tf.FixedLenFeature([], tf.string),
                    'x_shape_2': tf.FixedLenFeature([], tf.string),
                    'y_shape_2': tf.FixedLenFeature([], tf.string)}
    
        # Parse the serialized data so we get a dict with our data.
        parsed_data = tf.parse_single_example(serialized=serialized_data, features=features)
    
        # Get the images as raw bytes and decode afterwards.
        shape_1_images = parsed_data['shape_1_images']
        shape_1_images = tf.decode_raw(shape_1_images, tf.float32)
        shape_1_images = tf.cast(shape_1_images, tf.float32)
    
        shape_2_images = parsed_data['shape_2_images']
        shape_2_images = tf.decode_raw(shape_2_images, tf.float32)
        shape_2_images = tf.cast(shape_2_images, tf.float32)
        
        # Get the labels associated with the image and decode.
        shapelabels = parsed_data['shapelabels']
        shapelabels = tf.decode_raw(shapelabels, tf.float32)
        shapelabels = tf.cast(shapelabels, tf.int64)
        
        nshapeslabels = parsed_data['nshapeslabels']
        nshapeslabels = tf.decode_raw(nshapeslabels, tf.float32)
        
        nshapeslabels_idx = parsed_data['nshapeslabels_idx']
        nshapeslabels_idx = tf.decode_raw(nshapeslabels_idx, tf.float32)
        nshapeslabels_idx = tf.cast(nshapeslabels_idx, tf.int64)
        
        vernierlabels = parsed_data['vernierlabels']
        vernierlabels = tf.decode_raw(vernierlabels, tf.float32)
        
        x_shape_1 = parsed_data['x_shape_1']
        x_shape_1 = tf.decode_raw(x_shape_1, tf.float32)
        
        y_shape_1 = parsed_data['y_shape_1']
        y_shape_1 = tf.decode_raw(y_shape_1, tf.float32)
        
        x_shape_2 = parsed_data['x_shape_2']
        x_shape_2 = tf.decode_raw(x_shape_2, tf.float32)
        
        y_shape_2 = parsed_data['y_shape_2']
        y_shape_2 = tf.decode_raw(y_shape_2, tf.float32)

        # Reshaping:
        shape_1_images = tf.reshape(shape_1_images, [parameters.im_size[0], parameters.im_size[1], parameters.im_depth])
        shape_2_images = tf.reshape(shape_2_images, [parameters.im_size[0], parameters.im_size[1], parameters.im_depth])
        shapelabels = tf.reshape(shapelabels, [2])
        nshapeslabels = tf.reshape(nshapeslabels, [2])
        nshapeslabels_idx = tf.reshape(nshapeslabels_idx, [2])
        vernierlabels = tf.reshape(vernierlabels, [1])
        x_shape_1 = tf.reshape(x_shape_1, [1])
        y_shape_1 = tf.reshape(y_shape_1, [1])
        x_shape_2 = tf.reshape(x_shape_2, [1])
        y_shape_2 = tf.reshape(y_shape_2, [1])
        
        if parameters.train_procedure=='random':
            # For the random condition, we want to only add a noise image
            shape_2_images = tf.zeros([parameters.im_size[0], parameters.im_size[1], parameters.im_depth], tf.float32)


    ##################################
    #       Data augmentation:       #
    ##################################
    with tf.name_scope('Data_augmentation_trainset'):
        # Add some random gaussian TRAINING noise (always):
        noise1 = tf.random_uniform([1], parameters.train_noise[0], parameters.train_noise[1], tf.float32)
        noise2 = tf.random_uniform([1], parameters.train_noise[0], parameters.train_noise[1], tf.float32)
        shape_1_images = tf.add(shape_1_images, tf.random_normal(
                shape=[parameters.im_size[0], parameters.im_size[1], parameters.im_depth], mean=0.0, stddev=noise1))
        shape_2_images = tf.add(shape_2_images, tf.random_normal(
                shape=[parameters.im_size[0], parameters.im_size[1], parameters.im_depth], mean=0.0, stddev=noise2))
    
    
        # Adjust brightness and contrast by a random factor
        def bright_contrast():
            shape_1_images_augmented = tf.image.random_brightness(shape_1_images, parameters.delta_brightness)
            shape_2_images_augmented = tf.image.random_brightness(shape_2_images, parameters.delta_brightness)
            shape_1_images_augmented = tf.image.random_contrast(
                    shape_1_images_augmented,parameters.delta_contrast[0], parameters.delta_contrast[1])
            shape_2_images_augmented = tf.image.random_contrast(
                    shape_2_images_augmented, parameters.delta_contrast[0], parameters.delta_contrast[1])
            return shape_1_images_augmented, shape_2_images_augmented
        
        def contrast_bright():
            shape_1_images_augmented = tf.image.random_contrast(
                    shape_1_images, parameters.delta_contrast[0], parameters.delta_contrast[1])
            shape_2_images_augmented = tf.image.random_contrast(
                    shape_2_images, parameters.delta_contrast[0], parameters.delta_contrast[1])
            shape_1_images_augmented = tf.image.random_brightness(shape_1_images_augmented, parameters.delta_brightness)
            shape_2_images_augmented = tf.image.random_brightness(shape_2_images_augmented, parameters.delta_brightness)
            return shape_1_images_augmented, shape_2_images_augmented
    
        # Maybe change contrast and brightness:
        if parameters.allow_contrast_augmentation:
            pred = tf.less(tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32), 0.5)
            shape_1_images, shape_2_images = tf.cond(pred, bright_contrast, contrast_bright)
    
        # Flipping (code is messy since we r using tf cond atm, but this is the idea):
        # - change vernierlabels: ceil(abs( (vernierlabels * vernierlabels/2) - 0.5) )
        # - change shape coordinates / vernier coordinates:
        #       - x: im_size[1] - (x + nshapes*shapesize)
        #       - y: im_size[0] - (y + shapesize)
    
        # no flipping function:
        def flip0():
            shape_1_images_flipped = shape_1_images
            shape_2_images_flipped = shape_2_images
            vernierlabels_flipped = vernierlabels
            x_shape_1_flipped = x_shape_1
            y_shape_1_flipped = y_shape_1
            x_shape_2_flipped = x_shape_2
            y_shape_2_flipped = y_shape_2
            return [shape_1_images_flipped, shape_2_images_flipped, vernierlabels_flipped,
                    x_shape_1_flipped, y_shape_1_flipped, x_shape_2_flipped, y_shape_2_flipped]
    
        # flip left-right function:
        def flip1():
            shape_1_images_flipped = tf.image.flip_left_right(shape_1_images)
            shape_2_images_flipped = tf.image.flip_left_right(shape_2_images)      
            vernierlabels_flipped = tf.ceil(tf.abs(tf.subtract(tf.multiply(vernierlabels, tf.divide(vernierlabels, 2.)), 0.5)))
            if parameters.im_size[1] % 2 == 0:
                x_shape_1_flipped = tf.subtract(tf.constant(parameters.im_size[1], tf.float32),
                                                tf.add(x_shape_1, 
                                                tf.multiply(nshapeslabels[0], tf.constant(parameters.shape_size, tf.float32))))
                x_shape_2_flipped = tf.subtract(tf.constant(parameters.im_size[1], tf.float32), 
                                                tf.add(x_shape_2,
                                                tf.multiply(nshapeslabels[1], tf.constant(parameters.shape_size, tf.float32))))
            else:
                x_shape_1_flipped = tf.subtract(tf.constant(parameters.im_size[1]-1, tf.float32),
                                                tf.add(x_shape_1, 
                                                tf.multiply(nshapeslabels[0], tf.constant(parameters.shape_size, tf.float32))))
                x_shape_2_flipped = tf.subtract(tf.constant(parameters.im_size[1]-1, tf.float32), 
                                                tf.add(x_shape_2,
                                                tf.multiply(nshapeslabels[1], tf.constant(parameters.shape_size, tf.float32))))
            y_shape_1_flipped = y_shape_1
            y_shape_2_flipped = y_shape_2
            return [shape_1_images_flipped, shape_2_images_flipped, vernierlabels_flipped,
                    x_shape_1_flipped, y_shape_1_flipped, x_shape_2_flipped, y_shape_2_flipped]
    
        # flip up-down function:
        def flip2():
            shape_1_images_flipped = tf.image.flip_up_down(shape_1_images)
            shape_2_images_flipped = tf.image.flip_up_down(shape_2_images)
            vernierlabels_flipped = tf.ceil(tf.abs(tf.subtract(tf.multiply(vernierlabels, tf.divide(vernierlabels, 2)), 0.5)))
            x_shape_1_flipped = x_shape_1
            y_shape_1_flipped = tf.subtract(tf.constant(parameters.im_size[0], tf.float32), tf.add(y_shape_1, parameters.shape_size))
            x_shape_2_flipped = x_shape_2
            y_shape_2_flipped = tf.subtract(tf.constant(parameters.im_size[0], tf.float32), tf.add(y_shape_2, parameters.shape_size))
            return [shape_1_images_flipped, shape_2_images_flipped, vernierlabels_flipped,
                    x_shape_1_flipped, y_shape_1_flipped, x_shape_2_flipped, y_shape_2_flipped]
        
        if parameters.allow_flip_augmentation:
            # tf flip functions need 4D inputs:
            shape_1_images = tf.expand_dims(shape_1_images, 0)
            shape_2_images = tf.expand_dims(shape_2_images, 0)
        
            # Maybe flip left-right:
            pred_flip1 = tf.less(tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32), 0.5)
            shape_1_images, shape_2_images, vernierlabels, x_shape_1, y_shape_1, x_shape_2, y_shape_2 = tf.cond(pred_flip1, flip0, flip1)
            
            # Maybe flip up-down:
            pred_flip2 = tf.less(tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32), 0.5)
            shape_1_images, shape_2_images, vernierlabels, x_shape_1, y_shape_1, x_shape_2, y_shape_2 = tf.cond(pred_flip2, flip0, flip2)
            
            # Get rid of extra-dimension:
            shape_1_images = tf.squeeze(shape_1_images, axis=0)
            shape_2_images = tf.squeeze(shape_2_images, axis=0)
        
        # Lets clip the pixel values
        shape_1_images = tf.clip_by_value(shape_1_images, parameters.clip_values[0], parameters.clip_values[1])
        shape_2_images = tf.clip_by_value(shape_2_images, parameters.clip_values[0], parameters.clip_values[1])
        
        # Flipping calculations require these labels as float32, but we need int64
        vernierlabels = tf.cast(vernierlabels, tf.int64)
        nshapeslabels = tf.cast(nshapeslabels, tf.int64)
        x_shape_1 = tf.cast(x_shape_1, tf.int64)
        y_shape_1 = tf.cast(y_shape_1, tf.int64)
        x_shape_2 = tf.cast(x_shape_2, tf.int64)
        y_shape_2 = tf.cast(y_shape_2, tf.int64)

    return [shape_1_images, shape_2_images, shapelabels, nshapeslabels_idx, vernierlabels,
            x_shape_1, y_shape_1, x_shape_2, y_shape_2]


########################################
#      Parse tfrecords test set:       #
########################################
def parse_tfrecords_test(serialized_data):
    with tf.name_scope('Parsing_testset'):
        # Define a dict with the data-names and types we expect to find in the TFRecords file.
        features = {'shape_1_images': tf.FixedLenFeature([], tf.string),
                    'shape_2_images': tf.FixedLenFeature([], tf.string),
                    'shapelabels': tf.FixedLenFeature([], tf.string),
                    'nshapeslabels': tf.FixedLenFeature([], tf.string),
                    'nshapeslabels_idx': tf.FixedLenFeature([], tf.string),
                    'vernierlabels': tf.FixedLenFeature([], tf.string),
                    'x_shape_1': tf.FixedLenFeature([], tf.string),
                    'y_shape_1': tf.FixedLenFeature([], tf.string),
                    'x_shape_2': tf.FixedLenFeature([], tf.string),
                    'y_shape_2': tf.FixedLenFeature([], tf.string)}
    
        # Parse the serialized data so we get a dict with our data.
        parsed_data = tf.parse_single_example(serialized=serialized_data, features=features)
    
        # Get the images as raw bytes and decode afterwards.
        shape_1_images = parsed_data['shape_1_images']
        shape_1_images = tf.decode_raw(shape_1_images, tf.float32)
        shape_1_images = tf.cast(shape_1_images, tf.float32)
    
        shape_2_images = parsed_data['shape_2_images']
        shape_2_images = tf.decode_raw(shape_2_images, tf.float32)
        shape_2_images = tf.cast(shape_2_images, tf.float32)
        
        # Get the labels associated with the image and decode.
        shapelabels = parsed_data['shapelabels']
        shapelabels = tf.decode_raw(shapelabels, tf.float32)
        shapelabels = tf.cast(shapelabels, tf.int64)
        
        nshapeslabels = parsed_data['nshapeslabels']
        nshapeslabels = tf.decode_raw(nshapeslabels, tf.float32)
        nshapeslabels = tf.cast(nshapeslabels, tf.int64)
        
        nshapeslabels_idx = parsed_data['nshapeslabels_idx']
        nshapeslabels_idx = tf.decode_raw(nshapeslabels_idx, tf.float32)
        nshapeslabels_idx = tf.cast(nshapeslabels_idx, tf.int64)
        
        vernierlabels = parsed_data['vernierlabels']
        vernierlabels = tf.decode_raw(vernierlabels, tf.float32)
        vernierlabels = tf.cast(vernierlabels, tf.int64)
        
        x_shape_1 = parsed_data['x_shape_1']
        x_shape_1 = tf.decode_raw(x_shape_1, tf.float32)
        x_shape_1 = tf.cast(x_shape_1, tf.int64)
        
        y_shape_1 = parsed_data['y_shape_1']
        y_shape_1 = tf.decode_raw(y_shape_1, tf.float32)
        y_shape_1 = tf.cast(y_shape_1, tf.int64)
        
        x_shape_2 = parsed_data['x_shape_2']
        x_shape_2 = tf.decode_raw(x_shape_2, tf.float32)
        x_shape_2 = tf.cast(x_shape_2, tf.int64)
        
        y_shape_2 = parsed_data['y_shape_2']
        y_shape_2 = tf.decode_raw(y_shape_2, tf.float32)
        y_shape_2 = tf.cast(y_shape_2, tf.int64)
    
        # Reshaping:
        shape_1_images = tf.reshape(shape_1_images, [parameters.im_size[0], parameters.im_size[1], parameters.im_depth])
        shape_2_images = tf.reshape(shape_2_images, [parameters.im_size[0], parameters.im_size[1], parameters.im_depth])
        shapelabels = tf.reshape(shapelabels, [2])
        nshapeslabels_idx = tf.reshape(nshapeslabels_idx, [1])
        vernierlabels = tf.reshape(vernierlabels, [1])
        x_shape_1 = tf.reshape(x_shape_1, [1])
        y_shape_1 = tf.reshape(y_shape_1, [1])
        x_shape_2 = tf.reshape(x_shape_2, [1])
        y_shape_2 = tf.reshape(y_shape_2, [1])
    
        # For the test and validation set, we dont really need data augmentation,
        # but we'd still like some TEST noise
        noise1 = tf.random_uniform([1], parameters.test_noise[0], parameters.test_noise[1], tf.float32)
        noise2 = tf.random_uniform([1], parameters.test_noise[0], parameters.test_noise[1], tf.float32)
        shape_1_images = tf.add(shape_1_images, tf.random_normal(
                shape=[parameters.im_size[0], parameters.im_size[1], parameters.im_depth], mean=0.0,
                stddev=noise1))
        shape_2_images = tf.add(shape_2_images, tf.random_normal(
                shape=[parameters.im_size[0], parameters.im_size[1], parameters.im_depth], mean=0.0,
                stddev=noise2))
        
        # Lets clip the pixel values, so that if we add them the maximum pixel intensity will be 1:
        shape_1_images = tf.clip_by_value(shape_1_images, parameters.clip_values[0], parameters.clip_values[1])
        shape_2_images = tf.clip_by_value(shape_2_images, parameters.clip_values[0], parameters.clip_values[1])
    
    return [shape_1_images, shape_2_images, shapelabels, nshapeslabels_idx, vernierlabels,
            x_shape_1, y_shape_1, x_shape_2, y_shape_2]


###########################
#     Input function:     #
###########################
def input_fn(filenames, stage, parameters, buffer_size=1024):
    # Create a TensorFlow Dataset-object:
    dataset = tf.data.TFRecordDataset(filenames=filenames, num_parallel_reads=32)
    
    # Currently, I am using two different functions for parsing the train and 
    # test/eval set due to different data augmentation:
    if stage is 'train' or stage is 'eval':
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
    [shape_1_images, shape_2_images, shapelabels, nshapeslabels, vernierlabels, x_shape_1, y_shape_1, 
     x_shape_2, y_shape_2] = iterator.get_next()

    if stage is 'train':
        feed_dict = {'shape_1_images': shape_1_images,
                     'shape_2_images': shape_2_images,
                     'shapelabels': shapelabels,
                     'nshapeslabels': nshapeslabels,
                     'vernier_offsets': vernierlabels,
                     'x_shape_1': x_shape_1,
                     'y_shape_1': y_shape_1,
                     'x_shape_2': x_shape_2,
                     'y_shape_2': y_shape_2,
                     'mask_with_labels': True,
                     'is_training': True}

    else:
        feed_dict = {'shape_1_images': shape_1_images,
                     'shape_2_images': shape_2_images,
                     'shapelabels': shapelabels,
                     'nshapeslabels': nshapeslabels,
                     'vernier_offsets': vernierlabels,
                     'x_shape_1': x_shape_1,
                     'y_shape_1': y_shape_1,
                     'x_shape_2': x_shape_2,
                     'y_shape_2': y_shape_2,
                     'mask_with_labels': False,
                     'is_training': False}
    return feed_dict, shapelabels



##############################
#   Final input functions:   #
##############################
def train_input_fn():
    return input_fn(filenames=parameters.train_data_path, stage='train', parameters=parameters)

def eval_input_fn(filename):
    return input_fn(filenames=filename, stage='eval', parameters=parameters)

def predict_input_fn(filenames):
    return input_fn(filenames=filenames, stage='test', parameters=parameters)
