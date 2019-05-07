# -*- coding: utf-8 -*-
"""
My script to create tfrecords files based on batchmaker class
@author: Lynn

This code is inspired by this youtube-vid and code:
https://www.youtube.com/watch?v=oxrcZ9uUblI
https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/18_TFRecords_Dataset_API.ipynb

Last update on 07.05.2019
-> this my_make_tf was adapted for the new projects (temporal dynamics) that uses dicts for the test_configs
"""

import sys
import os
import tensorflow as tf
import numpy as np
from my_parameters import parameters
from my_batchmaker import stim_maker_fn

#import matplotlib.pyplot as plt

##################################
#       Extra parameters:        #
##################################
training = 1
testing = 1
testing_crowding = 1


##################################
#       Helper functions:        #
##################################
def wrap_int64(value):
    output = tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    return output

def wrap_bytes(value):
    output = tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    return output

def print_progress(count, total):
    percent_complete = float(count) / total
    msg = "\r- Progress: {0:.1%}".format(percent_complete)
    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()


##################################
#      tfrecords function:       #
##################################
def make_tfrecords(out_path, stim_maker, state, shape_types, n_shapes, n_samples,
                   train_procedure='random', overlap=True, stim_idx=None, centralize=False, reduce_df=False):
    '''Function to create tfrecord files based on stim_maker class'''
    # Inputs:
    # stim_maker instance
    # state: decide whether to create the training (=training) or testing (=testing) dataset;
    # shape_types: shape type(s) involved in the dataset. For testing, insert only one shape type;
    # n_shapes: one of the listed repetitions gets (randomly) chosen;
    # n_samples
    # noise
    # out_path
    # stim_idx: either create vernier, crowding or uncrowding stimuli for test set
    
    print("\nConverting: " + out_path)
    
    # Open a TFRecordWriter for the output-file.
    with tf.python_io.TFRecordWriter(out_path) as writer:

        # Create images one by one using stimMaker and save them
        for i in range(n_samples):
            print_progress(count=i, total=n_samples - 1)
            
            # Either create training or testing dataset
            if state=='training':
                [shape_1_images, shape_2_images, shapelabels, vernierlabels, nshapeslabels,
                 nshapeslabels_idx, x_shape_1, y_shape_1, x_shape_2, y_shape_2] = stim_maker.makeTrainBatch(
                 shape_types, n_shapes, 1, train_procedure, overlap, centralize, reduce_df)

            elif state=='testing':
                # In this case, the shape_types involve all configs in a dict
                test_configs = shape_types
                if len(test_configs) == 1:
                    # Either use the single test_config that we are giving:
                    config_idx = list(test_configs)[0]
                    chosen_config = test_configs[config_idx]
                else:
                    # Or chose one config randomly from all configs we r giving for the validation set:
                    config_idx = np.random.randint(0, len(test_configs))
                    chosen_config = test_configs[str(config_idx)]

                [shape_1_images, shape_2_images, shapelabels, vernierlabels, nshapeslabels,
                 nshapeslabels_idx, x_shape_1, y_shape_1, x_shape_2, y_shape_2] = stim_maker.makeTestBatch(
                 chosen_config, n_shapes, 1, stim_idx, centralize, reduce_df)

            # Convert the image to raw bytes.
            shape_1_images_bytes = shape_1_images.tostring()
            shape_2_images_bytes = shape_2_images.tostring()
            shapelabels_bytes = shapelabels.tostring()
            nshapeslabels_bytes = nshapeslabels.tostring()
            nshapeslabels_idx_bytes = nshapeslabels_idx.tostring()
            vernierlabels_bytes = vernierlabels.tostring()
            x_shape_1_bytes = x_shape_1.tostring()
            y_shape_1_bytes = y_shape_1.tostring()
            x_shape_2_bytes = x_shape_2.tostring()
            y_shape_2_bytes = y_shape_2.tostring()

            # Create a dict with the data to save in the TFRecords file
            data = {'shape_1_images': wrap_bytes(shape_1_images_bytes),
                    'shape_2_images': wrap_bytes(shape_2_images_bytes),
                    'shapelabels': wrap_bytes(shapelabels_bytes),
                    'nshapeslabels': wrap_bytes(nshapeslabels_bytes),
                    'nshapeslabels_idx': wrap_bytes(nshapeslabels_idx_bytes),
                    'vernierlabels': wrap_bytes(vernierlabels_bytes),
                    'x_shape_1': wrap_bytes(x_shape_1_bytes),
                    'y_shape_1': wrap_bytes(y_shape_1_bytes),
                    'x_shape_2': wrap_bytes(x_shape_2_bytes),
                    'y_shape_2': wrap_bytes(y_shape_2_bytes)}

            # Wrap the data as TensorFlow Features.
            feature = tf.train.Features(feature=data)

            # Wrap again as a TensorFlow Example.
            example = tf.train.Example(features=feature)

            # Serialize the data.
            serialized = example.SerializeToString()

            # Write the serialized data to the TFRecords file.
            writer.write(serialized)
    return


###################################
#     Create tfrecords files:     #
###################################
print('\n-------------------------------------------------------')
print('Creating tfrecords files of type:', parameters.train_procedure)
print('Overlap:', parameters.overlapping_shapes)

stim_maker = stim_maker_fn(parameters.im_size, parameters.shape_size, parameters.bar_width)

if not os.path.exists(parameters.data_path):
    os.mkdir(parameters.data_path)


# Create the training set:
if training:
    mode = 'training'
    shape_types_train = parameters.shape_types
    train_procedure = 'random'
    make_tfrecords(parameters.train_data_path, stim_maker, mode, shape_types_train, parameters.n_shapes,
                   parameters.n_train_samples, train_procedure, parameters.overlapping_shapes,
                   centralize=parameters.centralized_shapes, reduce_df=parameters.reduce_df)
    print('\n-------------------------------------------------------')
    print('Finished creation of training set')
    print('-------------------------------------------------------')


# Create the validation and the test set that uses the stimuli of the training set:
if testing:
    mode = 'training'
    shape_types_train = parameters.shape_types
    train_procedure = 'random'

    # Validation set with all possible training stimuli:
    make_tfrecords(parameters.val_data_path, stim_maker, mode, shape_types_train, parameters.n_shapes, 
                   parameters.n_test_samples, train_procedure, parameters.overlapping_shapes,
                   centralize=parameters.centralized_shapes, reduce_df=parameters.reduce_df)

    # Individual test sets:
    for i in range(len(parameters.test_data_paths)):
        # We use +1 here to skip a vernier-vernier configuration
        chosen_shape = shape_types_train[i+1]
        test_file_path = parameters.test_data_paths[i]
        make_tfrecords(test_file_path, stim_maker, mode, chosen_shape, parameters.n_shapes,
                       parameters.n_test_samples, train_procedure, parameters.overlapping_shapes,
                       centralize=parameters.centralized_shapes, reduce_df=parameters.reduce_df)
    print('\n-------------------------------------------------------')
    print('Finished creation of regular validation and test sets')
    print('-------------------------------------------------------')


# Create the validation and the test set that uses crowding/uncrowding stimuli:
if testing_crowding:
    mode = 'testing'
    test_configs = parameters.test_configs[0]
    
#    # Validation set with all possible stimuli:
    make_tfrecords(parameters.val_crowding_data_path, stim_maker, mode, test_configs, parameters.n_shapes,
                   parameters.n_test_samples, centralize=parameters.centralized_shapes, reduce_df=parameters.reduce_df)

    # Individual test sets:
    for i in range(len(test_configs)):
        test_config = {str(i): test_configs[str(i)]}
        test_data_path = parameters.test_crowding_data_paths[i]
        if not os.path.exists(test_data_path):
            os.mkdir(test_data_path)
        for stim_idx in range(2):
            test_file_path = test_data_path + '/' + str(stim_idx) + '.tfrecords'
            make_tfrecords(test_file_path, stim_maker, mode, test_config, parameters.n_shapes,
                           parameters.n_test_samples, stim_idx=stim_idx, centralize=parameters.centralized_shapes,
                           reduce_df=parameters.reduce_df)
    print('\n-------------------------------------------------------')
    print('Finished creation of crowding validaton and test sets')
    print('-------------------------------------------------------')


