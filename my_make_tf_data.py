# -*- coding: utf-8 -*-
"""
My script to create tfrecords files based on batchmaker class
@author: Lynn

This code is inspired by this youtube-vid and code:
https://www.youtube.com/watch?v=oxrcZ9uUblI
https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/18_TFRecords_Dataset_API.ipynb

Last update on 15.11.2018
-> added requirements for nshapes and location loss
-> added overlapping_shapes parameter
"""

import sys
import os
import tensorflow as tf
from my_parameters import parameters
from my_batchmaker import stim_maker_fn


##################################
#       Extra parameters:        #
##################################
training = 1
testing = 1


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
def make_tfrecords(stim_maker, state, shape_types, n_shapes, n_samples, noise, overlap, out_path, stim_idx=None):
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
                [vernier_images, shape_images, shapelabels, nshapeslabels, vernierlabels,
                 x_shape, y_shape, x_vernier, y_vernier] = stim_maker.makeTrainBatch(shape_types, n_shapes, 1, noise, overlap)
            elif state=='testing':
                chosen_shape = shape_types
                [vernier_images, shape_images, shapelabels, nshapeslabels, vernierlabels,
                 x_shape, y_shape, x_vernier, y_vernier] = stim_maker.makeTestBatch(chosen_shape, n_shapes, 1, stim_idx, noise)

            # Convert the image to raw bytes.
            vernier_images_bytes = vernier_images.tostring()
            shape_images_bytes = shape_images.tostring()
            shapelabels_bytes = shapelabels.tostring()
            nshapeslabels_bytes = nshapeslabels.tostring()
            vernierlabels_bytes = vernierlabels.tostring()
            x_shape_bytes = x_shape.tostring()
            y_shape_bytes = y_shape.tostring()
            x_vernier_bytes = x_vernier.tostring()
            y_vernier_bytes = y_vernier.tostring()

            # Create a dict with the data to save in the TFRecords file
            data = {'vernier_images': wrap_bytes(vernier_images_bytes),
                    'shape_images': wrap_bytes(shape_images_bytes),
                    'shapelabels': wrap_bytes(shapelabels_bytes),
                    'nshapeslabels': wrap_bytes(nshapeslabels_bytes),
                    'vernierlabels': wrap_bytes(vernierlabels_bytes),
                    'x_shape': wrap_bytes(x_shape_bytes),
                    'y_shape': wrap_bytes(y_shape_bytes),
                    'x_vernier': wrap_bytes(x_vernier_bytes),
                    'y_vernier': wrap_bytes(y_vernier_bytes)}

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
stim_maker = stim_maker_fn(parameters.im_size, parameters.shape_size, parameters.bar_width)

if not os.path.exists(parameters.data_path):
    os.mkdir(parameters.data_path)


if training:
    mode = 'training'
    shape_types_train = parameters.shape_types
    make_tfrecords(stim_maker, mode, shape_types_train, parameters.n_shapes,
                   parameters.n_train_samples, parameters.train_noise, parameters.overlapping_shapes, parameters.train_data_path)
    print('\n-------------------------------------------------------')
    print('Finished creation of training set')
    print('-------------------------------------------------------')


if testing:
    mode = 'testing'
    shape_types_test = parameters.shape_types
    shape_types_test.append(42)
    for i in range(len(parameters.shape_types)-1):
        chosen_shape = parameters.shape_types[i+1]
        test_data_path = parameters.test_data_paths[i]
        eval_file = test_data_path + '.tfrecords'
        make_tfrecords(stim_maker, mode, chosen_shape, parameters.n_shapes,
                       parameters.n_test_samples, parameters.test_noise, parameters.overlapping_shapes, eval_file)
        
        if not os.path.exists(test_data_path):
            os.mkdir(test_data_path)
        for stim_idx in range(3):
            test_file = test_data_path + '/' + str(stim_idx) + '.tfrecords'
            make_tfrecords(stim_maker, mode, chosen_shape, parameters.n_shapes,
                           parameters.n_test_samples, parameters.test_noise, parameters.overlapping_shapes, test_file, stim_idx)
    print('\n-------------------------------------------------------')
    print('Finished creation of test sets')
    print('-------------------------------------------------------')


