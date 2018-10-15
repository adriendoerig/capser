# -*- coding: utf-8 -*-
"""
My capsnet: make_tf_data to create tfrecords files
Version 1
Created on Tue Oct  9 16:40:02 2018
@author: Lynn

This code is inspired by this youtube-vid and code:
https://www.youtube.com/watch?v=oxrcZ9uUblI
https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/18_TFRecords_Dataset_API.ipynb
"""

import sys
import os
import tensorflow as tf
from my_parameters import params
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
def make_tfrecords(stim_maker, mode, shape_types, n_shapes, n_samples, noise, out_path):
    '''Function to create tfrecord files
    
    Inputs:
        stim_maker:
            instance of the StimMaker class (see my_batchmaker.py for details);
        mode:
            decide whether to create the training (=training) or testing (=testing) dataset;
        shape_types:
            shape type(s) involved in the dataset. For testing, insert only one shape type;
        n_shapes:
            one of the listed repetitions gets randomly chosen;
        n_samples:
            number of samples in the dataset;
        noise:
            Random gaussian noise between [0,noise] gets added;
        out_path:
            path and name of the output tfrecords file (e.g. ./example.tfrecords)'''
    print("\nConverting: " + out_path)
    
    # Open a TFRecordWriter for the output-file.
    with tf.python_io.TFRecordWriter(out_path) as writer:

        # Create images one by one using stimMaker and save them
        for i in range(n_samples):
            print_progress(count=i, total=n_samples - 1)
            
            # Either create training or testing dataset
            if mode=='training':
                image, label, nshapes, vernierlabels = stim_maker.makeTrainBatch(shape_types, n_shapes, 1, noise, overlap=None)
            elif mode=='testing':
                chosen_shape = shape_types
                image, label, nshapes, vernierlabels = stim_maker.makeTestBatch(chosen_shape, n_shapes, 1, noise)

            # Convert the image and label to raw bytes.
            image_bytes = image.tostring()
            label_bytes = label.tostring()
            nshapes_bytes = nshapes.tostring()
            vernierlabels_bytes = vernierlabels.tostring()

            # Create a dict with the data to save in the TFRecords file
            data = {'images': wrap_bytes(image_bytes),
                    'labels': wrap_bytes(label_bytes),
                    'nshapes': wrap_bytes(nshapes_bytes),
                    'vernierlabels': wrap_bytes(vernierlabels_bytes)}

            # Wrap the data as TensorFlow Features.
            feature = tf.train.Features(feature=data)

            # Wrap again as a TensorFlow Example.
            example = tf.train.Example(features=feature)

            # Serialize the data.
            serialized = example.SerializeToString()

            # Write the serialized data to the TFRecords file.
            writer.write(serialized)


###################################
#     Create tfrecords files:     #
###################################
stim_maker = stim_maker_fn(params.im_size, params.shape_size, params.bar_width)

if not os.path.exists(params.data_path):
    os.mkdir(params.data_path)


if training:
    mode = 'training'
    shape_types_train = params.shape_types
    make_tfrecords(stim_maker, mode, shape_types_train, params.n_shapes, params.n_train_samples, params.noise, params.train_data_path)
    print('\n-------------------------------------------------------')
    print('Finished creation of training set')
    print('-------------------------------------------------------')


if testing:
    mode = 'testing'
    shape_types_test = params.shape_types
    shape_types_test.append(42)
    for i in range(len(params.shape_types)-1):
        chosen_shape = params.shape_types[i+1]
        test_data_path = params.test_data_paths[i]
        make_tfrecords(stim_maker, mode, chosen_shape, params.n_shapes, params.n_test_samples, params.noise, test_data_path)
    print('\n-------------------------------------------------------')
    print('Finished creation of test sets')
    print('-------------------------------------------------------')
