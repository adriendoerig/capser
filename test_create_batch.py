# -*- coding: utf-8 -*-
"""
My capsnet: Main script over several networks plus reconstructions
Execute the training, evaluation and prediction of the capsnet
@author: Lynn

Last update on 05.06.2019
-> test whether I am creating the right inputs for project 3
"""

import numpy as np
import tensorflow as tf
import copy
import matplotlib.pyplot as plt

from my_parameters import parameters
from my_batchmaker import stim_maker_fn

###################################################

# Disable annoying logs:
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
#tf.logging.set_verbosity(tf.logging.ERROR)

print('-------------------------------------------------------')
print('TF version:', tf.__version__)
print('Starting capsnet script...')
print('Chosen training procedure:', parameters.train_procedure)
print('-------------------------------------------------------')


###########################
#      Preparations:      #
###########################
#plt.close('all')
get_reconstructions = 0
batch_size = 1

# For reproducibility:
tf.reset_default_graph()
np.random.seed(42)
tf.set_random_seed(42)

n_iterations = parameters.n_iterations
n_categories = len(parameters.test_crowding_data_paths)
n_rounds = parameters.n_rounds
n_idx = 2
routing_min = parameters.routing_min
routing_max = parameters.routing_max


###########################
#    Helper functions:    #
###########################
def create_batch(test_config, stim_idx, batch_size, parameters):
    im_size = parameters.im_size
    shape_size = parameters.shape_size
    bar_with = parameters.bar_width
    n_shapes = parameters.n_shapes
    centralize = parameters.centralized_shapes
    reduce_df = parameters.reduce_df
    test_noise = parameters.test_noise
    
    config_idx = list(test_config)[0]
    chosen_config = test_config[config_idx]
    
    stim_maker = stim_maker_fn(im_size, shape_size, bar_with)
    [shape_1_images, shape_2_images, shapelabels, vernierlabels, nshapeslabels, nshapeslabels_idx, 
     x_shape_1, y_shape_1, x_shape_2, y_shape_2] = stim_maker.makeTestBatch(
     chosen_config, n_shapes, batch_size, stim_idx, centralize, reduce_df)
    
    # For the test and validation set, we dont really need data augmentation,
    # but we'd still like some TEST noise
    noise1 = np.random.uniform(test_noise[0], test_noise[1], [1])
    noise2 = np.random.uniform(test_noise[0], test_noise[1], [1])
    shape_1_images = shape_1_images + np.random.normal(0.0, noise1, [batch_size, im_size[0], im_size[1], parameters.im_depth])
    shape_2_images = shape_2_images + np.random.normal(0.0, noise2, [batch_size, im_size[0], im_size[1], parameters.im_depth])
    
    # Lets clip the pixel values, so that if we add them the maximum pixel intensity will be 1:
    shape_1_images = np.clip(shape_1_images, parameters.clip_values[0], parameters.clip_values[1])
    shape_2_images = np.clip(shape_2_images, parameters.clip_values[0], parameters.clip_values[1])
    
    feed_dict = {'shape_1_images': shape_1_images,
                 'shape_2_images': shape_2_images,
                 'shapelabels': shapelabels,
                 'nshapeslabels': nshapeslabels_idx,
                 'vernier_offsets': vernierlabels,
                 'x_shape_1': x_shape_1,
                 'y_shape_1': y_shape_1,
                 'x_shape_2': x_shape_2,
                 'y_shape_2': y_shape_2}
    return feed_dict


def predict_input_fn(feed_dict):
    batch_size = feed_dict['shapelabels'].shape[0]
    
    shape_1_images = feed_dict['shape_1_images']
    shape_2_images = feed_dict['shape_2_images']
    shapelabels = feed_dict['shapelabels']
    nshapeslabels = feed_dict['nshapeslabels']
    vernier_offsets = feed_dict['vernier_offsets']
    x_shape_1 = feed_dict['x_shape_1']
    y_shape_1 = feed_dict['y_shape_1']
    x_shape_2 = feed_dict['x_shape_2']
    y_shape_2 = feed_dict['y_shape_2']
    
    dataset_test = tf.data.Dataset.from_tensor_slices((shape_1_images,
                                                       shape_2_images,
                                                       shapelabels,
                                                       nshapeslabels,
                                                       vernier_offsets,
                                                       x_shape_1,
                                                       y_shape_1,
                                                       x_shape_2,
                                                       y_shape_2))
    dataset_test = dataset_test.batch(batch_size, drop_remainder=True)
    
    # Use pipelining to speed up things (see https://www.youtube.com/watch?v=SxOsJPaxHME)
    dataset_test = dataset_test.prefetch(2)
    
    # Create an iterator for the dataset_test and the above modifications.
    iterator = dataset_test.make_one_shot_iterator()
    
    # Get the next batch of images and labels.
    [shape_1_images, shape_2_images, shapelabels, vernierlabels, nshapeslabels,
     x_shape_1, y_shape_1, x_shape_2, y_shape_2] = iterator.get_next()
    
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

    return feed_dict


for n_category in range(n_categories):
    category = parameters.test_crowding_data_paths[n_category]

    # For the create_batch function:
    test_configs = parameters.test_configs[0]
    category_idx = {str(n_category): test_configs[str(n_category)]}

    for stim_idx in range(n_idx):
        feed_dict_1 = create_batch(category_idx, stim_idx, batch_size, parameters)
        feed_dict_2 = copy.deepcopy(feed_dict_1)
        
        # for the no priming case, we simply override the vernier stimulus
        shape_1_images = np.zeros(shape=[batch_size, parameters.im_size[0],
                                         parameters.im_size[1], parameters.im_depth], dtype=np.float32)
        noise1 = np.random.uniform(parameters.test_noise[0], parameters.test_noise[1], [1])
        shape_1_images = shape_1_images + np.random.normal(0.0, noise1,
                                                           [batch_size, parameters.im_size[0],
                                                            parameters.im_size[1], parameters.im_depth])
        feed_dict_1['shape_1_images'] = shape_1_images
        
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(np.squeeze(feed_dict_1['shape_1_images'][0,:,:,0]) + np.squeeze(feed_dict_1['shape_2_images'][0,:,:,0]))
        ax1.axis('off')
        ax1.set_title('1')
        ax2.imshow(np.squeeze(feed_dict_2['shape_1_images'][0,:,:,0]) + np.squeeze(feed_dict_2['shape_2_images'][0,:,:,0]))
        ax2.axis('off')
        ax2.set_title('2')
        plt.pause(2)


print('... Finished capsnet script!')
print('-------------------------------------------------------')
