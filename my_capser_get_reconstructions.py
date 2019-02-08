# -*- coding: utf-8 -*-
"""
My capsnet: Get reconstructions
@author: Lynn

Last update on 01.02.2019
-> first draft
"""

import logging
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

from my_parameters import parameters
from my_batchmaker import stim_maker_fn
from my_capser_model_fn import model_fn


print('-------------------------------------------------------')
print('TF version:', tf.__version__)
print('Starting get_reconstruction script...')
print('-------------------------------------------------------')


###########################
#    Helper functions:    #
###########################
def create_batch(chosen_shape, stim_idx, batch_size, parameters):
    im_size = parameters.im_size
    shape_size = parameters.shape_size
    bar_with = parameters.bar_width
    n_shapes = parameters.n_shapes
    centralize = parameters.centralized_shapes
    reduce_df = parameters.reduce_df
    test_noise = parameters.test_noise
    
    stim_maker = stim_maker_fn(im_size, shape_size, bar_with)
    [shape_1_images, shape_2_images, shapelabels, vernierlabels, nshapeslabels, nshapeslabels_idx, 
     x_shape_1, y_shape_1, x_shape_2, y_shape_2] = stim_maker.makeTestBatch(chosen_shape, n_shapes, batch_size, stim_idx, centralize, reduce_df)
    
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


#def predict_input_fn(feed_dict, batch_size):    
#    return tf.estimator.inputs.numpy_input_fn(feed_dict, batch_size=batch_size, num_epochs=1, shuffle=False)

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


def plot_results(originals, results1, results2, verniers, save=None):
    Nr = 4
    Nc = originals.shape[0]
    fig, axes = plt.subplots(Nc, Nr)
    
    images = []
    for i in range(Nc):
        images.append(axes[i, 0].imshow(np.squeeze(originals[i,:,:,:])))
        axes[i, 0].axis('off')
        images.append(axes[i, 1].imshow(np.squeeze(results1[i,:,:,:])))
        axes[i, 1].axis('off')
        images.append(axes[i, 2].imshow(np.squeeze(results2[i,:,:,:])))
        axes[i, 2].axis('off')
        images.append(axes[i, 3].imshow(np.squeeze(verniers[i,:,:,:])))
        axes[i, 3].axis('off')

    if save is None:
        plt.show()
        plt.close()
    else:
        plt.savefig(save)
        plt.close()


###########################
#       Main script:      #
###########################
# For reproducibility:
tf.reset_default_graph()
np.random.seed(42)
tf.set_random_seed(42)

batch_size = 12
n_rounds = parameters.n_rounds
n_iterations = parameters.n_iterations
n_categories = len(parameters.test_shape_types)
n_idx = 3

for idx_execution in range(n_iterations):
    log_dir = parameters.logdir + str(idx_execution) + '/'

    ##################################
    #     Testing / Predictions:     #
    ##################################
    # Lets have less annoying logs:
    logging.getLogger().setLevel(logging.CRITICAL)        
    
    # Testing with crowding/uncrowding:
    for n_category in range(n_categories):
        category_idx = parameters.test_shape_types[n_category]
        category = parameters.test_crowding_data_paths[n_category]
        print('-------------------------------------------------------')
        print('Reconstruct for ' + category)
        
        for stim_idx in range(n_idx):            
            # Lets also get some reconstructions for prediction mode using the following path:
            capser = tf.estimator.Estimator(model_fn=model_fn, model_dir=log_dir, params={'log_dir': log_dir, 'get_reconstructions': True, 'batch_size': batch_size})
            feed_dict = create_batch(category_idx, stim_idx, batch_size, parameters)
            
            capser_out = list(capser.predict(lambda: predict_input_fn(feed_dict)))
            results1 = [p['decoder_output_img1'] for p in capser_out]
            results2 = [p['decoder_output_img2'] for p in capser_out]
            verniers = [p['decoder_vernier_img'] for p in capser_out]
            
            # Plotting and saving:
            img_path = log_dir + '/uncrowding'
            if not os.path.exists(img_path):
                os.mkdir(img_path)
            originals = feed_dict['shape_1_images'] + feed_dict['shape_2_images']
            plot_results(originals, np.asarray(results1), np.asarray(results2), np.asarray(verniers), img_path + '/' + category[21:] + str(stim_idx) + '.png')


print('... Finished get_reconstruction script!')
print('-------------------------------------------------------')
