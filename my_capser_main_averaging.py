# -*- coding: utf-8 -*-
"""
My capsnet: Main script over several networks plus reconstructions
Execute the training, evaluation and prediction of the capsnet
@author: Lynn

Last update on 21.05.2019
-> adaptation new project
-> including final stds now
-> additional reconstructions during testing via get_reconstructions = 1
"""

import re
import logging
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from tensorboard.plugins.beholder import Beholder
from tensorboard.plugins.beholder import BeholderHook

from my_parameters import parameters
from my_batchmaker import stim_maker_fn
from my_capser_model_fn import model_fn
from my_capser_input_fn import train_input_fn, eval_input_fn, predict_input_fn
from my_capser_functions import save_params, plot_uncrowding_results

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
get_reconstructions = 0
reconstruction_batch_size = 12

# For reproducibility:
tf.reset_default_graph()
np.random.seed(42)
tf.set_random_seed(42)

n_idx = 3
n_iterations = parameters.n_iterations
n_categories = len(parameters.test_crowding_data_paths)
routing_min = parameters.routing_min
routing_max = parameters.routing_max
results = np.zeros(shape=(n_categories, n_idx, n_iterations))


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


#def predict_input_fn(feed_dict, batch_size):    
#    return tf.estimator.inputs.numpy_input_fn(feed_dict, batch_size=batch_size, num_epochs=1, shuffle=False)

def predict_input_fn_2(feed_dict):
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
for idx_execution in range(n_iterations):
    log_dir = parameters.logdir + str(idx_execution) + '/'
    
    ##################################
    #    Training and evaluation:    #
    ##################################
    # Output the loss in the terminal every few steps:
    logging.getLogger().setLevel(logging.INFO)
    
    # Beholder to check on weights during training in tensorboard:
    beholder = Beholder(log_dir)
    beholder_hook = BeholderHook(log_dir)
    
    # Create the estimator:
    my_checkpointing_config = tf.estimator.RunConfig(keep_checkpoint_max = 2)  # Retain the 2 most recent checkpoints.
    
    capser = tf.estimator.Estimator(model_fn=model_fn, model_dir=log_dir, config=my_checkpointing_config,
                                    params={'log_dir': log_dir,
                                            'iter_routing': parameters.train_iter_routing})
    eval_spec = tf.estimator.EvalSpec(lambda: eval_input_fn(parameters.val_data_path), steps=parameters.eval_steps, throttle_secs=parameters.eval_throttle_secs)

    # Save parameters from parameter file for reproducability
    save_params(log_dir, parameters)

    # Lets go!
    n_rounds = parameters.n_rounds
    
    
    for idx_round in range(1, n_rounds+1):
        train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=parameters.n_steps*idx_round)
        tf.estimator.train_and_evaluate(capser, train_spec, eval_spec)
    
        ##################################
        #     Testing / Predictions:     #
        ##################################
        # Lets have less annoying logs:
        logging.getLogger().setLevel(logging.CRITICAL)

        for idx_routing in range(routing_min, routing_max+1):
            log_dir_results = log_dir + 'iter_routing_' + str(idx_routing) + '/'
            if not os.path.exists(log_dir_results):
                os.mkdir(log_dir_results)

            # Testing with crowding/uncrowding:
            cats = []
            res = []
            for n_category in range(n_categories):
                category = parameters.test_crowding_data_paths[n_category]
                cats.append(category[21:])
                
                # For the reconstructions:
                test_configs = parameters.test_configs[0]
                category_idx = {str(n_category): test_configs[str(n_category)]}

                print('-------------------------------------------------------')
                print('Compute vernier offset for ' + category)

                # Determine vernier_accuracy for our vernier/crowding/uncrowding stimuli
                results0 = np.zeros(shape=(n_idx,))
                for stim_idx in range(n_idx):
                    priming_input = np.zeros([parameters.batch_size, 1, parameters.caps2_ncaps, parameters.caps2_ndims, 1],
                                             dtype=np.float32)
                    test_filename = category + '/' + str(stim_idx) + '.tfrecords'

                    ###################################
                    #          Performance            #
                    ###################################
                    # Lets get all the results we need:
                    capser = tf.estimator.Estimator(model_fn=model_fn, model_dir=log_dir,
                                                    params={'log_dir': log_dir,
                                                            'get_reconstructions': False,
                                                            'iter_routing': idx_routing,
                                                            'priming_input': priming_input})
                    capser_out = list(capser.predict(lambda: predict_input_fn(test_filename)))
                    vernier_accuracy = [p['vernier_accuracy'] for p in capser_out]
                    rank_pred_shapes = [p['rank_pred_shapes'] for p in capser_out]
                    rank_pred_proba = [p['rank_pred_proba'] for p in capser_out]

                    # Get the the results for averaging over several trained networks only from the final round:
                    if idx_round==n_rounds:
                        results[n_category, stim_idx, idx_execution] = np.mean(vernier_accuracy)

                    # Get all the other results per round:
                    results0[stim_idx] = np.mean(vernier_accuracy)
                    results1 = np.unique(rank_pred_shapes)
                    results2 = np.mean(rank_pred_proba, 0)
                    res.append(np.mean(vernier_accuracy))

                    print('Finished calculations for stimulus type ' + str(stim_idx))
                    print('Result: ' + str(results0[stim_idx]) + '; test_samples used: ' + str(len(vernier_accuracy)))
                    
                    
                    ###################################
                    #         Reconstructions         #
                    ###################################
                    if get_reconstructions:
                        capser = tf.estimator.Estimator(model_fn=model_fn, model_dir=log_dir,
                                                        params={'log_dir': log_dir,
                                                                'get_reconstructions': True,
                                                                'batch_size': reconstruction_batch_size,
                                                                'iter_routing': idx_routing})
                        feed_dict = create_batch(category_idx, stim_idx, reconstruction_batch_size, parameters)
                        
                        capser_out = list(capser.predict(lambda: predict_input_fn_2(feed_dict)))
                        rec_results1 = [p['decoder_output_img1'] for p in capser_out]
                        rec_results2 = [p['decoder_output_img2'] for p in capser_out]
                        rec_verniers = [p['decoder_vernier_img'] for p in capser_out]
                        
                        # Plotting and saving:
                        img_path = log_dir_results + '/uncrowding'
                        if not os.path.exists(img_path):
                            os.mkdir(img_path)
                        originals = feed_dict['shape_1_images'] + feed_dict['shape_2_images']
                        plot_results(originals, np.asarray(rec_results1), np.asarray(rec_results2), np.asarray(rec_verniers),
                                     img_path + '/' + category[21:] + str(stim_idx) + '.png')

                    

                    # Saving ranking results:
                    txt_ranking_file_name = log_dir_results + '/ranking_step_' + str(parameters.n_steps*idx_round) + '.txt'
                    if not os.path.exists(txt_ranking_file_name):
                        with open(txt_ranking_file_name, 'w') as f:
                            f.write(category + str(stim_idx) + ' : \t' + str(results1) + ' : \t' + str(results2) + '\n')
                    else:
                        with open(txt_ranking_file_name, 'a') as f:
                            f.write(category + str(stim_idx) + ' : \t' + str(results1) + ' : \t' + str(results2) + '\n')

                # Saving performance results:
                txt_file_name = log_dir_results + '/uncrowding_results_step_' + str(parameters.n_steps*idx_round) + \
                '_noise_' + str(parameters.test_noise[0]) + '_' + str(parameters.test_noise[1]) + '.txt'
                if not os.path.exists(txt_file_name):
                    with open(txt_file_name, 'w') as f:
                        f.write(category + ' : \t' + str(results0) + '\n')
                else:
                    with open(txt_file_name, 'a') as f:
                        f.write(category + ' : \t' + str(results0) + '\n')

            # Plotting:
            plot_uncrowding_results(res, cats, n_idx, save=log_dir_results + '/uncrowding_results_step_' + str(parameters.n_steps*idx_round) +
                                                           '_noise_' + str(parameters.test_noise[0]) + '_' + str(parameters.test_noise[1]) + '.png')




# Getting final performance means:
for idx_routing in range(routing_min, routing_max+1):
    results = np.zeros(shape=(n_categories, n_idx, n_iterations))

    for idx_execution in range(n_iterations):
        log_dir = parameters.logdir + str(idx_execution) + '/'
        log_dir_results = log_dir + 'iter_routing_' + str(idx_routing) + '/'

        for n_category in range(n_categories):
            # Getting data:
            txt_file_name = log_dir_results + '/uncrowding_results_step_' + str(parameters.n_steps*n_rounds) + \
            '_noise_' + str(parameters.test_noise[0]) + '_' + str(parameters.test_noise[1]) + '.txt'

            with open(txt_file_name, 'r') as f:
                lines = f.read()
                numbers = re.findall(r"[-+]?\d*\.\d+|\d+", lines)
                numbers = np.float32(numbers)
                # Since the categories 4stars and 6stars involve numbers, we have to get rid of them
                numbers = numbers[numbers!=4]
                numbers = numbers[numbers!=6]
                results[:, :, idx_execution] = np.reshape(numbers, [-1, n_idx])


    # Saving final means:
    final_result_file = parameters.logdir + '/final_results_mean_iterations_' + str(n_iterations) + '_iter_routing_' + str(idx_routing) + '.txt'
    final_results = np.mean(results, 2)
    for n_category in range(n_categories):
        category = parameters.test_crowding_data_paths[n_category]
        if not os.path.exists(final_result_file):
            with open(final_result_file, 'w') as f:
                f.write(category + ' : \t' + str(final_results[n_category, :]) + '\n')
        else:
            with open(final_result_file, 'a') as f:
                f.write(category + ' : \t' + str(final_results[n_category, :]) + '\n')


# Getting final performance stds:
for idx_routing in range(routing_min, routing_max+1):
    results = np.zeros(shape=(n_categories, n_idx, n_iterations))

    for idx_execution in range(n_iterations):
        log_dir = parameters.logdir + str(idx_execution) + '/'
        log_dir_results = log_dir + 'iter_routing_' + str(idx_routing) + '/'

        for n_category in range(n_categories):
            # Getting data:
            txt_file_name = log_dir_results + '/uncrowding_results_step_' + str(parameters.n_steps*n_rounds) + \
            '_noise_' + str(parameters.test_noise[0]) + '_' + str(parameters.test_noise[1]) + '.txt'

            with open(txt_file_name, 'r') as f:
                lines = f.read()
                numbers = re.findall(r"[-+]?\d*\.\d+|\d+", lines)
                numbers = np.float32(numbers)
                # Since the categories 4stars and 6stars involve numbers, we have to get rid of them
                numbers = numbers[numbers!=4]
                numbers = numbers[numbers!=6]
                results[:, :, idx_execution] = np.reshape(numbers, [-1, n_idx])


    # Saving final means:
    final_result_file = parameters.logdir + '/final_results_std_iterations_' + str(n_iterations) + '_iter_routing_' + str(idx_routing) + '.txt'
    final_results = np.std(results, 2)
    for n_category in range(n_categories):
        category = parameters.test_crowding_data_paths[n_category]
        if not os.path.exists(final_result_file):
            with open(final_result_file, 'w') as f:
                f.write(category + ' : \t' + str(final_results[n_category, :]) + '\n')
        else:
            with open(final_result_file, 'a') as f:
                f.write(category + ' : \t' + str(final_results[n_category, :]) + '\n')

print('... Finished capsnet script!')
print('-------------------------------------------------------')


# Get reconstructions:
#os.system('python my_capser_get_reconstructions.py')