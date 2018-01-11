# from https://github.com/ageron/handson-ml/blob/master/extra_capsnets.ipynb
# video: https://www.youtube.com/watch?v=2Kawrd5szHE&feature=youtu.be
from __future__ import division, print_function, unicode_literals
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from data_handling_functions import make_stimuli
from capser_general import capser_general_2_caps_layers
import os

# reproducibility
tf.reset_default_graph()
np.random.seed(42)
tf.set_random_seed(42)

# model name, directory etc.
MODEL_NAME = 'capser_3'
HOME = './'
LOGDIR = HOME + MODEL_NAME + '_logdir'
checkpoint_path = LOGDIR + '/' + MODEL_NAME + "_model.ckpt"
output_image_dir = HOME + '/output_images/' + MODEL_NAME + '/'
assert os.path.exists(LOGDIR), 'LOGDIR ' + LOGDIR + ' does not exist'
if not os.path.exists(output_image_dir):
    os.makedirs(output_image_dir)

# stimulus type to run
stim_type = 'square'

# create datasets
im_size = (30, 64)
image_batch, image_labels = make_stimuli(stim_type=stim_type, offset='left', n_repeats=1, image_size=im_size, resize_factor=0.25)

# placeholder for input images and labels
X = tf.placeholder(shape=[None, im_size[0], im_size[1], 1], dtype=tf.float32, name="X")
x_image = tf.reshape(X,[-1, im_size[0], im_size[1],1])
tf.summary.image('input', x_image,6)
y = tf.placeholder(shape=[None], dtype=tf.int64, name="y")
mask_with_labels = tf.placeholder_with_default(False, shape=(), name="mask_with_labels")


########################################################################################################################
# Parameters
########################################################################################################################


# early conv layers
conv1_params = {"filters": 64,
                    "kernel_size": 7,
                    "strides": 1,
                    "padding": "valid",
                    "activation": tf.nn.relu,
                }
conv2_params = {"filters": 64,
                        "kernel_size": 7,
                        "strides": 1,
                        "padding": "valid",
                        "activation": tf.nn.relu,
                }
conv3_params = None

# primary capsules
caps1_n_maps = 8  # number of capsules at level 1 of capsules
caps1_n_dims = 8  # number of dimension per capsule
conv_caps_params = {"filters": caps1_n_maps * caps1_n_dims,
                        "kernel_size": 9,
                        "strides": 2,
                        "padding": "valid",
                        "activation": tf.nn.relu,
                   }

# output capsules
caps2_n_caps = 8  # number of capsules
caps2_n_dims = 10 # of n dimensions ### TRY 50????

# decoder layer sizes
n_hidden1 = 512
n_hidden2 = 1024
n_hidden3 = None
n_output = im_size[0] * im_size[1]


########################################################################################################################
# Create Networks
########################################################################################################################


capser = capser_general_2_caps_layers(X, y, im_size, conv1_params, conv2_params, conv3_params,
                                      caps1_n_maps, caps1_n_dims, conv_caps_params,
                                      caps2_n_caps, caps2_n_dims,
                                      n_hidden1, n_hidden2, n_hidden3, n_output,
                                      mask_with_labels)


########################################################################################################################
# Start the session, restore model to get decoder weights
########################################################################################################################


with tf.Session() as sess:
    # First restore the network
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_path)

    n_images = image_batch.shape[0]
    caps_to_visualize = range(caps2_n_caps)
    n_caps_to_visualize = len(caps_to_visualize)
    color_masks = np.array([[220, 76, 70],    # 0: squares, red
                            [196, 143, 101],  # 1: circles, beige
                            [79, 132, 196],   # 2: hexagons, blue
                            [246, 209, 85],   # 3: octagons, yellow
                            [237, 205, 194],  # 4: stars, pale pink
                            [181, 101, 167],  # 5: lines, purple
                            [121, 199, 83],   # 6: vernier, green
                            [210, 105, 30]])  # 7: bonus capsule, orange

    decoder_outputs_all = np.zeros(shape=(n_images, im_size[0]*im_size[1], 3, n_caps_to_visualize))
    decoder_outputs_overlay = np.zeros(shape=(n_images, im_size[0]*im_size[1], 3))

    done_caps = 0
    for cap in caps_to_visualize:
        for rgb in range(3):
            this_decoder_output = capser["decoder_output"].eval({X: image_batch, y: np.ones(image_labels.shape)*cap, mask_with_labels: True})
            peak_intensity = 255
            this_decoder_output = np.divide(this_decoder_output, peak_intensity)
            temp = np.multiply(this_decoder_output, color_masks[cap, rgb])
            decoder_outputs_all [:, :, rgb, done_caps] = temp
            decoder_outputs_overlay[:, :, rgb] += temp
            if False:
                print(this_decoder_output.shape)
                check_image = np.reshape(temp[0, :], [im_size[0], im_size[1]])
                plt.imshow(check_image)
                plt.show()
        done_caps += 1

    decoder_outputs_overlay[decoder_outputs_overlay>255] = 255

    decoder_images_all = np.reshape(decoder_outputs_all, [n_images, im_size[0], im_size[1], 3, n_caps_to_visualize])
    overlay_images = np.reshape(decoder_outputs_overlay, [n_images, im_size[0], im_size[1], 3])

    plt.figure(figsize=(n_images / .5, n_images / .5))
    for im in range(n_images):
        plt.subplot(np.ceil(np.sqrt(n_images)), np.ceil(np.sqrt(n_images)), im+1)
        plt.imshow(overlay_images[im,:,:,:])
        plt.axis("off")
    plt.savefig(output_image_dir + 'capsule overlay image ' + stim_type)
    plt.show()


    for im in range(n_images):
        plt.figure(figsize=(n_caps_to_visualize / .5, n_caps_to_visualize / .5))
        plt.subplot(np.ceil(np.sqrt(n_caps_to_visualize)) + 1, np.ceil(np.sqrt(n_caps_to_visualize)), 1)
        plt.imshow(image_batch[im, :, :, 0], cmap='gray')
        plt.axis("off")
        for cap in range(n_caps_to_visualize):
            plt.subplot(np.ceil(np.sqrt(n_caps_to_visualize))+1, np.ceil(np.sqrt(n_caps_to_visualize)), cap + 2)
            plt.imshow(decoder_images_all[im, :, :, :, cap])
            plt.axis("off")
        net_prediction = int(capser["y_pred"].eval({X: np.expand_dims(image_batch[im, :, :, :],0), y: np.expand_dims(image_labels[im],1), mask_with_labels: False}))
        num_2_shape = {0:'squares', 1:'circles', 2:'hexagons', 3:'octagons', 4:'stars', 5:'lines', 6:'vernier', 7:'bonus capsule'}
        plt.suptitle('Network prediction: '+num_2_shape[net_prediction])
        plt.savefig(output_image_dir + 'all caps images ' + stim_type + str(im))
        plt.show()
