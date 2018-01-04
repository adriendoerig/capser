# from https://github.com/ageron/handson-ml/blob/master/extra_capsnets.ipynb
# video: https://www.youtube.com/watch?v=2Kawrd5szHE&feature=youtu.be
from __future__ import division, print_function, unicode_literals
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from create_sprite import images_to_sprite, invert_grayscale
from data_handling_functions import make_stimuli
from capsule_functions import primary_caps_layer, primary_to_fc_caps_layer, \
    create_masked_decoder_input, create_multiple_masked_inputs, decoder_with_mask, \
    each_capsule_decoder_with_mask, create_capsule_overlay, compute_reconstruction_loss

# reproducibility
tf.reset_default_graph()
np.random.seed(42)
tf.set_random_seed(42)

# stimulus type to run
stim_type = 'square'

# create datasets
im_size = (60, 128)
image_batch, image_labels = make_stimuli(stim_type=stim_type, offset='left', n_repeats=1)

# placeholder for input images and labels
X = tf.placeholder(shape=[None, im_size[0], im_size[1], 1], dtype=tf.float32, name="X")
x_image = tf.reshape(X,[-1, im_size[0], im_size[1],1])
tf.summary.image('input', x_image,6)
y = tf.placeholder(shape=[None], dtype=tf.int64, name="y")


########################################################################################################################
# From input to caps1
########################################################################################################################


# primary capsules -- The first layer will be composed of 8 maps of
# (im_size[0]-2*(conv_kernel_size-1)-(kernel_size-1))/2)*((im_size[1]-2*(conv_kernel_size-1)-(kernel_size-1))/2)) capsules each,
# where each capsule will output an 32D activation vector.

conv_kernel_size = 7
kernel_size = 9
caps_conv_stride = 2
caps1_n_maps = 8
# here we need to be careful about the num
caps1_n_caps = int(caps1_n_maps * ((im_size[0]-2*(conv_kernel_size-1)-(kernel_size-1))/2)*((im_size[1]-2*(conv_kernel_size-1)-(kernel_size-1))/2))  # number of primary capsules: 2*kernel_size convs, stride = 2 in caps conv layer
caps1_n_dims = 8

print_conv_shapes = 0
if print_conv_shapes:
    print('caps1_n_maps, feature map size (y,x):')
    print((caps1_n_maps, ((im_size[0]-2*(conv_kernel_size-1)-(kernel_size-1))/2),((im_size[1]-2*(conv_kernel_size-1)-(kernel_size-1))/2)))

conv1_params = {
    "filters": 64,
    "kernel_size": conv_kernel_size,
    "strides": 1,
    "padding": "valid",
    "activation": tf.nn.relu,
}
conv1 = tf.layers.conv2d(X, name="conv1", **conv1_params) # ** means that conv1_params is a dict {param_name:param_value}
tf.summary.histogram('1st_conv_layer', conv1)
conv1b = tf.layers.conv2d(conv1, name="conv1b", **conv1_params) # ** means that conv1_params is a dict {param_name:param_value}
tf.summary.histogram('1st_b_conv_layer', conv1b)

# create furst capsule layer
caps1_output = primary_caps_layer(conv1b, caps1_n_maps, caps1_n_caps, caps1_n_dims,
                     kernel_size, caps_conv_stride, conv_padding='valid', conv_activation=tf.nn.relu, print_shapes=False)


########################################################################################################################
# From caps1 to caps2
########################################################################################################################


caps2_n_caps = 8 # number of capsules
caps2_n_dims = 16 # of n dimensions

# it is all taken care of by the function
caps2_output = primary_to_fc_caps_layer(X, caps1_output, caps1_n_caps, caps1_n_dims, caps2_n_caps, caps2_n_dims, rba_rounds=3, print_shapes=False)


########################################################################################################################
# Decoder
########################################################################################################################

with tf.name_scope('decoder'):

    # create the mask. first, we create a placeholder that will tell the program whether to use the true
    # or the predicted labels
    mask_with_labels = tf.placeholder_with_default(False, shape=(),
                                                   name="mask_with_labels")
    # create the mask
    decoder_input = create_masked_decoder_input(y, y, caps2_output, caps2_n_caps, caps2_n_dims,
                                                mask_with_labels, print_shapes=False)

    # decoder layer sizes
    n_hidden1 = 512
    n_hidden2 = 1024
    n_output = im_size[0] * im_size[1]

    # run decoder
    decoder_output = decoder_with_mask(decoder_input, n_hidden1, n_hidden2, n_output)


########################################################################################################################
# Start the session, restore model to get caps2_output and decoder weights
########################################################################################################################


with tf.Session() as sess:
    # First restore the network
    model = 'capser_1e_i'
    model_files = './'+model+'_logdir'
    checkpoint_path = model_files+"/"+model+"_model.ckpt"
    output_image_dir = './output_images/' + model + '/'
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_path)
    # get caps2_output
    caps2_output = sess.run([caps2_output],
                            feed_dict={X: image_batch,
                                       y: image_labels,
                                       mask_with_labels: True})

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
            this_decoder_output = decoder_output.eval({X: image_batch, y: np.ones(image_labels.shape)*cap, mask_with_labels: True})
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
        for cap in range(n_caps_to_visualize):
            plt.subplot(np.ceil(np.sqrt(n_caps_to_visualize)), np.ceil(np.sqrt(n_caps_to_visualize)), cap + 1)
            plt.imshow(decoder_images_all[im, :, :, :, cap])
            plt.axis("off")
        plt.savefig(output_image_dir + 'all caps images ' + stim_type + str(im))
        plt.show()
