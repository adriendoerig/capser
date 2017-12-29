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

# create datasets
im_size = (60, 128)
image_batch, image_labels = make_stimuli(stim_type='square', offset='left')

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
caps2_n_dims = 8 # of n dimensions

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
    decoder_input = create_masked_decoder_input(y, y_pred, caps2_output, caps2_n_caps, caps2_n_dims,
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
    model = 'capser_1e'
    model_files = './'+model+' files/'
    saver = tf.train.import_meta_graph(model_files+'model_'+model+'.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./'+model+' files'))

    # get caps2_output
    caps2_output = sess.run([caps2_output],
                            feed_dict={X: image_batch,
                                       y: image_labels,
                                       mask_with_labels: True})













model = 'capser_1d'

image_batch, image_labels = make_stimuli(stim_type='square', offset='left')

with tf.Session() as sess:

    # First let's load meta graph and restore weights
    model_files = './'+model+' files/'
    saver = tf.train.import_meta_graph(model_files+'model_'+model+'.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./'+model+' files'))
    graph = tf.get_default_graph()
    X = graph.get_tensor_by_name('X:0')
    y = graph.get_tensor_by_name('y:0')
    mask_with_labels = graph.get_tensor_by_name('mask_with_labels:0')
    [print(n.name) for n in tf.get_default_graph().as_graph_def().node]
    # caps2_output = graph.get_tensor_by_name('primary_to_first_fc/rba_output:0')
    caps2_output = sess.run([caps2_output],
                          feed_dict={X: image_batch,
                                     y: image_labels,
                                     mask_with_labels: True})
    im_size = (60,128)
    caps2_n_caps = 8
    caps2_n_dims = 10

    print(caps2_output)

    with tf.name_scope('Visualize_colored_capsule_outputs'):
        caps_to_visualize = range(caps2_n_caps)
        decoder_inputs = create_multiple_masked_inputs(caps_to_visualize,caps2_output,caps2_n_caps,caps2_n_dims,mask_with_labels)

        # decoder layer sizes
        n_hidden1 = 512
        n_hidden2 = 1024
        n_output = im_size[0] * im_size[1]

        # run decoder
        decoder_outputs = decoder_with_mask(decoder_inputs, n_hidden1, n_hidden2, n_output)
        decoder_output_images = tf.reshape(decoder_outputs, [-1, im_size[0], im_size[1], caps_to_visualize])

        decoder_outputs_overlay = np.zeros(shape=(im_size[0],im_size[1],3))
        for cap in caps_to_visualize:
            color_mask = [1,0,0]
            decoder_outputs_overlay += np.reshape(decoder_output_images[:, :, :, cap], image_batch.shape[0])*color_mask
        tf.summary.image('decoder_outputs_overlay', decoder_outputs_overlay)

    # summary writer
    summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter('run_stimuli_' + model + '_logdir', sess.graph)

    decoder_output_images, summ = sess.run(
        [decoder_output_images, summary],
        feed_dict={X: image_batch,
                   y: image_labels,
                   mask_with_labels: False})

    writer.add_summary(summ, 1)