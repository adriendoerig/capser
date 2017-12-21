#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# from https://github.com/ageron/handson-ml/blob/master/extra_capsnets.ipynb
# video: https://www.youtube.com/watch?v=2Kawrd5szHE&feature=youtu.be
from __future__ import division, print_function, unicode_literals
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from create_sprite import images_to_sprite, invert_grayscale
from data_handling_functions import make_shape_sets
from capsule_functions import primary_caps_layer, primary_to_fc_caps_layer, \
    fc_to_fc_caps_layer, caps_prediction, compute_margin_loss, create_masked_decoder_input, \
    decoder_with_mask, compute_reconstruction_loss

# reproducibility
tf.reset_default_graph()
np.random.seed(42)
tf.set_random_seed(42)

# create datasets
im_size = (100,150)
train_set, train_labels, valid_set, valid_labels, test_set, test_labels = make_shape_sets(image_size=im_size)

show_samples = 0
if show_samples:

    n_samples = 5

    plt.figure(figsize=(n_samples * 2, 3))
    for index in range(n_samples):
        plt.subplot(1, n_samples, index + 1)
        sample_image = train_set[index,:,:].reshape(im_size[0], im_size[1])
        plt.imshow(sample_image, cmap="binary")
        plt.axis("off")
    plt.show()

# create sprites and embedding labels from test set for embedding visualization in tensorboard
sprites = invert_grayscale(images_to_sprite(np.squeeze(test_set)))
plt.imsave(os.path.join(os.getcwd(), 'shape_sprites.png'),sprites,cmap='gray')

with open(os.path.join(os.getcwd(), 'shape_embedding_labels_3layers.tsv'),'w') as f:
    f.write("Index\tLabel\n")
    for index,label in enumerate(test_labels):
        f.write("%d\t%d\n" % (index,label))

show_sprites = 0
if show_sprites:
    plt.imshow(sprites,cmap='gray')
    plt.show()

# placeholder for input images and labels
X = tf.placeholder(shape=[None, im_size[0], im_size[1], 1], dtype=tf.float32, name="X")
x_image = tf.reshape(X,[-1, im_size[0], im_size[1],1])
tf.summary.image('input',x_image,6)
y = tf.placeholder(shape=[None], dtype=tf.int64, name="y")

########################################################################################################################
# From input to caps1
########################################################################################################################


# primary capsules -- The first layer will be composed of 32 maps of 6×6 capsules each,
# where each capsule will output an 8D activation vector.

kernel_size = 9
caps_conv_stride = 2
caps1_n_maps = 50
caps1_n_caps = int(caps1_n_maps * ((im_size[0]-2*(kernel_size-1))/2)*((im_size[1]-2*(kernel_size-1))/2))  # number of primary capsules: 2*kernel_size convs, stride = 2 in caps conv layer
caps1_n_dims = 8

# The first step is to create a convolutional layer as a feature extractor. We will feed it to the primary_caps_layer
# function to create the first capsule layer.
conv1_params = {
    "filters": 256,
    "kernel_size": kernel_size,
    "strides": 1,
    "padding": "valid",
    "activation": tf.nn.relu,
}
conv1 = tf.layers.conv2d(X, name="conv1", **conv1_params) # ** means that conv1_params is a dict {param_name:param_value}
tf.summary.histogram('1st_conv_layer', conv1)

# create furst capsule layer
caps1_output = primary_caps_layer(conv1, caps1_n_maps, caps1_n_caps, caps1_n_dims,
                     kernel_size, caps_conv_stride, conv_padding='valid', conv_activation=tf.nn.relu)


########################################################################################################################
# From caps1 to caps2
########################################################################################################################


caps2_n_caps = 7 # number of capsules
caps2_n_dims = 16 # of n dimensions

# it is all taken care of by the function
caps2_output = primary_to_fc_caps_layer(X, caps1_output, caps1_n_caps, caps1_n_dims, caps2_n_caps, caps2_n_dims, rba_rounds=3, print_shapes=False)


########################################################################################################################
# From caps2 to caps3
########################################################################################################################


caps3_n_caps = 7 # number of capsules
caps3_n_dims = 16 # of n dimensions

# it is all taken care of by the function
caps3_output = fc_to_fc_caps_layer(X, caps2_output, caps2_n_caps, caps2_n_dims, caps3_n_caps, caps3_n_dims, rba_rounds=3, print_shapes=False)


########################################################################################################################
# Create embedding of the secondary capsules output
########################################################################################################################


LABELS = os.path.join(os.getcwd(), 'shape_embedding_labels_3layers.tsv')
SPRITES = os.path.join(os.getcwd(), 'shape_sprites.png')
embedding_input = tf.reshape(caps3_output,[-1,caps3_n_caps*caps3_n_dims])
embedding_size = caps3_n_caps*caps3_n_dims
embedding = tf.Variable(tf.zeros([test_set.shape[0],embedding_size]), name='final_capsules_embedding')
assignment = embedding.assign(embedding_input)
config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
embedding_config = config.embeddings.add()
embedding_config.tensor_name = embedding.name
embedding_config.sprite.image_path = SPRITES
embedding_config.metadata_path = LABELS
# Specify the width and height (in this order!) of a single thumbnail.
embedding_config.sprite.single_image_dim.extend([max(im_size), max(im_size)])


########################################################################################################################
# Estimated class probabilities
########################################################################################################################


y_pred = caps_prediction(caps3_output, print_shapes=False)# get index of max probability


########################################################################################################################
# Compute the margin loss
########################################################################################################################


# parameters for the margin loss
m_plus = 0.9
m_minus = 0.1
lambda_ = 0.5

margin_loss = compute_margin_loss(y, caps3_output, caps3_n_caps, m_plus, m_minus, lambda_)


########################################################################################################################
# Reconstruction & reconstruction error
########################################################################################################################

# create the mask. first, we create a placeholder that will tell the program whether to use the true
# or the predicted labels
mask_with_labels = tf.placeholder_with_default(False, shape=(),
                                               name="mask_with_labels")

# create the mask
decoder_input = create_masked_decoder_input(y, y_pred, caps3_output, caps3_n_caps, caps3_n_dims,
                                            mask_with_labels, print_shapes=False)

# decoder layer sizes
n_hidden1 = 512
n_hidden2 = 1024
n_output = im_size[0] * im_size[1]

# run decoder
decoder_output = decoder_with_mask(decoder_input,n_hidden1,n_hidden2,n_output)
decoder_output_image = tf.reshape(decoder_output,[-1, im_size[0], im_size[1],1])
tf.summary.image('decoder_output',decoder_output_image,6)

### RECONSTRUCTION LOSS ###

reconstruction_loss = compute_reconstruction_loss(X,decoder_output)


########################################################################################################################
# Final loss, accuracy, training operations, init & saver
########################################################################################################################


### FINAL LOSS & ACCURACY ###

alpha = 0.0005

with tf.name_scope('total_loss'):
    loss = tf.add(margin_loss, alpha * reconstruction_loss, name="loss")
    tf.summary.scalar('total_loss',loss)

with tf.name_scope('accuracy'):
    correct = tf.equal(y, y_pred, name="correct")
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
    tf.summary.scalar('accuracy',accuracy)

### TRAINING OPERATIONS ###

optimizer = tf.train.AdamOptimizer()
training_op = optimizer.minimize(loss, name="training_op")

### INIT & SAVER ###

init = tf.global_variables_initializer()
saver = tf.train.Saver()


########################################################################################################################
# Training
########################################################################################################################


n_epochs = 100
batch_size = 10
restore_checkpoint = True
n_iterations_per_epoch = train_set.shape[0] // batch_size
n_iterations_validation = valid_set.shape[0] // batch_size
best_loss_val = np.infty
checkpoint_path = "./my_capsule_network_shapes_3layers"

with tf.Session() as sess:

    summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter('caps_logdir_shapes_3layers',sess.graph)
    tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)

    if restore_checkpoint and tf.train.checkpoint_exists(checkpoint_path):
        saver.restore(sess, checkpoint_path)
        pass
    else:
        init.run()

        for epoch in range(n_epochs):
            for iteration in range(1, n_iterations_per_epoch + 1):

                # get data in the batches
                offset = (iteration * batch_size) % (train_labels.shape[0] - batch_size)
                batch_data = train_set[offset:(offset + batch_size), :, :, :]
                batch_labels = train_labels[offset:(offset + batch_size)]

                # Run the training operation and measure the loss:
                _, loss_train, summ = sess.run(
                    [training_op, loss, summary],
                    feed_dict={X: batch_data,
                               y: batch_labels,
                               mask_with_labels: True})
                #print("\rIteration: {}/{} ({:.1f}%)  Loss: {:.5f}".format(
                 #         iteration, n_iterations_per_epoch,
                  #        iteration * 100 / n_iterations_per_epoch,
                   #       loss_train),
                    #  end="")
                if iteration % 5 == 0:
                    writer.add_summary(summ,epoch*n_iterations_per_epoch+iteration)
                if iteration == n_iterations_per_epoch and epoch is n_epochs:
                    # X_batch, y_batch = mnist.validation.next_batch(1024)
                    sess.run(assignment,feed_dict={X: test_set,
                                                   y: batch_labels,
                                                   mask_with_labels: True})
                    saver.save(sess, os.path.join('caps_logdir_shapes_3layers','model.ckpt'),epoch)

            # At the end of each epoch,
            # measure the validation loss and accuracy:
            loss_vals = []
            acc_vals = []
            for iteration in range(1, n_iterations_validation + 1):

                # get data in the batches
                offset = (iteration * batch_size) % (valid_labels.shape[0] - batch_size)
                batch_data = valid_set[offset:(offset + batch_size), :, :, :]
                batch_labels = valid_labels[offset:(offset + batch_size)]

                loss_val, acc_val = sess.run(
                        [loss, accuracy],
                        feed_dict={X: batch_data,
                                   y: batch_labels,
                                   mask_with_labels: True})
                loss_vals.append(loss_val)
                acc_vals.append(acc_val)
                # print("\rEvaluating the model: {}/{} ({:.1f}%)".format(
                 #         iteration, n_iterations_validation,
                  #        iteration * 100 / n_iterations_validation),
                   #   end=" " * 10)
            loss_val = np.mean(loss_vals)
            acc_val = np.mean(acc_vals)
            # print("\rEpoch: {}  Val accuracy: {:.4f}%  Loss: {:.6f}{}".format(
             #   epoch + 1, acc_val * 100, loss_val,
              #  " (improved)" if loss_val < best_loss_val else ""))

            # And save the model if it improved:
            # if loss_val < best_loss_val:
            #     save_path = saver.save(sess, checkpoint_path)
            #     best_loss_val = loss_val
        save_path = saver.save(sess, checkpoint_path)
        best_loss_val = loss_val

########################################################################################################################
# Testing
########################################################################################################################


do_testing = 1

n_iterations_test = test_set.shape[0] // batch_size

if do_testing:
    with tf.Session() as sess:
        saver.restore(sess, checkpoint_path)

        loss_tests = []
        acc_tests = []
        for iteration in range(1, n_iterations_test + 1):

            # get data in the batches
            offset = (iteration * batch_size) % (test_labels.shape[0] - batch_size)
            batch_data = test_set[offset:(offset + batch_size), :, :, :]
            batch_labels = test_labels[offset:(offset + batch_size)]

            loss_test, acc_test = sess.run(
                    [loss, accuracy],
                    feed_dict={X: batch_data,
                                   y: batch_labels,
                                   mask_with_labels: True})
            loss_tests.append(loss_test)
            acc_tests.append(acc_test)
            #print("\rEvaluating the model: {}/{} ({:.1f}%)".format(
            #          iteration, n_iterations_test,
            #          iteration * 100 / n_iterations_test),
            #      end=" " * 10)
        loss_test = np.mean(loss_tests)
        acc_test = np.mean(acc_tests)
        #print("\rFinal test accuracy: {:.4f}%  Loss: {:.6f}".format(
         #   acc_test * 100, loss_test))


########################################################################################################################
# View predictions
########################################################################################################################


# Now let's make some predictions! We first fix a few images from the test set, then we start a session,
# restore the trained model, evaluate caps3_output to get the capsule network's output vectors, decoder_output
# to get the reconstructions, and y_pred to get the class predictions.
n_samples = 5

sample_images = test_set[:n_samples,:,:]

with tf.Session() as sess:
    saver.restore(sess, checkpoint_path)
    caps3_output_value, decoder_output_value, y_pred_value = sess.run(
            [caps3_output, decoder_output, y_pred],
            feed_dict={X: sample_images,
                       y: np.array([], dtype=np.int64)})

# plot images with their reconstruction
sample_images = sample_images.reshape(-1, im_size[0], im_size[1])
reconstructions = decoder_output_value.reshape([-1, im_size[0], im_size[1]])

plt.figure(figsize=(n_samples * 2, 3))
for index in range(n_samples):
    plt.subplot(1, n_samples, index + 1)
    plt.imshow(sample_images[index], cmap="binary")
    plt.title("Label:" + str(test_labels[index]))
    plt.axis("off")

plt.show()

plt.figure(figsize=(n_samples * 2, 3))
for index in range(n_samples):
    plt.subplot(1, n_samples, index + 1)
    plt.title("Predicted:" + str(y_pred_value[index]))
    plt.imshow(reconstructions[index], cmap="binary")
    plt.axis("off")

plt.show()


########################################################################################################################
# Interpreting the output vectors
########################################################################################################################


# caps3_output is now a numpy array. let's check its shape
# print('shape of caps3_output np array: '+str(caps3_output_value.shape))

# Let's create a function that will tweak each of the 16 pose parameters (dimensions) in all output vectors.
# Each tweaked output vector will be identical to the original output vector, except that one of its pose
# parameters will be incremented by a value varying from -0.5 to 0.5. By default there will be 11 steps
# (-0.5, -0.4, ..., +0.4, +0.5). This function will return an array of shape
# (tweaked pose parameters=16, steps=11, batch size=5, 1, caps3_n_caps, caps3_n_dims, 1):
def tweak_pose_parameters(output_vectors, min=-0.5, max=0.5, n_steps=11):
    steps = np.linspace(min, max, n_steps) # -0.25, -0.15, ..., +0.25
    pose_parameters = np.arange(caps3_n_dims) # 0, 1, ..., 15
    tweaks = np.zeros([caps3_n_dims, n_steps, 1, 1, 1, caps3_n_dims, 1])
    tweaks[pose_parameters, :, 0, 0, 0, pose_parameters, 0] = steps
    output_vectors_expanded = output_vectors[np.newaxis, np.newaxis]
    return tweaks + output_vectors_expanded

# get a tweaked parameters array for the caps3_output_value array, and reshape (i.e., flattent) it to feed to decoder.
n_steps = 11
tweaked_vectors = tweak_pose_parameters(caps3_output_value, n_steps=n_steps)
tweaked_vectors_reshaped = tweaked_vectors.reshape(
    [-1, 1, caps3_n_caps, caps3_n_dims, 1])

# feed to decoder
tweak_labels = np.tile(test_labels[:n_samples], caps3_n_dims * n_steps)

# get reconstruction
with tf.Session() as sess:
    saver.restore(sess, checkpoint_path)
    decoder_output_value = sess.run(
            decoder_output,
            feed_dict={caps3_output: tweaked_vectors_reshaped,
                       mask_with_labels: True,
                       y: tweak_labels})

# reshape to make things easier to travel in dimension, n_steps and sample
tweak_reconstructions = decoder_output_value.reshape(
        [caps3_n_dims, n_steps, n_samples, im_size[0], im_size[1]])

# plot the tweaked versions!
for dim in range(caps3_n_caps):
    #print("Tweaking output dimension #{}".format(dim))
    plt.figure(figsize=(n_steps / 1.2, n_samples / 1.5))
    for row in range(n_samples):
        for col in range(n_steps):
            plt.subplot(n_samples, n_steps, row * n_steps + col + 1)
            plt.imshow(tweak_reconstructions[dim, col, row], cmap="binary")
            plt.axis("off")
    plt.show()


