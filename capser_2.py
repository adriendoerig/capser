# from https://github.com/ageron/handson-ml/blob/master/extra_capsnets.ipynb
# video: https://www.youtube.com/watch?v=2Kawrd5szHE&feature=youtu.be
from __future__ import division, print_function, unicode_literals
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from create_sprite import images_to_sprite, invert_grayscale
from data_handling_functions import make_shape_sets
from capsule_functions import primary_caps_layer, primary_to_fc_caps_layer, \
    caps_prediction, compute_margin_loss, create_masked_decoder_input, \
    create_multiple_masked_inputs, decoder_with_mask, each_capsule_decoder_with_mask, \
    create_capsule_overlay, compute_reconstruction_loss

# reproducibility
tf.reset_default_graph()
np.random.seed(42)
tf.set_random_seed(42)

# choose true to compute a color overlay image of each capsule's decoder
do_color_image = True

# create datasets
im_size = (60, 128)
train_set, train_labels, valid_set, valid_labels, test_set, test_labels = make_shape_sets(image_size=im_size, n_repeats=5)

show_samples = 0
if show_samples:

    n_samples = 5

    plt.figure(figsize=(n_samples * 2, 3))
    for index in range(n_samples):
        plt.subplot(1, n_samples, index + 1)
        sample_image = train_set[index, :, :].reshape(im_size[0], im_size[1])
        plt.imshow(sample_image, cmap="binary")
        plt.axis("off")
    plt.show()

# create sprites and embedding labels from test set for embedding visualization in tensorboard
sprites = invert_grayscale(images_to_sprite(np.squeeze(test_set)))
plt.imsave(os.path.join(os.getcwd(), 'capser_2_sprites.png'), sprites, cmap='gray')

with open(os.path.join(os.getcwd(), 'capser_2_embedding_labels.tsv'), 'w') as f:
    f.write("Index\tLabel\n")
    for index, label in enumerate(test_labels):
        f.write("%d\t%d\n" % (index, label))

show_sprites = 0
if show_sprites:
    plt.imshow(sprites, cmap='gray')
    plt.show()

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
print(conv1,conv1b)

# create furst capsule layer
caps1_output = primary_caps_layer(conv1b, caps1_n_maps, caps1_n_caps, caps1_n_dims,
                     kernel_size, caps_conv_stride, conv_padding='valid', conv_activation=tf.nn.relu, print_shapes=True)


########################################################################################################################
# From caps1 to caps2
########################################################################################################################


caps2_n_caps = 8 # number of capsules
caps2_n_dims = 10 # of n dimensions

# it is all taken care of by the function
caps2_output = primary_to_fc_caps_layer(X, caps1_output, caps1_n_caps, caps1_n_dims, caps2_n_caps, caps2_n_dims, rba_rounds=3, print_shapes=False)


########################################################################################################################
# Create embedding of the secondary capsules output
########################################################################################################################


LABELS = os.path.join(os.getcwd(), 'capser_2_embedding_labels.tsv')
SPRITES = os.path.join(os.getcwd(), 'capser_2_sprites.png')
embedding_input = tf.reshape(caps2_output,[-1,caps2_n_caps*caps2_n_dims])
embedding_size = caps2_n_caps*caps2_n_dims
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


y_pred = caps_prediction(caps2_output, print_shapes=False)# get index of max probability


########################################################################################################################
# Compute the margin loss
########################################################################################################################


# parameters for the margin loss
m_plus = 0.9
m_minus = 0.1
lambda_ = 0.5

if do_color_image is False:
    margin_loss = compute_margin_loss(y, caps2_output, caps2_n_caps, m_plus, m_minus, lambda_)


########################################################################################################################
# Reconstruction & reconstruction error
########################################################################################################################

with tf.name_scope('decoder'):

    # create the mask. first, we create a placeholder that will tell the program whether to use the true
    # or the predicted labels
    mask_with_labels = tf.placeholder_with_default(False, shape=(),
                                                   name="mask_with_labels")
    if do_color_image is False:

        # create the mask
        decoder_input = create_masked_decoder_input(y, y_pred, caps2_output, caps2_n_caps, caps2_n_dims,
                                                    mask_with_labels, print_shapes=False)

        # decoder layer sizes
        n_hidden1 = 512
        n_hidden2 = 1024
        n_output = im_size[0] * im_size[1]

        # run decoder
        decoder_output = decoder_with_mask(decoder_input, n_hidden1, n_hidden2, n_output)
        decoder_output_image = tf.reshape(decoder_output,[-1, im_size[0], im_size[1], 1])
        tf.summary.image('decoder_output',decoder_output_image,6)

        ### RECONSTRUCTION LOSS ###

        reconstruction_loss = compute_reconstruction_loss(X,decoder_output)

    ### CREATE COLORED RECONSTRUCTION IMAGE: ONE COLOR PER CAPSULE ###

    if do_color_image is True:
        print('DOING COLOR IS TRUE: COMPUTING ONE COLOR PER CAPSULE, NOT COMPUTING LOSS ETC.')
        with tf.name_scope('Visualize_colored_capsule_outputs'):

            caps_to_visualize = range(caps2_n_caps)
            decoder_inputs = create_multiple_masked_inputs(caps_to_visualize, caps2_output, caps2_n_caps, caps2_n_dims,
                                                           mask_with_labels)

            # run decoder
            decoder_outputs = each_capsule_decoder_with_mask(decoder_inputs, n_hidden1, n_hidden2, n_output)

            #create overlay image
            decoder_output_images = tf.reshape(decoder_outputs, [-1, im_size[0], im_size[1], caps_to_visualize])
            decoder_outputs_overlay = create_capsule_overlay(decoder_output_images,range(caps2_n_caps),im_size)
            tf.summary.image('decoder_outputs_overlay', decoder_outputs_overlay, min(10,X.shape[0]))

########################################################################################################################
# Final loss, accuracy, training operations, init & saver
########################################################################################################################


### FINAL LOSS & ACCURACY ###

alpha = 0.0005

if do_color_image is False:
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


n_epochs = 5
batch_size = 65
restore_checkpoint = True
n_iterations_per_epoch = train_set.shape[0] // batch_size
n_iterations_validation = valid_set.shape[0] // batch_size
best_loss_val = np.infty
checkpoint_path = "./capser_1d files/model_capser_1d"

with tf.Session() as sess:

    summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter('capser_2_logdir',sess.graph)
    tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)

    if restore_checkpoint and tf.train.checkpoint_exists(checkpoint_path):
        saver.restore(sess, checkpoint_path)
        print('Checkpoint found, skipping training.')
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
                print("\rIteration: {}/{} ({:.1f}%)  Loss: {:.5f}".format(
                          iteration, n_iterations_per_epoch,
                          iteration * 100 / n_iterations_per_epoch,
                          loss_train),
                      end="")
                if iteration % 5 == 0:
                    writer.add_summary(summ,epoch*n_iterations_per_epoch+iteration)

                if iteration == n_iterations_per_epoch and epoch is n_epochs:
                    print('Creating embedding')
                    sess.run(assignment, feed_dict={X: test_set,
                                                    y: test_labels,
                                                    mask_with_labels: False})
                    saver.save(sess, os.path.join('capser_2_logdir', 'model.ckpt'),
                               n_epochs * n_iterations_per_epoch)


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
                                   mask_with_labels: False})
                loss_vals.append(loss_val)
                acc_vals.append(acc_val)
                print("\rEvaluating the model: {}/{} ({:.1f}%)".format(
                          iteration, n_iterations_validation,
                          iteration * 100 / n_iterations_validation),
                      end=" " * 10)
            loss_val = np.mean(loss_vals)
            acc_val = np.mean(acc_vals)
            print("\rEpoch: {}  Val accuracy: {:.4f}%  Loss: {:.6f}{}".format(
                epoch + 1, acc_val * 100, loss_val,
                " (improved)" if loss_val < best_loss_val else ""))

            # And save the model if it improved:
            # if loss_val < best_loss_val:
            #     save_path = saver.save(sess, checkpoint_path)
            #     best_loss_val = loss_val

        # save the model at the end
        save_path = saver.save(sess, checkpoint_path)
        best_loss_val = loss_val

########################################################################################################################
# Testing
########################################################################################################################


do_testing = 0

n_iterations_test = test_set.shape[0] // batch_size

if do_testing:
    with tf.Session() as sess:
        print('testing')
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
            print("\rEvaluating the model: {}/{} ({:.1f}%)".format(
                      iteration, n_iterations_test,
                      iteration * 100 / n_iterations_test),
                  end=" " * 10)
        loss_test = np.mean(loss_tests)
        acc_test = np.mean(acc_tests)
        print("\rFinal test accuracy: {:.4f}%  Loss: {:.6f}".format(
            acc_test * 100, loss_test))


########################################################################################################################
# View predictions or capsule overlay image
########################################################################################################################

### CAPSULE OVERLAY IMAGE ###

if do_color_image is True:
    with tf.Session() as sess:
        saver.restore(sess, checkpoint_path)
        color_writer = tf.summary.FileWriter('capser_2_color_logdir', sess.graph)
        decoder_outputs_overlay_result, summ = sess.run(
                [decoder_outputs_overlay, summary],
                feed_dict={X: train_set,
                           y: train_labels})
        color_writer.add_summary(summ,1)

### PREDICTIONS ###

if do_color_image is False:
    image_output_dir = './output images/capser_2'

    # Now let's make some predictions! We first fix a few images from the test set, then we start a session,
    # restore the trained model, evaluate caps2_output to get the capsule network's output vectors, decoder_output
    # to get the reconstructions, and y_pred to get the class predictions.
    n_samples = 25

    sample_images = test_set[:n_samples,:,:]

    with tf.Session() as sess:
        saver.restore(sess, checkpoint_path)
        caps2_output_value, decoder_output_value, y_pred_value = sess.run(
                [caps2_output, decoder_output, y_pred],
                feed_dict={X: sample_images,
                           y: np.array([], dtype=np.int64)})

    # plot images with their reconstruction
    sample_images = sample_images.reshape(-1, im_size[0], im_size[1])
    reconstructions = decoder_output_value.reshape([-1, im_size[0], im_size[1]])

    plt.figure(figsize=(n_samples / 1.5, n_samples / 1.5))
    for index in range(n_samples):
        plt.subplot(5, 5, index + 1)
        plt.imshow(sample_images[index], cmap="binary")
        plt.title("Label:" + str(test_labels[index]))
        plt.axis("off")

    plt.savefig(image_output_dir+'sample images')
    #plt.show()

    plt.figure(figsize=(n_samples / 1.5, n_samples / 1.5))
    for index in range(n_samples):
        plt.subplot(5, 5, index + 1)
        plt.title("Predicted:" + str(y_pred_value[index]))
        plt.imshow(reconstructions[index], cmap="binary")
        plt.axis("off")

    plt.savefig(image_output_dir+'sample images reconstructed')
    #plt.show()


    ########################################################################################################################
    # Interpreting the output vectors
    ########################################################################################################################


    # caps2_output is now a numpy array. let's check its shape
    # print('shape of caps2_output np array: '+str(caps2_output_value.shape))

    # Let's create a function that will tweak each of the 16 pose parameters (dimensions) in all output vectors.
    # Each tweaked output vector will be identical to the original output vector, except that one of its pose
    # parameters will be incremented by a value varying from -0.5 to 0.5. By default there will be 11 steps
    # (-0.5, -0.4, ..., +0.4, +0.5). This function will return an array of shape
    # (tweaked pose parameters=16, steps=11, batch size=5, 1, caps2_n_caps, caps2_n_dims, 1):
    def tweak_pose_parameters(output_vectors, min=-0.5, max=0.5, n_steps=11):
        steps = np.linspace(min, max, n_steps) # -0.25, -0.15, ..., +0.25
        pose_parameters = np.arange(caps2_n_dims) # 0, 1, ..., 15
        tweaks = np.zeros([caps2_n_dims, n_steps, 1, 1, 1, caps2_n_dims, 1])
        tweaks[pose_parameters, :, 0, 0, 0, pose_parameters, 0] = steps
        output_vectors_expanded = output_vectors[np.newaxis, np.newaxis]
        return tweaks + output_vectors_expanded

    # get a tweaked parameters array for the caps2_output_value array, and reshape (i.e., flattent) it to feed to decoder.
    n_steps = 11
    tweaked_vectors = tweak_pose_parameters(caps2_output_value, n_steps=n_steps)
    tweaked_vectors_reshaped = tweaked_vectors.reshape(
        [-1, 1, caps2_n_caps, caps2_n_dims, 1])

    # feed to decoder
    tweak_labels = np.tile(test_labels[:n_samples], caps2_n_dims * n_steps)

    # get reconstruction
    with tf.Session() as sess:
        saver.restore(sess, checkpoint_path)
        decoder_output_value = sess.run(
                decoder_output,
                feed_dict={caps2_output: tweaked_vectors_reshaped,
                           mask_with_labels: True,
                           y: tweak_labels})

    # reshape to make things easier to travel in dimension, n_steps and sample
    tweak_reconstructions = decoder_output_value.reshape(
            [caps2_n_dims, n_steps, n_samples, im_size[0], im_size[1]])

    # plot the tweaked versions!
    for dim in range(caps2_n_dims):
        print("Tweaking output dimension #{}".format(dim))
        plt.figure(figsize=(n_steps / 1.2, n_samples / 1.5))
        for row in range(n_samples):
            for col in range(n_steps):
                plt.subplot(n_samples, n_steps, row * n_steps + col + 1)
                plt.imshow(tweak_reconstructions[dim, col, row], cmap="binary")
                plt.axis("off")
        # plt.show()
        plt.savefig(image_output_dir + 'tweak dimension' + str(dim))



