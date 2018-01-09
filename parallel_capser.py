import tensorflow as tf
from capser_general import capser_general_2_caps_layers
import numpy as np
from data_handling_functions import make_shape_sets
import matplotlib.pyplot as plt
import os
from create_sprite import images_to_sprite, invert_grayscale

####################################################################################################################
# Reproducibility & directories
####################################################################################################################


# reproducibility
tf.reset_default_graph()
np.random.seed(42)
tf.set_random_seed(42)

# directories
MAIN_NAME = 'PARALLEL_CAPSER'
LOGDIR = MAIN_NAME+'_logdir'
image_output_dir = 'output_images/'+MAIN_NAME

if not os.path.exists(LOGDIR):
    os.makedirs(LOGDIR)
if not os.path.exists(image_output_dir):
    os.makedirs(image_output_dir)


########################################################################################################################
# Data handling
########################################################################################################################


# create datasets
im_size = (70, 145)
train_set, train_labels, valid_set, valid_labels, test_set, test_labels \
    = make_shape_sets(folder='./crowding_images/shapes_simple_large',image_size=im_size, n_repeats=10)

# placeholders for input images and labels
X = tf.placeholder(shape=[None, im_size[0], im_size[1], 1], dtype=tf.float32, name="X")
x_image = tf.reshape(X,[-1, im_size[0], im_size[1],1])
tf.summary.image('input', x_image,6)
y = tf.placeholder(shape=[None], dtype=tf.int64, name="y")
# create a placeholder that will tell the program whether to use the true or the predicted labels
mask_with_labels = tf.placeholder_with_default(False, shape=(), name="mask_with_labels")

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


########################################################################################################################
# Network parameters
########################################################################################################################

MODEL_NAMES = {
    0: 'CAPSER_ZERO',
    1: 'CAPSER_ONE'
}

# early conv layers
conv1_params = [{
                    "filters": 64,
                    "kernel_size": 5,
                    "strides": 1,
                    "padding": "valid",
                    "activation": tf.nn.relu,
                },
                {   "filters": 64,
                    "kernel_size": 5,
                    "strides": 1,
                    "padding": "valid",
                    "activation": tf.nn.relu,
                }]
conv2_params = [{
                        "filters": 64,
                        "kernel_size": 5,
                        "strides": 1,
                        "padding": "valid",
                        "activation": tf.nn.relu,
                    },
                    {
                        "filters": 64,
                        "kernel_size": 5,
                        "strides": 1,
                        "padding": "valid",
                        "activation": tf.nn.relu,
                    }]
conv3_params = [None, None]

# primary capsules
caps1_n_maps = [8, 8]  # number of capsules at level 1 of capsules
caps1_n_dims = [16, 16]  # number of dimension per capsule
conv_caps_params = [{
                        "filters": caps1_n_maps[0] * caps1_n_dims[0],
                        "kernel_size": 7,
                        "strides": 2,
                        "padding": "valid",
                        "activation": tf.nn.relu,
                    },
                    {
                        "filters": caps1_n_maps[1] * caps1_n_dims[1],
                        "kernel_size": 7,
                        "strides": 2,
                        "padding": "valid",
                        "activation": tf.nn.relu,
                    }]

# output capsules
caps2_n_caps = [8, 8]  # number of capsules
caps2_n_dims = [16, 16] # of n dimensions ### TRY 50????

# decoder layer sizes
n_hidden1 = [256, 1024]
n_hidden2 = [512, 2048]
n_hidden3 = [None, None]
n_output = im_size[0] * im_size[1]


########################################################################################################################
# Create Networks
########################################################################################################################

with tf.variable_scope(MODEL_NAMES[0]):
    with tf.device('/cpu:0'):
        capser_zero = capser_general_2_caps_layers(X, y, im_size, conv1_params[0], conv2_params[0], conv3_params[0],
                                                   caps1_n_maps[0], caps1_n_dims[0], conv_caps_params[0],
                                                   caps2_n_caps[0], caps2_n_dims[0],
                                                   n_hidden1[0], n_hidden2[0], n_hidden3[0], n_output,
                                                   mask_with_labels,
                                                   MODEL_NAMES[0])
with tf.variable_scope(MODEL_NAMES[1]):
    with tf.device('/cpu:1'):
        capser_one = capser_general_2_caps_layers(X, y, im_size, conv1_params[1], conv2_params[1], conv3_params[1],
                                                   caps1_n_maps[1], caps1_n_dims[1], conv_caps_params[1],
                                                   caps2_n_caps[1], caps2_n_dims[1],
                                                   n_hidden1[1], n_hidden2[1], n_hidden3[1], n_output,
                                                   mask_with_labels,
                                                   MODEL_NAMES[0])

# op to train all networks
do_training = [capser_zero["training_op"],capser_zero["training_op"]]

### SAVER & SUMMARY ###

saver = tf.train.Saver()
summary = tf.summary.merge_all()

########################################################################################################################
# Run whatever needed
########################################################################################################################


# training parameters
n_epochs = 2
batch_size = 25
restore_checkpoint = True
n_iterations_per_epoch = train_set.shape[0] // batch_size
n_iterations_validation = valid_set.shape[0] // batch_size
best_loss_val = np.infty
checkpoint_path = os.path.join(os.getcwd(),LOGDIR+'/'+MAIN_NAME+"_model.ckpt")
init = tf.global_variables_initializer()

with tf.Session() as sess:

    summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(LOGDIR,sess.graph)

    if restore_checkpoint and tf.train.checkpoint_exists(checkpoint_path):
        saver.restore(sess, checkpoint_path)
        print('Checkpoint found, skipping training.')
        # pass # ATTENTION ENLEVER PASS ICI POUR RECOMMENCER TRAINING DEPUIS ICI
    # else:    # ATTENTION ENLEVER LE ELSE AUSSI DONC TOUT CE QUI EST DANS CE ELSE SERA DETABE (LE IF DU HAUT SE FERME)
    init.run()

    for epoch in range(1,1+n_epochs):
        for iteration in range(1, 1+n_iterations_per_epoch):

            # get data in the batches
            offset = (iteration * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_set[offset:(offset + batch_size), :, :, :]
            batch_labels = train_labels[offset:(offset + batch_size)]

            # Run the training operation and measure the loss:
            _, loss_train, summ = sess.run(
                [do_training, capser_one["loss"], summary],
                feed_dict={X: batch_data,
                           y: batch_labels,
                           mask_with_labels: True})
            print("\rIteration: {}/{} ({:.1f}%)  Loss: {:.5f}".format(
                      iteration, n_iterations_per_epoch,
                      iteration * 100 / n_iterations_per_epoch,
                      loss_train),
                  end="")
            if iteration % 5 == 0:
                writer.add_summary(summ,(epoch-1)*n_iterations_per_epoch+iteration)
