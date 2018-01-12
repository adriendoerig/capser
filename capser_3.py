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
MODEL_NAME = 'capser_3'
LOGDIR = MODEL_NAME+'_logdir'
image_output_dir = 'output_images/'+MODEL_NAME

if not os.path.exists(LOGDIR):
    os.makedirs(LOGDIR)
if not os.path.exists(image_output_dir):
    os.makedirs(image_output_dir)


########################################################################################################################
# Data handling
########################################################################################################################


# create datasets
im_size = (60, 128)
train_set, train_labels, valid_set, valid_labels, test_set, test_labels \
    = make_shape_sets(folder='./crowding_images/shapes_simple',image_size=im_size, n_repeats=1000)

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
# Parameters
########################################################################################################################

# training parameters
n_epochs = 1
batch_size = 65
restore_checkpoint = True
continue_training_from_checkpoint = False

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

# op to train all networks
do_training = capser["training_op"]


########################################################################################################################
# Create embedding of the secondary capsules output
########################################################################################################################


# create sprites and embedding labels from test set for embedding visualization in tensorboard
sprites = invert_grayscale(images_to_sprite(np.squeeze(test_set)))
plt.imsave(os.path.join(os.getcwd(), LOGDIR+'/'+MODEL_NAME+'_sprites.png'), sprites, cmap='gray')

with open(os.path.join(os.getcwd(), LOGDIR+'/'+MODEL_NAME+'_embedding_labels.tsv'), 'w') as f:
    f.write("Index\tLabel\n")
    for index, label in enumerate(test_labels):
        f.write("%d\t%d\n" % (index, label))

show_sprites = 0
if show_sprites:
    plt.imshow(sprites, cmap='gray')
    plt.show()

LABELS = os.path.join(os.getcwd(), LOGDIR+'/'+MODEL_NAME+'_embedding_labels.tsv')
SPRITES = os.path.join(os.getcwd(), LOGDIR+'/'+MODEL_NAME+'_sprites.png')
embedding_input = tf.reshape(capser["caps2_output"],[-1,caps2_n_caps*caps2_n_dims])
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
# Run whatever needed
########################################################################################################################


# training parameters
n_iterations_per_epoch = train_set.shape[0] // batch_size
n_iterations_validation = valid_set.shape[0] // batch_size
best_loss_val = np.infty

saver = tf.train.Saver()
checkpoint_path = os.path.join(os.getcwd(),LOGDIR+'/'+MODEL_NAME+"_model.ckpt")
init = tf.global_variables_initializer()
summary = tf.summary.merge_all()

with tf.Session() as sess:

    writer = tf.summary.FileWriter(LOGDIR,sess.graph)
    tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)

    if restore_checkpoint and tf.train.checkpoint_exists(checkpoint_path):
        saver.restore(sess, checkpoint_path)
        print('Checkpoint found, skipping training.')
    if (restore_checkpoint and tf.train.checkpoint_exists(checkpoint_path) and continue_training_from_checkpoint) \
            or not restore_checkpoint or not tf.train.checkpoint_exists(checkpoint_path):

        init.run()

        for epoch in range(1,1+n_epochs):
            for iteration in range(1, 1+n_iterations_per_epoch):

                # get data in the batches
                offset = (iteration * batch_size) % (train_labels.shape[0] - batch_size)
                batch_data = train_set[offset:(offset + batch_size), :, :, :]
                batch_labels = train_labels[offset:(offset + batch_size)]

                # Run the training operation and measure the loss:
                _, loss_train, summ = sess.run(
                    [do_training, capser["loss"], summary],
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
                        [capser["loss"], capser["accuracy"]],
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
                epoch, acc_val * 100, loss_val,
                " (improved)" if loss_val < best_loss_val else ""))

        # save the model at the end
        save_path = saver.save(sess, checkpoint_path)
        best_loss_val = loss_val


########################################################################################################################
# Embedding
########################################################################################################################


config = tf.ConfigProto(device_count = {'GPU': 0})
with tf.Session(config=config) as sess:
    print(' ... final iteration: Creating embedding')
    saver.restore(sess, checkpoint_path)
    sess.run(assignment, feed_dict={X: test_set,
        	                        y: test_labels,
    	                            mask_with_labels: False})
    saver.save(sess, checkpoint_path)


########################################################################################################################
# Testing
########################################################################################################################


do_testing = 0

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
                    [capser["loss"], capser["accuracy"]],
                    feed_dict={X: batch_data,
                                   y: batch_labels,
                                   mask_with_labels: False})
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
# View predictions
########################################################################################################################


# Now let's make some predictions! We first fix a few images from the test set, then we start a session,
# restore the trained model, evaluate caps2_output to get the capsule network's output vectors, decoder_output
# to get the reconstructions, and y_pred to get the class predictions.
n_samples = 16

sample_images = test_set[:n_samples,:,:]

with tf.Session() as sess:
    saver.restore(sess, checkpoint_path)
    caps2_output_value, decoder_output_value, y_pred_value = sess.run(
            [capser["caps2_output"], capser["decoder_output"], capser["y_pred"]],
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

plt.savefig(image_output_dir+'/sample images')
#plt.show()

plt.figure(figsize=(n_samples / 1.5, n_samples / 1.5))
for index in range(n_samples):
    plt.subplot(5, 5, index + 1)
    plt.title("Predicted:" + str(y_pred_value[index]))
    plt.imshow(reconstructions[index], cmap="binary")
    plt.axis("off")

plt.savefig(image_output_dir+'/sample images reconstructed')
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
        capser["decoder_output"],
            feed_dict={capser["caps2_output"]: tweaked_vectors_reshaped,
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
    plt.savefig(image_output_dir + '/tweak dimension' + str(dim))