import tensorflow as tf
from capser_model import capser_model
from batchMaker import StimMaker
import numpy as np
import matplotlib.pyplot as plt
import os
from create_sprite import images_to_sprite, invert_grayscale
from capsule_functions import vernier_classifier, vernier_x_entropy, vernier_correct_mean, safe_norm, run_test_stimuli

########################################################################################################################
# Parameters
########################################################################################################################


# data parameters
fixed_stim_position = None  # put top left corner of all stimuli at fixed_position
normalize_images = False    # make each image mean=0, std=1
max_rows, max_cols = 1, 5   # max number of rows, columns of shape grids
vernier_grids = False       # if true, verniers come in grids like other shapes. Only single verniers otherwise.
im_size = (30, 60)         # IF USING THE DECONVOLUTION DECODER NEED TO BE EVEN NUMBERS (NB. this suddenly changed. before that, odd was needed... that's odd.)
shape_size = 10             # size of a single shape in pixels
simultaneous_shapes = 2     # number of different shapes in an image. NOTE: more than 2 is not supported at the moment
bar_width = 1              # thickness of elements' bars
noise_level = 0  # 10       # add noise
shape_types = [0, 1, 2, 6]  # see batchMaker.drawShape for number-shape correspondences
group_last_shapes = 1       # attributes the same label to the last n shapeTypes
label_to_shape = {0: 'vernier', 1: 'squares', 2:'circles', 3:'stars'}
shape_to_label = dict( [ [v, k] for k, v in label_to_shape.items() ] )

stim_maker = StimMaker(im_size, shape_size, bar_width)  # handles data generation
# test_stimuli = {'squares':       [None, [[1]], [[1, 1, 1]]],
#                 'circles':       [None, [[2]], [[2, 2, 2]]]}
test_stimuli = {'squares':       [None, [[1]], [[1, 1, 1, 1, 1]]],
                'circles':       [None, [[2]], [[2, 2, 2, 2, 2]]],
                '7stars':        [None, [[6]], [[6, 6, 6, 6, 6]]],
                'irreg':         [None, [[7]], [[7, 7, 7, 7, 7]]],
                'squares_stars': [None, [[1]], [[1, 6, 1, 6, 1]]]}
# test_stimuli = {'squares':       [None, [[1]], [[1, 1, 1, 1, 1, 1, 1]]],
#                 'circles':       [None, [[2]], [[2, 2, 2, 2, 2, 2, 2]]],
#                 '7stars':        [None, [[6]], [[6, 6, 6, 6, 6, 6, 6]]],
#                 'irreg':         [None, [[7]], [[7, 7, 7, 7, 7, 7, 7]]],
#                 'squares_stars': [None, [[1]], [[6, 1, 6, 1, 6, 1, 6]]]}  #,
                # 'config':        [None, [[1]], [[6, 1, 6, 1, 6, 1, 6],
                #                                 [6, 1, 6, 1, 6, 1, 6],
                #                                 [6, 1, 6, 1, 6, 1, 6]]]}

# training parameters
n_batches = 1000000
batch_size = 6
conv_batch_norm = False
decoder_batch_norm = False
train_new_vernier_decoder = True  # to use a "fresh" new decoder for the vernier testing.
plot_uncrowding_during_training = False  # to plot uncrowding results while training
vernier_label_encoding = 'nothinglr_012'  # 'lr_10' or 'nothinglr_012'
if simultaneous_shapes > 1:
    vernier_label_encoding = 'nothinglr_012'  # needs to be nothinglr_012 if we use simultaneous shapes

# saving/loading parameters
restore_checkpoint = True
version_to_restore = None
continue_training_from_checkpoint = False

# conv layers
activation_function = tf.nn.relu
conv1_params = {"filters": 64,
                "kernel_size": 3,
                "strides": 1,
                "padding": "valid",
                "activation": activation_function,
                }
conv2_params = {"filters": 64,
                "kernel_size": 3,
                "strides": 1,
                "padding": "valid",
                "activation": activation_function,
                }
conv2_params = None
# conv3_params = {"filters": 32,
#                 "kernel_size": 5,
#                 "strides": 1,
#                 "padding": "valid",
#                 "activation": activation_function,
#                 }
conv3_params = None

# primary capsules
caps1_n_maps = len(label_to_shape)  # number of capsules at level 1 of capsules
caps1_n_dims = 8  # number of dimension per capsule
conv_caps_params = {"filters": caps1_n_maps * caps1_n_dims,
                    "kernel_size": 4,
                    "strides": 2,
                    "padding": "valid",
                    "activation": activation_function,
                    }

# output capsules
caps2_n_caps = len(label_to_shape)  # number of capsules
caps2_n_dims = 16                    # of n dimensions
rba_rounds = 3

# margin loss parameters
alpha_margin = 3.333
m_plus = .9
m_minus = .1
lambda_ = .5

# optional loss on a decoder trying to determine vernier orientation from the vernier output capsule
vernier_offset_loss = False
alpha_vernier_offset = 0


# optional loss requiring output capsules to give the number of shapes in the display
n_shapes_loss = False
if simultaneous_shapes > 1:  # you can't do the n_shapes loss with simultaneous shapes
    n_shapes_loss = False
alpha_n_shapes = 0

# optional loss to the primary capsules
primary_caps_loss = False
alpha_primary = 0
m_plus_primary = .9
m_minus_primary = .2
lambda_primary = .5

# choose reconstruction loss type and alpha
alpha_reconstruction = .0005
reconstruction_loss_type = 'squared_difference'  # 'squared_difference', 'sparse', 'rescale', 'threshold' or 'plot_all'
vernier_gain = 1

# decoder layer sizes
primary_caps_decoder = False
primary_caps_decoder_n_hidden1 = 256
primary_caps_decoder_n_hidden2 = 512
primary_caps_decoder_n_hidden3 = None
primary_caps_decoder_n_output = shape_size**2

output_caps_decoder_n_hidden1 = 512
output_caps_decoder_n_hidden2 = 1024
output_caps_decoder_n_hidden3 = None
output_caps_decoder_n_output = im_size[0] * im_size[1]
output_decoder_deconv_params = {'use_deconvolution_decoder': False,
                                'fc_width': (im_size[1]+2-shape_size)//2,
                                'fc_height': (im_size[0]+2-shape_size)//2,
                                'deconv_filters2': len(label_to_shape)+1,  # the +1 is for two vernier offsets
                                'deconv_kernel2': shape_size,
                                'deconv_strides2': 2,
                                'final_fc': True
                                }


####################################################################################################################
# Reproducibility & directories
####################################################################################################################


# directories
MODEL_NAME = 'capser_6'
LOGDIR = './' + MODEL_NAME + '_logdir/'  # will be redefined below
if not os.path.exists(LOGDIR):
    os.makedirs(LOGDIR)

# find what this version should be named
if version_to_restore is None:
    version = 0
    last_version = -1
    for file in os.listdir(LOGDIR):
        if 'version_' in file:
            version_number = int(file[-1])
            if version_number > last_version:
                last_version = version_number
    version = last_version + 1
else:
    version = version_to_restore

print('#################### MODEL_NAME_VERSION: ' + MODEL_NAME+'_'+str(version) + ' ####################')

LOGDIR = './' + MODEL_NAME + '_logdir/version_' + str(version)
if not os.path.exists(LOGDIR):
    os.makedirs(LOGDIR)

image_output_dir = LOGDIR + '/output_images/'
if not os.path.exists(image_output_dir):
    os.makedirs(image_output_dir)
# path for saving the network
checkpoint_path = LOGDIR+'/'+MODEL_NAME+'_'+str(version)+"_model.ckpt"

# create a file summarizing the parameters if they are a new version
if version_to_restore is None:
    with open(LOGDIR+'/'+MODEL_NAME+'_'+str(version)+'_parameters.txt', 'w') as f:
        f.write("Parameter : value\n \n")
        variables = locals()
        variables = {key: value for key, value in variables.items()
                     if ('__' not in str(key)) and ('variable' not in str(key)) and ('module' not in str(value))
                     and ('function' not in str(value) and ('TextIOWrapper' not in str(value)))}
        [f.write(str(key)+' : '+str(value)+'\n') for key, value in variables.items()]
        print('Parameter values saved.')

# reproducibility
tf.reset_default_graph()
np.random.seed(42)
tf.set_random_seed(42)

do_all = 0
if do_all:
    if simultaneous_shapes > 1:
        do_embedding, plot_final_norms, do_output_images, do_color_capsules, do_vernier_decoding = 1, 1, 0, 1, 1
    else:
        do_embedding, plot_final_norms, do_output_images, do_color_capsules, do_vernier_decoding = 1, 1, 1, 1, 1
else:
    do_embedding = 0
    plot_final_norms = 1
    do_output_images = 0
    do_color_capsules = 1
    do_vernier_decoding = 1

########################################################################################################################
# Data handling
########################################################################################################################

# create sample dataset (we need it to create adequately sized networks)
if simultaneous_shapes > 1:
    batch_data, batch_single_shape_images, batch_labels, vernier_offset_labels, n_elements = stim_maker.makeMultiShapeBatch(batch_size, shape_types, n_shapes=simultaneous_shapes, noiseLevel=noise_level, group_last_shapes=group_last_shapes, vernierLabelEncoding=vernier_label_encoding, max_rows=max_rows, max_cols=max_cols, vernier_grids=vernier_grids, normalize=normalize_images, fixed_position=fixed_stim_position)
else:
    batch_data, batch_labels, vernier_offset_labels, n_elements = stim_maker.makeBatch(batch_size, shape_types, noise_level, group_last_shapes, vernierLabelEncoding=vernier_label_encoding, max_rows=max_rows, max_cols=max_cols, vernier_grids=vernier_grids, normalize=normalize_images, fixed_position=fixed_stim_position)

# placeholders for input images and labels, and optional stuff
X = tf.placeholder(shape=[None, im_size[0], im_size[1], 1], dtype=tf.float32, name="X")
x_image = tf.reshape(X, [-1, im_size[0], im_size[1], 1])
tf.summary.image('input', x_image, 6)
reconstruction_targets = tf.placeholder(shape=[None, im_size[0], im_size[1], simultaneous_shapes], dtype=tf.float32, name="reconstruction_targets")
if simultaneous_shapes > 1:
    y = tf.placeholder(shape=[None, simultaneous_shapes], dtype=tf.int64, name="y")
    n_shapes = tf.placeholder_with_default(tf.zeros(shape=(batch_size, simultaneous_shapes)), shape=[None, simultaneous_shapes], name="n_shapes_labels")
else:
    y = tf.placeholder(shape=[None], dtype=tf.int64, name="y")
    n_shapes = tf.placeholder_with_default(tf.zeros(shape=(batch_size)), shape=[None], name="n_shapes_labels")
vernier_offsets = tf.placeholder_with_default(tf.zeros(shape=(batch_size)), shape=[None], name="vernier_offset_labels")

# create a placeholder that will tell the program whether to use the true or the predicted labels
mask_with_labels = tf.placeholder_with_default(False, shape=(), name="mask_with_labels")
# placeholder specifying if training or not (for batch normalization)
is_training = tf.placeholder_with_default(False, shape=(), name='is_training')

show_samples = 0
if show_samples:

    n_samples = 5

    plt.figure(figsize=(n_samples * 2, 3))
    for index in range(n_samples):
        plt.subplot(1, n_samples, index + 1)
        sample_image = batch_data[index, :, :].reshape(im_size[0], im_size[1])
        plt.imshow(sample_image, cmap="binary")
        plt.axis("off")
    plt.show()


########################################################################################################################
# Create Networks
########################################################################################################################

capser = capser_model(X, y, reconstruction_targets, im_size, conv1_params, conv2_params, conv3_params,
                      caps1_n_maps, caps1_n_dims, conv_caps_params,
                      primary_caps_decoder_n_hidden1, primary_caps_decoder_n_hidden2, primary_caps_decoder_n_hidden3, primary_caps_decoder_n_output,
                      caps2_n_caps, caps2_n_dims, rba_rounds,
                      m_plus, m_minus, lambda_, alpha_margin,
                      m_plus_primary, m_minus_primary, lambda_primary, alpha_primary,
                      output_caps_decoder_n_hidden1, output_caps_decoder_n_hidden2, output_caps_decoder_n_hidden3, reconstruction_loss_type, alpha_reconstruction, vernier_gain,
                      is_training, mask_with_labels,
                      primary_caps_decoder, primary_caps_loss, n_shapes_loss, vernier_offset_loss,
                      n_shapes, max_cols*max_rows, alpha_n_shapes,
                      vernier_offsets, alpha_vernier_offset,
                      0, conv_batch_norm, decoder_batch_norm,
                      **output_decoder_deconv_params)

# op to train all networks
do_training = capser["training_op"]

embedding_writer = tf.summary.FileWriter(LOGDIR)  # to write summaries


########################################################################################################################
# Create embedding of the secondary capsules output
########################################################################################################################


if do_embedding:

    with tf.device('/cpu:0'):

        samples_train = 20  # for train set embedding
        samples_per_condition = 4  # for test set embedding


        embedding_train_data, embedding_train_labels, embedding_train_vernier_labels, embedding_train_n_elements = stim_maker.makeBatch(samples_train, shape_types, noise_level, group_last_shapes, max_rows=max_rows, max_cols=max_cols, vernier_grids=vernier_grids, normalize=normalize_images, fixed_position=fixed_stim_position)
        # create sprites and embedding labels from test set for embedding visualization in tensorboard
        sprites_train = invert_grayscale(images_to_sprite(np.squeeze(embedding_train_data)))
        plt.imsave(LOGDIR+'/'+MODEL_NAME+'_'+str(version)+'_sprites_train.png', sprites_train, cmap='gray')

        with open(LOGDIR+'/'+MODEL_NAME+'_'+str(version)+'_embedding_train_labels.tsv', 'w') as f:
            f.write("Index\tLabel\n")
            for index, label in enumerate(embedding_train_labels):
                f.write("%d\t%d\n" % (index, label))

        show_sprites = 0
        if show_sprites:
            plt.imshow(sprites_train, cmap='gray')
            plt.show()

        LABELS_train = os.path.join(os.getcwd(), LOGDIR[2:]+'/'+MODEL_NAME+'_'+str(version)+'_embedding_train_labels.tsv')
        SPRITES_train = os.path.join(os.getcwd(), LOGDIR[2:]+'/'+MODEL_NAME+'_'+str(version)+'_sprites_train.png')
        embedding_input_train = tf.reshape(capser["caps2_output"], [-1, caps2_n_caps*caps2_n_dims])
        embedding_size_train = caps2_n_caps*caps2_n_dims
        embedding_train = tf.Variable(tf.zeros([embedding_train_data.shape[0], embedding_size_train]), name='final_capsules_embedding_train')
        assignment_train = embedding_train.assign(embedding_input_train)

        # an embedding using the testing images
        embedding_test_labels = np.zeros(shape=(samples_per_condition*(len(test_stimuli)*2+1)))
        embedding_test_data = np.zeros(shape=(len(embedding_test_labels), im_size[0], im_size[1], 1))
        this_cat = 0
        for category in test_stimuli.keys():
            stim_matrices = test_stimuli[category]
            for stim in range(1,3):
                embedding_test_data[(this_cat*2+(stim-1))*samples_per_condition:(this_cat*2+stim)*samples_per_condition,:,:,:], embedding_test_labels[(this_cat*2+(stim-1))*samples_per_condition:(this_cat*2+stim)*samples_per_condition] = stim_maker.makeConfigBatch(samples_per_condition, stim_matrices[stim], noise_level, normalize=normalize_images, fixed_position=fixed_stim_position)
            this_cat += 1
        embedding_test_data[-samples_per_condition:,:,:,:], embedding_test_labels[-samples_per_condition:] = stim_maker.makeConfigBatch(samples_per_condition, None, noise_level, normalize=normalize_images, fixed_position=fixed_stim_position)

        # create sprites and embedding labels from test set for embedding visualization in tensorboard
        sprites_test = invert_grayscale(images_to_sprite(np.squeeze(embedding_test_data)))
        plt.imsave(LOGDIR + '/' + MODEL_NAME + '_' + str(version) + '_sprites_test.png', sprites_test, cmap='gray')

        with open(LOGDIR + '/' + MODEL_NAME + '_' + str(version) + '_embedding_test_labels.tsv', 'w') as f:
            f.write("Index\tLabel\n")
            for index, label in enumerate(embedding_test_labels):
                f.write("%d\t%d\n" % (index, label))

        show_sprites = 0
        if show_sprites:
            plt.imshow(sprites_test, cmap='gray')
            plt.show()

        LABELS_test = os.path.join(os.getcwd(), LOGDIR[2:] + '/' + MODEL_NAME + '_' + str(version) + '_embedding_test_labels.tsv')
        SPRITES_test = os.path.join(os.getcwd(), LOGDIR[2:] + '/' + MODEL_NAME + '_' + str(version) + '_sprites_test.png')
        embedding_input_test = tf.reshape(capser["caps2_output"], [-1, caps2_n_caps * caps2_n_dims])
        embedding_size_test = caps2_n_caps * caps2_n_dims
        embedding_test = tf.Variable(tf.zeros([embedding_test_data.shape[0], embedding_size_test]), name='final_capsules_embedding_test')
        assignment_test = embedding_test.assign(embedding_input_test)

        # configure embedding visualizer
        # training set embedding
        config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
        embedding_config_train = config.embeddings.add()
        embedding_config_train.tensor_name = embedding_train.name
        embedding_config_train.sprite.image_path = SPRITES_train
        embedding_config_train.metadata_path = LABELS_train
        embedding_config_train.sprite.single_image_dim.extend([max(im_size), max(im_size)])
        # testing set embedding
        embedding_config_test = config.embeddings.add()
        embedding_config_test.tensor_name = embedding_test.name
        embedding_config_test.sprite.image_path = SPRITES_test
        embedding_config_test.metadata_path = LABELS_test
        embedding_config_test.sprite.single_image_dim.extend([max(im_size), max(im_size)])


########################################################################################################################
# Training
########################################################################################################################

saver = tf.train.Saver()
init = tf.global_variables_initializer()
summary = tf.summary.merge_all()
uncrowding_exp_summary_writer = tf.summary.FileWriter(LOGDIR + '/uncrowding_exp')
uncrowding_exp_summary = tf.Summary()

with tf.Session() as sess:

    writer = tf.summary.FileWriter(LOGDIR, sess.graph)

    if do_embedding:
        tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)

    if restore_checkpoint and tf.train.checkpoint_exists(checkpoint_path):
        saver.restore(sess, checkpoint_path)
        print('Capser checkpoint found.')
    else:
        init.run()

    if (restore_checkpoint and tf.train.checkpoint_exists(checkpoint_path) and continue_training_from_checkpoint) \
            or not restore_checkpoint or not tf.train.checkpoint_exists(checkpoint_path):

        print('Training')

        for batch in range(n_batches):

            # get new batch
            if simultaneous_shapes > 1:
                batch_data, batch_single_shape_images, batch_labels, vernier_offset_labels, n_elements = stim_maker.makeMultiShapeBatch(batch_size, shape_types, n_shapes=simultaneous_shapes, noiseLevel=noise_level, group_last_shapes=group_last_shapes, vernierLabelEncoding=vernier_label_encoding, max_rows=max_rows, max_cols=max_cols, vernier_grids=vernier_grids, normalize=normalize_images, fixed_position=fixed_stim_position)
                batch_reconstruction_targets = batch_single_shape_images
            else:
                batch_data, batch_labels, vernier_offset_labels, n_elements = stim_maker.makeBatch(batch_size, shape_types, noise_level, group_last_shapes, vernierLabelEncoding=vernier_label_encoding, max_rows=max_rows, max_cols=max_cols, vernier_grids=vernier_grids, normalize=normalize_images, fixed_position=fixed_stim_position)
                batch_reconstruction_targets = batch_data

            # Run the training operation and measure the loss:
            _, loss_train, summ = sess.run(
                [do_training, capser["loss"], summary],
                feed_dict={X: batch_data,
                           reconstruction_targets: batch_reconstruction_targets,
                           y: batch_labels,
                           n_shapes: n_elements,
                           vernier_offsets: vernier_offset_labels,
                           mask_with_labels: True,
                           is_training: True})

            print("\rBatch: {}/{} ({:.1f}%) Total loss: {:.5f}".format(batch, n_batches, batch * 100 / n_batches, loss_train), end="")

            if batch % 250 == 0:
                # printed = sess.run(capser["printed"], feed_dict={X: batch_data,
                #                reconstruction_targets: batch_reconstruction_targets,
                #                y: batch_labels,
                #                n_shapes: n_elements,
                #                vernier_offsets: vernier_offset_labels,
                #                mask_with_labels: True,
                #                is_training: True})

                writer.add_summary(summ, batch)

            if batch % 10000 == 0 and batch > 0 and plot_uncrowding_during_training:
                    run_test_stimuli(test_stimuli, 400, stim_maker, batch_size, noise_level, normalize_images, fixed_stim_position, simultaneous_shapes, capser, X, y, reconstruction_targets, vernier_offsets, mask_with_labels, sess, LOGDIR, label_encoding=vernier_label_encoding, summary_writer=uncrowding_exp_summary_writer, global_step=batch)  # create (un-)crowding plots to see evolution

            if batch == n_batches-1 or (batch % 50000 == 0 and batch > 0):
                if do_embedding:

                    # we will feed an empty y that will not be used, but needs to have the right shape (called y_serge)
                    if simultaneous_shapes == 1:
                        y_serge = np.zeros(shape=(len(embedding_train_labels)))
                    else:
                        y_serge = np.zeros(shape=(len(embedding_train_labels),2))

                    sess.run(assignment_train, feed_dict={X: embedding_train_data,
                                                          reconstruction_targets: batch_reconstruction_targets,
                                                          y: y_serge,
                                                          n_shapes: n_elements,
                                                          vernier_offsets: vernier_offset_labels,
                                                          mask_with_labels: False,
                                                          is_training: True})
                    sess.run(assignment_test, feed_dict={X: embedding_test_data,
                                                         reconstruction_targets: batch_reconstruction_targets,
                                                         y: y_serge,
                                                         n_shapes: n_elements,
                                                         vernier_offsets: vernier_offset_labels,
                                                         mask_with_labels: False,
                                                         is_training: True})

                saver.save(sess, checkpoint_path)

    else:
        print('Skipping training.')


########################################################################################################################
# Output norms of trained network
########################################################################################################################


if plot_final_norms:

    n_plots = batch_size
    res_path = image_output_dir + '/final_capsule_norms'
    if not os.path.exists(res_path):
        os.makedirs(res_path)

    # get norms
    with tf.Session() as sess:
        saver.restore(sess, checkpoint_path)

        # get new batch
        if simultaneous_shapes > 1:
            batch_data, batch_single_shape_images, batch_labels, vernier_offset_labels, n_elements = stim_maker.makeMultiShapeBatch(batch_size, shape_types, n_shapes=simultaneous_shapes, noiseLevel=noise_level, group_last_shapes=group_last_shapes, vernierLabelEncoding=vernier_label_encoding, max_rows=max_rows, max_cols=max_cols, vernier_grids=vernier_grids, normalize=normalize_images, fixed_position=fixed_stim_position)
            batch_reconstruction_targets = batch_single_shape_images
        else:
            batch_data, batch_labels, vernier_offset_labels, n_elements = stim_maker.makeBatch(batch_size, shape_types, noise_level, group_last_shapes, vernierLabelEncoding=vernier_label_encoding, max_rows=max_rows, max_cols=max_cols, vernier_grids=vernier_grids, normalize=normalize_images, fixed_position=fixed_stim_position)
            batch_reconstruction_targets = batch_data

        caps2_output_final, predictions = sess.run([capser["caps2_output"], capser["y_pred"]],
                                                    feed_dict={X: batch_data[:n_plots, :, :, :],
                                                               y: batch_labels[:n_plots],
                                                               reconstruction_targets: batch_reconstruction_targets,
                                                               mask_with_labels: True,
                                                               is_training: True})

        x_labels = [label_to_shape[key] for key in label_to_shape]

        ####### PLOT RESULTS #######
        for i in range(n_plots):
            caps_output_norm = tf.squeeze(safe_norm(caps2_output_final[i, :, :, :], axis=-2, keep_dims=False,
                                                    name="caps2_output_norm")).eval(feed_dict={X: batch_data[:n_plots, :, :, :],
                                                               y: batch_labels[:n_plots],
                                                               reconstruction_targets: batch_reconstruction_targets,
                                                               mask_with_labels: True,
                                                               is_training: True})
            ind = np.arange(len(x_labels))  # the x locations for the groups
            width = 0.25  # the width of the bars

            fig, ax = plt.subplots()
            plot_color = (0. / 255, 91. / 255, 150. / 255)
            rects1 = ax.bar(ind, caps_output_norm, width, color=plot_color)

            # add some text for labels, title and axes ticks, and save figure
            ax.set_ylabel('Capsule norms')
            ax.set_title('Predicted : ' + str(predictions[i]) + ', true label : ' + str(batch_labels[i]))
            ax.set_xticks(ind + width / 2)
            ax.set_xticklabels(x_labels)
            plt.savefig(res_path + '/stimulus_' + str(i) + '.png')
            plt.close()

########################################################################################################################
# View predictions
########################################################################################################################

if do_output_images:
    # Now let's make some predictions! We first fix a few images from the test set, then we start a session,
    # restore the trained model, evaluate caps2_output to get the capsule network's output vectors, decoder_output
    # to get the reconstructions, and y_pred to get the class predictions.

    res_path = image_output_dir + '/reconstructions'
    if not os.path.exists(res_path):
        os.makedirs(res_path)

    n_samples = 5

    # get new batch
    if simultaneous_shapes > 1:
        batch_data, batch_single_shape_images, batch_labels, vernier_offset_labels, n_elements = stim_maker.makeMultiShapeBatch(batch_size, shape_types, n_shapes=simultaneous_shapes, noiseLevel=noise_level, group_last_shapes=group_last_shapes, vernierLabelEncoding=vernier_label_encoding, max_rows=max_rows, max_cols=max_cols, vernier_grids=vernier_grids, normalize=normalize_images, fixed_position=fixed_stim_position)
        batch_reconstruction_targets = batch_single_shape_images
    else:
        batch_data, batch_labels, vernier_offset_labels, n_elements = stim_maker.makeBatch(batch_size, shape_types, noise_level, group_last_shapes, vernierLabelEncoding=vernier_label_encoding, max_rows=max_rows, max_cols=max_cols, vernier_grids=vernier_grids, normalize=normalize_images,fixed_position=fixed_stim_position)
        batch_reconstruction_targets = batch_data
    sample_images = batch_data[:n_samples, :, :]

    with tf.Session() as sess:
        saver.restore(sess, checkpoint_path)
        caps2_output_value, decoder_output_value, y_pred_value = sess.run(
                [capser["caps2_output"], capser["decoder_output_output_caps"], capser["y_pred"]],
                feed_dict={X: sample_images,
                           reconstruction_targets: batch_reconstruction_targets,
                           y: batch_labels,
                           is_training: True})

    # plot images with their reconstruction
    sample_images = sample_images.reshape(-1, im_size[0], im_size[1])
    reconstructions = decoder_output_value.reshape([-1, im_size[0], im_size[1]])

    plt.figure(figsize=(n_samples / 1.5, n_samples / 1.5))
    for index in range(n_samples):
        plt.subplot(5, 5, index + 1)
        plt.imshow(sample_images[index], cmap="binary")
        plt.title("Label:" + str(batch_labels[index]))
        plt.axis("off")

    plt.savefig(res_path+'/sample images')
    plt.close()
    # plt.show()

    plt.figure(figsize=(n_samples / 1.5, n_samples / 1.5))
    for index in range(n_samples):
        plt.subplot(5, 5, index + 1)
        plt.title("Predicted:" + str(y_pred_value[index]))
        plt.imshow(reconstructions[index], cmap="binary")
        plt.axis("off")

    plt.savefig(res_path+'/sample images reconstructed')
    plt.close()
    # plt.show()

    ####################################################################################################################
    # Interpreting the output vectors
    ####################################################################################################################

    # caps2_output is now a numpy array. let's check its shape
    # print('shape of caps2_output np array: '+str(caps2_output_value.shape))

    res_path = image_output_dir + '/parameter_teaks'
    if not os.path.exists(res_path):
        os.makedirs(res_path)

    # Let's create a function that will tweak each of the n pose parameters (dimensions) in all output vectors.
    # Each tweaked output vector will be identical to the original output vector, except that one of its pose
    # parameters will be incremented by a value varying from -0.5 to 0.5. By default there will be 11 steps
    # (-0.5, -0.4, ..., +0.4, +0.5). This function will return an array of shape
    # (tweaked pose parameters=16, steps=11, batch size=5, 1, caps2_n_caps, caps2_n_dims, 1):
    def tweak_pose_parameters(output_vectors, mini=-0.5, maxi=0.5, num_steps=11):
        steps = np.linspace(mini, maxi, num_steps)   # -0.25, -0.15, ..., +0.25
        pose_parameters = np.arange(caps2_n_dims)  # 0, 1, ..., 15
        tweaks = np.zeros([caps2_n_dims, num_steps, 1, 1, 1, caps2_n_dims, 1])
        tweaks[pose_parameters, :, 0, 0, 0, pose_parameters, 0] = steps
        output_vectors_expanded = output_vectors[np.newaxis, np.newaxis]
        return tweaks + output_vectors_expanded

    # get a tweaked parameters array for the caps2_output_value array,
    # and reshape (i.e., flatten) it to feed to decoder.
    n_steps = 11
    tweaked_vectors = tweak_pose_parameters(caps2_output_value, num_steps=n_steps)
    tweaked_vectors_reshaped = tweaked_vectors.reshape([-1, 1, caps2_n_caps, caps2_n_dims, 1])

    # feed to decoder
    tweak_labels = np.tile(batch_labels[:n_samples], caps2_n_dims * n_steps)

    # get reconstruction
    with tf.Session() as sess:
        saver.restore(sess, checkpoint_path)
        decoder_output_value = sess.run(
            capser["decoder_output_output_caps"],
            feed_dict={capser["caps2_output"]: tweaked_vectors_reshaped,
                       reconstruction_targets: batch_reconstruction_targets,
                       mask_with_labels: True,
                       y: tweak_labels,
                       is_training: True})

    # reshape to make things easier to travel in dimension, n_steps and sample
    tweak_reconstructions = decoder_output_value.reshape([caps2_n_dims, n_steps, n_samples, im_size[0], im_size[1]])

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
        plt.savefig(res_path + '/tweak dimension' + str(dim))
        plt.close()


########################################################################################################################
# Testing segmentation on actual crowding stimuli
########################################################################################################################


if do_color_capsules:


    # old version
    if simultaneous_shapes == 1:

        show_grayscale = False  # you can choose to plot the decoder output for each capsule without colors
        n_trials = 8            # times we run each stimulus
        peak_intensity = 128    # limit the pixel intnsity to avoid saturation in the output image

        with tf.Session() as sess:

            # First restore the network
            saver.restore(sess, checkpoint_path)

            for category in test_stimuli.keys():

                print('Creating capsule visualizations for : ' + category)

                stim_matrices = test_stimuli[category]

                res_path = image_output_dir + '/' + category
                if not os.path.exists(res_path):
                    os.makedirs(res_path)

                for stim in range(3):

                    # get a few copies of the current stimulus
                    batch_data, vernier_labels = stim_maker.makeConfigBatch(n_trials, configMatrix=stim_matrices[stim], noiseLevel=noise_level, normalize=normalize_images, fixed_position=fixed_stim_position)
                    reconstruction_targets_serge = np.zeros(shape=(batch_size, im_size[0], im_size[1], simultaneous_shapes))

                    # choose which capsules to vizualize
                    caps_to_visualize = None
                    for key, value in shape_to_label.items():
                        if category is key:
                            caps_to_visualize = [0, value]
                    if caps_to_visualize is None:
                        print('CANNOT FIND WHICH CAPSULES TO DISPLAY FOR ' + category + ': DISPLAYING ALL CAPSULES.')
                        caps_to_visualize = range(caps2_n_caps)  # to see where the irreg shapes end up
                    n_caps_to_visualize = len(caps_to_visualize)

                    color_masks = np.array([[121, 199, 83],   # 0: vernier, green
                                            [220, 76, 70],    # 1: red
                                            [196, 143, 101],  # 2: beige
                                            [79, 132, 196],   # 3: blue
                                            [246, 209, 85],   # 4: yellow
                                            [237, 205, 194],  # 5: pale pink
                                            [181, 101, 167],  # 6: purple
                                            [210, 105, 30]])  # 7: orange

                    decoder_outputs_all = np.zeros(shape=(n_trials, im_size[0]*im_size[1], 3, n_caps_to_visualize))
                    decoder_outputs_overlay = np.zeros(shape=(n_trials, im_size[0]*im_size[1], 3))

                    done_caps = 0
                    for cap in caps_to_visualize:
                        for rgb in range(3):

                            cap_labels = np.ones(n_trials)*cap
                            if simultaneous_shapes > 1:  # we need to have a [batch_size, simultaneous_shapes] array as input for y. Columns tell us which capsule to decode from in a stimulus (rows). So we add a columns of zeros to lways decode from the vernier capsule. The masking function will do the right thing.
                                cap_labels = np.transpose(np.stack(cap_labels, cap_labels*0))
                                print('CAP_LABELS.SHAPE FOR COLOR CAPSULES: ' + str(cap_labels.shape))

                            this_decoder_output = capser["decoder_output_output_caps"].eval({X: batch_data,
                                                                                             reconstruction_targets: reconstruction_targets_serge,
                                                                                             y: cap_labels,  # decode from capsule of interest
                                                                                             mask_with_labels: True,
                                                                                             is_training: True})
                            this_decoder_output = np.divide(this_decoder_output, peak_intensity)
                            temp = np.multiply(this_decoder_output, color_masks[cap, rgb])
                            decoder_outputs_all[:, :, rgb, done_caps] = temp
                            decoder_outputs_overlay[:, :, rgb] += temp

                        if show_grayscale:
                            print(this_decoder_output.shape, batch_data.shape)
                            check_image = np.reshape(this_decoder_output[0, :], [im_size[0], im_size[1]])
                            plt.subplot(1, 2, 1)
                            plt.imshow(batch_data[0, :, :, 0], cmap="binary")
                            plt.subplot(1, 2, 2)
                            plt.imshow(check_image, cmap="binary")
                            plt.title('Left: stimulus, right: reconstruction from capsule ' + str(cap))
                            plt.show()

                        done_caps += 1

                    decoder_outputs_overlay[decoder_outputs_overlay > 255] = 255

                    decoder_images_all = np.reshape(decoder_outputs_all, [n_trials, im_size[0], im_size[1], 3, n_caps_to_visualize])
                    overlay_images = np.reshape(decoder_outputs_overlay, [n_trials, im_size[0], im_size[1], 3])

                    plt.figure(figsize=(n_trials / .5, n_trials / .5))
                    for im in range(n_trials):
                        plt.subplot(np.ceil(np.sqrt(n_trials)), np.ceil(np.sqrt(n_trials)), im+1)
                        plt.imshow(overlay_images[im, :, :, :])
                        plt.axis("off")
                    plt.savefig(res_path + '/capsule_overlay_image_' + category + '_' + str(stim))
                    plt.close()
                    # plt.show()

                    for im in range(n_trials):
                        plt.figure(figsize=(n_caps_to_visualize / .5, n_caps_to_visualize / .5))
                        plt.subplot(np.ceil(np.sqrt(n_caps_to_visualize)) + 1, np.ceil(np.sqrt(n_caps_to_visualize)), 1)
                        plt.imshow(batch_data[im, :, :, 0], cmap='gray')
                        plt.axis("off")
                        for cap in range(n_caps_to_visualize):
                            plt.subplot(np.ceil(np.sqrt(n_caps_to_visualize))+1, np.ceil(np.sqrt(n_caps_to_visualize)), cap + 2)
                            plt.imshow(decoder_images_all[im, :, :, :, cap])
                            plt.axis("off")
                        plt.savefig(res_path + '/all_caps_images_' + category + '_' + str(stim) + '_' + str(im))
                        plt.close()
                        # plt.show()
    else:

        n_trials = 6

        with tf.Session() as sess:

            # First restore the network
            saver.restore(sess, checkpoint_path)

            for category in test_stimuli.keys():

                CAT_LOGDIR = LOGDIR + '/test_' + category
                stim_matrices = test_stimuli[category]

                for stim in range(3):

                    THIS_LOGDIR = CAT_LOGDIR + '/' + str(stim)

                    this_writer = tf.summary.FileWriter(THIS_LOGDIR, sess.graph)

                    batch_data, vernier_labels = stim_maker.makeConfigBatch(n_trials, configMatrix=stim_matrices[stim], noiseLevel=noise_level, normalize=normalize_images, fixed_position=fixed_stim_position)
                    reconstruction_targets_serge = np.zeros(shape=(batch_size, im_size[0], im_size[1], simultaneous_shapes))
                    y_serge = np.zeros(shape=(batch_size, simultaneous_shapes))

                    summ = sess.run(summary, feed_dict={X: batch_data,
                                                        reconstruction_targets: reconstruction_targets_serge,
                                                        y: y_serge,
                                                        mask_with_labels: False})
                    this_writer.add_summary(summ)


########################################################################################################################
# Determine performance by training a decoder to identify vernier orientation based on the vernier capsule activity
########################################################################################################################

if do_vernier_decoding:

    n_stimuli = 1000

    res_path = image_output_dir + '/uncrowding_plots'
    if not os.path.exists(res_path):
        os.makedirs(res_path)

    if train_new_vernier_decoder is False:

        with tf.Session() as sess:
            correct_responses = run_test_stimuli(test_stimuli, 1000, stim_maker, batch_size, noise_level, normalize_images, fixed_stim_position, simultaneous_shapes, capser, X, y, reconstruction_targets, vernier_offsets, mask_with_labels, sess, LOGDIR, label_encoding=vernier_label_encoding, res_path=res_path, plot_ID='FINAL_TEST', saver=saver, checkpoint_path=checkpoint_path)

    else:

        decode_capsule = 0
        batch_size = batch_size
        n_batches = 10000
        n_hidden1 = None
        n_hidden2 = None
        vernier_batch_norm = False
        vernier_dropout = False
        decode_from_reconstruction = False

        LOGDIR = LOGDIR + '/vernier_decoder'
        vernier_checkpoint_path = LOGDIR+'/'+MODEL_NAME+'_'+str(version)+"vernier_decoder_model.ckpt"
        vernier_restore_checkpoint = False
        vernier_continue_training_from_checkpoint = False


        with tf.variable_scope('decode_vernier'):

            if decode_from_reconstruction:
                decoder_output = capser["decoder_output_output_caps"]  # decode from reconstruction
                vernier_decoder_input = decoder_output
            else:
                caps2_output = capser["caps2_output"]  # decode from capsule
                vernier_decoder_input = caps2_output[:, :, decode_capsule, :, :]

            classifier = vernier_classifier(vernier_decoder_input, True, n_hidden1=n_hidden1, n_hidden2=n_hidden2, batch_norm=vernier_batch_norm, dropout=vernier_dropout, name='vernier_decoder')
            if simultaneous_shapes > 1:
                x_entropy = vernier_x_entropy(classifier, y[:, 0])  # the [:, 0] is just in case we have many columns due to more than one simultaneous_shapes
                correct_mean = vernier_correct_mean(tf.argmax(classifier, axis=1), y[:, 0])
            else:
                x_entropy = vernier_x_entropy(classifier, y)  # the [:, 0] is just in case we have many columns due to more than one simultaneous_shapes
                correct_mean = vernier_correct_mean(tf.argmax(classifier, axis=1), y)
            update_batch_norm_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            train_op = tf.train.AdamOptimizer().minimize(x_entropy, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decode_vernier'), name="training_op")

        vernier_init = tf.variables_initializer(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decode_vernier'), name='vernier_init')
        summary = tf.summary.merge_all()
        master_training_op = [train_op, update_batch_norm_ops]
        vernier_saver = tf.train.Saver()

        ####################################################################################################################
        # Train decoder on different verniers
        ####################################################################################################################

        with tf.Session() as sess:

            # First restore the network
            saver.restore(sess, checkpoint_path)
            print('Training a new vernier decoder...')
            writer = tf.summary.FileWriter(LOGDIR, sess.graph)

            if vernier_restore_checkpoint and tf.train.checkpoint_exists(vernier_checkpoint_path):
                print('Vernier decoder checkpoint found, will not bother training.')
                vernier_saver.restore(sess, vernier_checkpoint_path)
            if (vernier_restore_checkpoint and tf.train.checkpoint_exists(
                    vernier_checkpoint_path) and vernier_continue_training_from_checkpoint) \
                    or not vernier_restore_checkpoint or not tf.train.checkpoint_exists(vernier_checkpoint_path):

                vernier_init.run()

                for batch in range(n_batches):

                    # get a vernier batch
                    vernier_data, vernier_labels = stim_maker.makeVernierBatch(batch_size, noise_level, normalize=normalize_images, fixed_position=fixed_stim_position)
                    reconstruction_targets_serge = np.zeros(shape=(batch_size, im_size[0], im_size[1], simultaneous_shapes))

                    if simultaneous_shapes > 1:  # we need to have a [batch_size, simultaneous_shapes] array as input for y. We copy the labels in each column. The masking function will do the right thing
                        vernier_labels = np.transpose(np.tile(vernier_labels, [2, 1]))

                    if batch % 250 == 0:

                        # Run the training operation, measure the losses and write summary:
                        _, summ = sess.run(
                            [master_training_op, summary],
                            feed_dict={X: vernier_data,
                                       reconstruction_targets: reconstruction_targets_serge,
                                       y: vernier_labels,
                                       mask_with_labels: False,
                                       is_training: True})
                        writer.add_summary(summ, batch)

                    else:

                        # Run the training operation and measure the losses:
                        _ = sess.run(master_training_op,
                                     feed_dict={X: vernier_data,
                                                y: vernier_labels,
                                                mask_with_labels: False,
                                                is_training: True})

                    print("\rbatch: {}/{} ({:.1f}%)".format(
                        batch, n_batches,
                        batch * 100 / n_batches),
                        end="")

                # save the model at the end
                vernier_save_path = vernier_saver.save(sess, vernier_checkpoint_path)

        ####################################################################################################################
        # Run trained decoder on actual stimuli
        ####################################################################################################################

        for category in test_stimuli.keys():

            print('Decoding vernier orientation for : '  + category)

            stim_matrices = test_stimuli[category]

            with tf.Session() as sess:

                vernier_saver.restore(sess, vernier_checkpoint_path)

                # we will collect correct responses here
                correct_responses = np.zeros(shape=(3))

                for this_stim in range(3):

                    n_batches = n_stimuli//batch_size

                    for batch in range(n_batches):


                        curr_stim = stim_matrices[this_stim]

                        # get a batch of the current stimulus
                        batch_data, vernier_labels = stim_maker.makeConfigBatch(batch_size, curr_stim, noise_level, normalize=normalize_images, fixed_position=fixed_stim_position)
                        if simultaneous_shapes > 1:  # we need to have a [batch_size, simultaneous_shapes] array as input for y. We copy the labels in each column. The masking function will do the right thing
                            vernier_labels = np.transpose(np.tile(vernier_labels, [2, 1]))
                            reconstruction_targets_serge = np.zeros(shape=(batch_size, im_size[0], im_size[1], simultaneous_shapes))

                        # Run the training operation and measure the losses:
                        correct_in_this_batch_all = sess.run(correct_mean,
                                                             feed_dict={X: batch_data,
                                                                        reconstruction_targets: reconstruction_targets_serge,
                                                                        y: vernier_labels,
                                                                        mask_with_labels: False,
                                                                        is_training: True})

                        correct_responses[this_stim] += np.array(correct_in_this_batch_all)

                percent_correct = correct_responses*100/n_batches
                print('... testing done.')
                print('Percent correct for vernier decoders with stimuli: ' + category)
                print(percent_correct)
                print('Writing data and plot')
                np.save(LOGDIR+'/'+category+'_percent_correct', percent_correct)

                # PLOT
                x_labels = ['vernier', 'crowded', 'uncrowded']

                ####### PLOT RESULTS #######

                N = len(x_labels)
                ind = np.arange(N)  # the x locations for the groups
                width = 0.25  # the width of the bars

                fig, ax = plt.subplots()
                plot_color = (0./255, 91./255, 150./255)
                rects1 = ax.bar(ind, percent_correct, width, color=plot_color)

                # add some text for labels, title and axes ticks, and save figure
                ax.set_ylabel('Percent correct')
                # ax.set_title('Vernier decoding from alexnet layers')
                ax.set_xticks(ind + width / 2)
                ax.set_xticklabels(x_labels)
                ax.plot([-0.3, N], [50, 50], 'k--')  # chance level cashed line
                ax.legend(rects1, ('vernier', '1 ' + category[:-1], '7 ' + category))
                plt.savefig(res_path + '/' + category + '_uncrowding_plot.png')
                plt.close()
