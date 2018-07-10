import tensorflow as tf
import sys, os
import numpy as np
from capser_model import capser_model
from make_tf_dataset import train_input_fn, test_input_fn

# first we create a dataset object with our data

with tf.device('/cpu:0'):

    data_path = './data'

    # reproducibility
    tf.reset_default_graph()
    np.random.seed(42)
    tf.set_random_seed(42)

    def model_fn(features, labels, mode, params):
        fixed_stim_position = None  # put top left corner of all stimuli at fixed_position
        normalize_images = False  # make each image mean=0, std=1
        max_rows, max_cols = 1, 5  # max number of rows, columns of shape grids
        vernier_grids = False  # if true, verniers come in grids like other shapes. Only single verniers otherwise.
        im_size = (30, 60)  # IF USING THE DECONVOLUTION DECODER NEED TO BE EVEN NUMBERS (NB. this suddenly changed. before that, odd was needed... that's odd.)
        shape_size = 10  # size of a single shape in pixels
        simultaneous_shapes = 2  # number of different shapes in an image. NOTE: more than 2 is not supported at the moment
        bar_width = 1  # thickness of elements' bars
        noise_level = 0  # 10       # add noise
        shape_types = [0, 1, 2, 9]  # see batchMaker.drawShape for number-shape correspondences
        group_last_shapes = 1  # attributes the same label to the last n shapeTypes
        label_to_shape = {0: 'vernier', 1: 'squares', 2: 'circles', 3: 'stuff'}
        shape_to_label = dict([[v, k] for k, v in label_to_shape.items()])

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
        conv1_params = {"filters": 16,
                        "kernel_size": 5,
                        "strides": 1,
                        "padding": "valid",
                        "activation": activation_function,
                        }
        conv2_params = {"filters": 16,
                        "kernel_size": 5,
                        "strides": 1,
                        "padding": "valid",
                        "activation": activation_function,
                        }
        # conv2_params = None
        # conv3_params = {"filters": 32,
        #                 "kernel_size": 5,
        #                 "strides": 1,
        #                 "padding": "valid",
        #                 "activation": activation_function,
        #                 }
        conv3_params = None

        # primary capsules
        caps1_n_maps = 16  # number of capsules at level 1 of capsules
        caps1_n_dims = 8  # number of dimension per capsule
        conv_caps_params = {"filters": caps1_n_maps * caps1_n_dims,
                            "kernel_size": 5,
                            "strides": 2,
                            "padding": "valid",
                            "activation": activation_function,
                            }

        # output capsules
        caps2_n_caps = len(label_to_shape)  # number of capsules
        caps2_n_dims = 16  # of n dimensions
        rba_rounds = 3

        # margin loss parameters
        alpha_margin = 3.333
        m_plus = .9
        m_minus = .1
        lambda_ = .5

        # optional loss on a decoder trying to determine vernier orientation from the vernier output capsule
        vernier_offset_loss = False
        alpha_vernier_offset = 1

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
        primary_caps_decoder_n_output = shape_size ** 2

        output_caps_decoder_n_hidden1 = 512
        output_caps_decoder_n_hidden2 = 1024
        output_caps_decoder_n_hidden3 = None
        output_caps_decoder_n_output = im_size[0] * im_size[1]
        output_decoder_deconv_params = {'use_deconvolution_decoder': False,
                                        'fc_width': (im_size[1] + 2 - shape_size) // 2,
                                        'fc_height': (im_size[0] + 2 - shape_size) // 2,
                                        'deconv_filters2': len(label_to_shape) + 1,  # the +1 is for two vernier offsets
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

        print('#################### MODEL_NAME_VERSION: ' + MODEL_NAME + '_' + str(version) + ' ####################')

        LOGDIR = './' + MODEL_NAME + '_logdir/version_' + str(version)
        if not os.path.exists(LOGDIR):
            os.makedirs(LOGDIR)

        image_output_dir = LOGDIR + '/output_images/'
        if not os.path.exists(image_output_dir):
            os.makedirs(image_output_dir)
        # path for saving the network
        checkpoint_path = LOGDIR + '/' + MODEL_NAME + '_' + str(version) + "_model.ckpt"

        # create a file summarizing the parameters if they are a new version
        if version_to_restore is None:
            with open(LOGDIR + '/' + MODEL_NAME + '_' + str(version) + '_parameters.txt', 'w') as f:
                f.write("Parameter : value\n \n")
                variables = locals()
                variables = {key: value for key, value in variables.items()
                             if ('__' not in str(key)) and ('variable' not in str(key)) and ('module' not in str(value))
                             and ('function' not in str(value) and ('TextIOWrapper' not in str(value)))}
                [f.write(str(key) + ' : ' + str(value) + '\n') for key, value in variables.items()]
                print('Parameter values saved.')

        do_all = 1
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

        # placeholders for input images and labels, and optional stuff
        # X = tf.placeholder(shape=[None, im_size[0], im_size[1], 1], dtype=tf.float32, name="X")
        X = features['X']
        x_image = tf.reshape(X, [-1, im_size[0], im_size[1], 1])
        tf.summary.image('input', x_image, 6)
        reconstruction_targets = features['reconstruction_targets']
        # reconstruction_targets = tf.placeholder(shape=[None, im_size[0], im_size[1], simultaneous_shapes], dtype=tf.float32,
        #                                         name="reconstruction_targets")
        if simultaneous_shapes > 1:
            y = tf.cast(features['y'], tf.int64)
            n_shapes = features['n_shapes']
            # y = tf.placeholder(shape=[None, simultaneous_shapes], dtype=tf.int64, name="y")
            # n_shapes = tf.placeholder_with_default(tf.zeros(shape=(params['batch_size'], simultaneous_shapes)),
            #                                        shape=[None, simultaneous_shapes], name="n_shapes_labels")
        else:
            # y = tf.placeholder(shape=[None], dtype=tf.int64, name="y")
            y = tf.cast(features['y'], tf.int64)
            n_shapes = tf.placeholder_with_default(tf.zeros(shape=(params['batch_size'])), shape=[None], name="n_shapes_labels")
        vernier_offsets = tf.placeholder_with_default(tf.zeros(shape=(params['batch_size'])), shape=[None],
                                                      name="vernier_offset_labels")

        # create a placeholder that will tell the program whether to use the true or the predicted labels
        mask_with_labels = tf.placeholder_with_default(features['mask_with_labels'], shape=(), name="mask_with_labels")
        # placeholder specifying if training or not (for batch normalization)
        is_training = tf.placeholder_with_default(features['is_training'], shape=(), name='is_training')
        is_training = features['is_training']

        capser = capser_model(X, y, reconstruction_targets, im_size, conv1_params, conv2_params, conv3_params,
                              caps1_n_maps, caps1_n_dims, conv_caps_params,
                              primary_caps_decoder_n_hidden1, primary_caps_decoder_n_hidden2,
                              primary_caps_decoder_n_hidden3, primary_caps_decoder_n_output,
                              caps2_n_caps, caps2_n_dims, rba_rounds,
                              m_plus, m_minus, lambda_, alpha_margin,
                              m_plus_primary, m_minus_primary, lambda_primary, alpha_primary,
                              output_caps_decoder_n_hidden1, output_caps_decoder_n_hidden2, output_caps_decoder_n_hidden3,
                              reconstruction_loss_type, alpha_reconstruction, vernier_gain,
                              is_training, mask_with_labels,
                              primary_caps_decoder, primary_caps_loss, n_shapes_loss, vernier_offset_loss,
                              n_shapes, max_cols * max_rows, alpha_n_shapes,
                              vernier_offsets, alpha_vernier_offset,
                              0, conv_batch_norm, decoder_batch_norm,
                              **output_decoder_deconv_params)

        # op to train all networks
        train_op = capser["loss_training_op"]

        # Define the evaluation metrics,
        # in this case the classification accuracy.
        # metrics = \
        #     {
        #         "loss": capser["loss"],
        #         "accuracy": capser["accuracy"]
        #     }

        # Wrap all of this in an EstimatorSpec.
        spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=capser["loss"],
            train_op=train_op,
            eval_metric_ops={})

        return spec


    model = tf.estimator.Estimator(model_fn=model_fn,
                                   params={'batch_size': 16},
                                   model_dir="./checkpoints_tutorial17-2/")

    model.train(input_fn=train_input_fn, steps=20000)

