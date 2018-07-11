import sys
import numpy as np
import matplotlib.pyplot as plt
from batchMaker import StimMaker
from parameters import *

data_path = './data'
if not os.path.exists(data_path):
    os.mkdir(data_path)

# an instance of the StimMaker class we will use to create stimuli
stim_maker = StimMaker(im_size, shape_size, bar_width)  # handles data generation

# helper functions for later
def wrap_int64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def wrap_bytes(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def print_progress(count, total):
    # Percentage completion.
    pct_complete = float(count) / total

    # Status-message.
    # Note the \r which means the line should overwrite itself.
    msg = "\r- Progress: {0:.1%}".format(pct_complete)

    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()


# create the tfRecord file
def make_multi_shape_tfRecords(stim_maker, shape_types, n_samples, out_path):
    # Args:
    # stim_maker        an instance of the StimMaker class (see batchMaker script)
    # shape_types       a list describing which shapes to use (see batchMaker script).
    # n_samples         the total number of stimuli.
    # out_path          File-path for the TFRecords output file.


    print("\nConverting: " + out_path)

    # Number of images. Used when printing the progress.

    # Open a TFRecordWriter for the output-file.
    with tf.python_io.TFRecordWriter(out_path) as writer:

        # Create images one by one using stimMaker and save them
        for i in range(n_samples):
            # Print the percentage-progress.
            print_progress(count=i, total=n_samples - 1)

            multi_shape_img, single_shape_img, labels, vernier_label, n_elements = stim_maker.makeMultiShapeBatch(1, shape_types, n_shapes=simultaneous_shapes, noiseLevel=noise_level, group_last_shapes=group_last_shapes, vernierLabelEncoding=vernier_label_encoding, max_rows=max_rows, max_cols=max_cols, vernier_grids=vernier_grids, normalize=normalize_images, fixed_position=fixed_stim_position)

            # Convert the image to raw bytes.
            multi_shape_img_bytes = multi_shape_img.tostring()
            single_shape_img_bytes = single_shape_img.tostring()
            labels_bytes = labels.tostring()
            vernier_label_bytes = vernier_label.tostring()
            n_elements_bytes = n_elements.tostring()

            # Create a dict with the data we want to save in the
            # TFRecords file. You can add more relevant data here.
            data = \
                {
                    'multi_shape_img': wrap_bytes(multi_shape_img_bytes),
                    'single_shape_img': wrap_bytes(single_shape_img_bytes),
                    'labels': wrap_bytes(labels_bytes),
                    'vernier_label': wrap_bytes(vernier_label_bytes),
                    'n_elements': wrap_bytes(n_elements_bytes)
                }

            # Wrap the data as TensorFlow Features.
            feature = tf.train.Features(feature=data)

            # Wrap again as a TensorFlow Example.
            example = tf.train.Example(features=feature)

            # Serialize the data.
            serialized = example.SerializeToString()

            # Write the serialized data to the TFRecords file.
            writer.write(serialized)


def make_config_tfRecords(stim_maker, stim_matrices, n_samples, out_path):
    # Kind of stupid, but we must create tfRecords differently for config_batches because they don't return the same
    # things as multi_shape_batches (see batchMaker script)
    # Args:
    # stim_maker        an instance of the StimMaker class (see batchMaker script)
    # stim_matrices     list of config matrices. See batchMaker script
    # n_samples         the total number of stimuli.
    # out_path          File-path for the TFRecords output file.

    print("\nConverting: " + out_path)

    # Number of images. Used when printing the progress.

    # Open a TFRecordWriter for the output-file.
    with tf.python_io.TFRecordWriter(out_path) as writer:

        # Create images one by one using stimMaker and save them
        for i in range(n_samples):
            # Print the percentage-progress.
            print_progress(count=i, total=n_samples - 1)

            for stim in range(len(stim_matrices)):
                config_img, vernier_label = stim_maker.makeConfigBatch(1, configMatrix=stim_matrices[stim], noiseLevel=noise_level, normalize=normalize_images, fixed_position=fixed_stim_position)

                # Convert the image to raw bytes.
                config_img_bytes = config_img.tostring()
                vernier_label_bytes = vernier_label.tostring()

                # Create a dict with the data we want to save in the
                # TFRecords file. You can add more relevant data here.
                data = \
                    {
                        'config_img': wrap_bytes(config_img_bytes),
                        'vernier_label': wrap_bytes(vernier_label_bytes),
                    }

                # Wrap the data as TensorFlow Features.
                feature = tf.train.Features(feature=data)

                # Wrap again as a TensorFlow Example.
                example = tf.train.Example(features=feature)

                # Serialize the data.
                serialized = example.SerializeToString()

                # Write the serialized data to the TFRecords file.
                writer.write(serialized)


def parse_multi_shape(serialized):
    # variables are in byte form in the tfrecord file. this converts back to the right format.
    # Define a dict with the data-names and types we expect to
    # find in the TFRecords file.
    # It is a bit awkward that this needs to be specified again,
    # because it could have been written in the header of the
    # TFRecords file instead.
    features = \
        {
            'multi_shape_img': tf.FixedLenFeature([], tf.string),
            'single_shape_img': tf.FixedLenFeature([], tf.string),
            'labels': tf.FixedLenFeature([], tf.string),
            'vernier_label': tf.FixedLenFeature([], tf.string),
            'n_elements': tf.FixedLenFeature([], tf.string)
        }

    # Parse the serialized data so we get a dict with our data.
    parsed_example = tf.parse_single_example(serialized=serialized, features=features)

    # Get the image as raw bytes.
    multi_shape_img_raw = parsed_example['multi_shape_img']
    single_shape_img_raw = parsed_example['single_shape_img']
    labels_raw = parsed_example['labels']
    vernier_label_raw = parsed_example['vernier_label']
    n_elements_raw = parsed_example['n_elements']

    # Decode the raw bytes so it becomes a tensor with type.
    multi_shape_img = tf.decode_raw(multi_shape_img_raw, tf.float32)
    single_shape_img = tf.decode_raw(single_shape_img_raw, tf.float32)
    labels = tf.decode_raw(labels_raw, tf.float32)
    vernier_label = tf.decode_raw(vernier_label_raw, tf.int64)
    n_elements = tf.decode_raw(n_elements_raw, tf.int64)

    return multi_shape_img, single_shape_img, labels, vernier_label, n_elements


def parse_config(serialized):
    # variables are in byte form in the tfrecord file. this converts back to the right format.
    # Define a dict with the data-names and types we expect to
    # find in the TFRecords file.
    # It is a bit awkward that this needs to be specified again,
    # because it could have been written in the header of the
    # TFRecords file instead.
    features = \
        {
            'config_img': tf.FixedLenFeature([], tf.string),
            'vernier_label': tf.FixedLenFeature([], tf.string),
        }

    # Parse the serialized data so we get a dict with our data.
    parsed_example = tf.parse_single_example(serialized=serialized, features=features)

    # Get the image as raw bytes.
    config_img_raw = parsed_example['config_img']
    vernier_label_raw = parsed_example['vernier_label']

    # Decode the raw bytes so it becomes a tensor with type.
    config_img = tf.decode_raw(config_img_raw, tf.float32)
    vernier_label = tf.decode_raw(vernier_label_raw, tf.float32)

    return config_img, vernier_label


def input_fn_multi_shape(filenames, train, n_epochs=n_epochs, batch_size=batch_size, buffer_size=buffer_size):
    # needed for estimators in the main script
    # Args:
    # filenames:   Filenames for the TFRecords files.
    # train:       Boolean whether training (True) or testing (False).
    # batch_size:  Return batches of this size.
    # buffer_size: Read buffers of this size. The random shuffling
    #              is done on the buffer, so it must be big enough.

    # Create a TensorFlow Dataset-object which has functionality
    # for reading and shuffling data from TFRecords files.
    dataset = tf.data.TFRecordDataset(filenames=filenames)

    # Parse the serialized data in the TFRecords files.
    # This returns TensorFlow tensors for the image and labels.
    dataset = dataset.map(parse_multi_shape)

    if train:
        # If training then read a buffer of the given size and
        # randomly shuffle it.
        dataset = dataset.shuffle(buffer_size=buffer_size)

        # Allow infinite reading of the data.
        num_repeat = n_epochs
    else:
        # If testing then don't shuffle the data.

        # Only go through the data once.
        num_repeat = 1

    # Repeat the dataset the given number of times.
    dataset = dataset.repeat(num_repeat)

    # Get a batch of data with the given size.
    dataset = dataset.batch(batch_size)

    # Create an iterator for the dataset and the above modifications.
    iterator = dataset.make_one_shot_iterator()

    # Get the next batch of images and labels.
    multi_shape_img, single_shape_img, labels, vernier_label, n_elements = iterator.get_next()

    # reshape images (they were flattened when transformed into bytes
    multi_shape_img = tf.reshape(multi_shape_img, [-1, im_size[0], im_size[1], 1])
    single_shape_img = tf.reshape(single_shape_img, [-1, im_size[0], im_size[1], 2])

    # The input-function must return a dict wrapping the images.
    if train:
        feed_dict = {'X': multi_shape_img,
                     'reconstruction_targets': single_shape_img,
                     'y': labels,
                     'n_shapes': n_elements,
                     'vernier_offsets': vernier_label,
                     'mask_with_labels': True,
                     'is_training': True}
    else:
        feed_dict = {'X': multi_shape_img,
                     'reconstruction_targets': single_shape_img,
                     'y': labels,
                     'n_shapes': n_elements,
                     'vernier_offsets': vernier_label,
                     'mask_with_labels': False,
                     'is_training': False}

    return feed_dict, labels


def input_fn_config(filenames, batch_size=batch_size):
    # needed for estimators in the main script
    # Args:
    # filenames:   Filenames for the TFRecords files.
    # train:       Boolean whether training (True) or testing (False).
    # batch_size:  Return batches of this size.
    # buffer_size: Read buffers of this size. The random shuffling
    #              is done on the buffer, so it must be big enough.

    # Create a TensorFlow Dataset-object which has functionality
    # for reading and shuffling data from TFRecords files.
    dataset = tf.data.TFRecordDataset(filenames=filenames)

    # Parse the serialized data in the TFRecords files.
    # This returns TensorFlow tensors for the image and labels.
    dataset = dataset.map(parse_config)

    # Get a batch of data with the given size.
    dataset = dataset.batch(batch_size)

    # Create an iterator for the dataset and the above modifications.
    iterator = dataset.make_one_shot_iterator()

    # Get the next batch of images and labels.
    config_img, vernier_label = iterator.get_next()

    # reshape images (they were flattened when transformed into bytes
    config_img = tf.reshape(config_img, [-1, im_size[0], im_size[1], 1])

    # this is not very elegant, but in the testing phase we don't need the reconstruction targets nor the shape labels.
    # but the model expects to receive them, so we fill them with zeros.
    feed_dict = {'X': config_img,
                 'reconstruction_targets': np.zeros(shape=(batch_size, im_size[0], im_size[1], simultaneous_shapes)),
                 'y': vernier_label,
                 'vernier_offsets': vernier_label,
                 'mask_with_labels': False,
                 'is_training': False}

    return feed_dict, vernier_label


# these functions are needed to accomodate the estimator API
def train_input_fn():
    return input_fn_multi_shape(train_data_path, train=True)


def test_input_fn(test_data_path):
    return input_fn_config(test_data_path)


# if you want to check data at the end to make sure everything is ok.
def show_data(filename, type):
    # shows processed data from the filename file
    # type: # 'config' or 'multi_shape'

    with tf.Session() as sess:

        if type is 'multi_shape':

            data_out, labels_out = input_fn_multi_shape(filename, 0)
            img, single_shape_img, labels, n_elements, vernier_offsets = sess.run([data_out['X'], data_out['reconstruction_targets'], data_out['y'], data_out['n_shapes'], data_out['vernier_offsets']])

            # Loop over each example in batch
            for i in range(10):
                plt.imshow(img[i, :, :, 0])
                plt.title('Class label = ' + str(labels[i, :]) + ', vernier = ' + str(vernier_offsets[i]) + ', n_elements = ' + str(n_elements[i,:]))
                plt.show()
                plt.imshow(single_shape_img[i, :, :, 0])
                plt.show()
                plt.imshow(single_shape_img[i, :, :, 1])
                plt.show()

        elif type is 'config':

            data_out, labels_out = input_fn_config(filename)
            img, labels = sess.run([data_out['X'], data_out['y']])

            # Loop over each example in batch
            for i in range(10):
                plt.imshow(img[i, :, :, 0])
                plt.title('Class label = ' + str(labels[i]))
                plt.show()
        else:
            raise Exception('Unknown data type. Enter "multi_shape" or "config" or set check_data = None in parameters')