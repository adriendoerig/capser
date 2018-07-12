import numpy as np
from parameters import *
from tensorflow.contrib.data import batch_and_drop_remainder

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
    # Use pipelining to speed up things (see https://www.youtube.com/watch?v=SxOsJPaxHME)
    dataset = tf.data.TFRecordDataset(filenames=filenames, num_parallel_reads=32)

    # Parse the serialized data in the TFRecords files.
    # This returns TensorFlow tensors for the image and labels.
    # do the mapping in parallel to speed up things (see https://www.youtube.com/watch?v=SxOsJPaxHME))
    dataset = dataset.map(parse_multi_shape, num_parallel_calls=64)

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
    dataset = dataset.repeat(num_repeat).apply(batch_and_drop_remainder(batch_size))


    # Get a batch of data with the given size.
    dataset = dataset.batch(batch_size)

    # Use pipelining to speed up things (see https://www.youtube.com/watch?v=SxOsJPaxHME)
    dataset = dataset.prefetch(2)

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
    # Use pipelining to speed up things (see https://www.youtube.com/watch?v=SxOsJPaxHME)
    dataset = tf.data.TFRecordDataset(filenames=filenames, num_parallel_reads=32)

    # Parse the serialized data in the TFRecords files.
    # This returns TensorFlow tensors for the image and labels.
    # Apply parsing in parallel to speed up (see https://www.youtube.com/watch?v=SxOsJPaxHME)
    dataset = dataset.map(parse_config, num_parallel_calls=64)

    # Get a batch of data with the given size.
    dataset = dataset.batch(batch_size)

    # Use pipelining to speed up things (see https://www.youtube.com/watch?v=SxOsJPaxHME)
    dataset = dataset.prefetch(2)

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

def train_input_fn_tpu(params):
    # for a TPUEstimator, a params argument MUST be provided (even though here we don't use it).
    return input_fn_multi_shape(train_data_path, train=True)

def test_input_fn(test_data_path):
    return input_fn_config(test_data_path)
