import os, sys, random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from image_decoder_parameters import batch_size, buffer_size, im_size

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


def make_image_decoder_tfrecords(in_path, n_samples, out_path):
    # Kind of stupid, but we must create tfRecords differently for config_batches because they don't return the same
    # things as multi_shape_batches (see batchMaker script)
    # Args:
    # where np arrays are   an instance of the StimMaker class (see batchMaker script)
    # n_samples             the total number of stimuli.
    # out_path              File-path for the TFRecords output file.

    print("\nConverting: " + out_path)

    # Open a TFRecordWriter for the output-file.
    with tf.python_io.TFRecordWriter(out_path) as writer:

        # list files in folders
        files_im = os.listdir(in_path)
        random.shuffle(files_im)

        im_sample = np.load(in_path + '/' + files_im[0])
        im_size = im_sample.shape

        for n in range(n_samples):

            print_progress(count=n, total=n_samples - 1)

            image = np.load(in_path + '/' + files_im[n])
            image = np.expand_dims(image, 0)
            if files_im[n][-10] == 'L':
                label = np.array(0., dtype=np.float32)
            elif files_im[n][-10] == 'R':
                label = np.array(1., dtype=np.float32)
            else:
                print('CAREFUL, LABELS WILL BE FUCKED UP')

            # Convert the image to raw bytes.
            images_bytes = image.tostring()
            labels_bytes = label.tostring()

            # Create a dict with the data we want to save in the
            # TFRecords file. You can add more relevant data here.
            data = \
                {
                    'images': wrap_bytes(images_bytes),
                    'labels': wrap_bytes(labels_bytes),
                }

            # Wrap the data as TensorFlow Features.
            feature = tf.train.Features(feature=data)

            # Wrap again as a TensorFlow Example.
            example = tf.train.Example(features=feature)

            # Serialize the data.
            serialized = example.SerializeToString()

            # Write the serialized data to the TFRecords file.
            writer.write(serialized)


def parse_image_decoder(serialized):
    # variables are in byte form in the tfrecord file. this converts back to the right format.
    # Define a dict with the data-names and types we expect to
    # find in the TFRecords file.
    # It is a bit awkward that this needs to be specified again,
    # because it could have been written in the header of the
    # TFRecords file instead.
    features = \
        {
            'images': tf.FixedLenFeature([], tf.string),
            'labels': tf.FixedLenFeature([], tf.string),
        }

    # Parse the serialized data so we get a dict with our data.
    parsed_example = tf.parse_single_example(serialized=serialized, features=features)

    # Get the image as raw bytes.
    images_raw = parsed_example['images']
    labels_raw = parsed_example['labels']

    # Decode the raw bytes so it becomes a tensor with type.
    images = tf.decode_raw(images_raw, tf.float32)
    labels = tf.squeeze(tf.decode_raw(labels_raw, tf.float32))

    return images, labels


def image_decoder_input_function(filenames, batch_size=batch_size, buffer_size=buffer_size):

    # Create a TensorFlow Dataset-object which has functionality
    # for reading and shuffling data from TFRecords files.
    # Use pipelining to speed up things (see https://www.youtube.com/watch?v=SxOsJPaxHME)
    dataset = tf.data.TFRecordDataset(filenames=filenames, num_parallel_reads=32)

    # Parse the serialized data in the TFRecords files.
    # This returns TensorFlow tensors for the image and labels.
    # Apply parsing in parallel to speed up (see https://www.youtube.com/watch?v=SxOsJPaxHME)
    dataset = dataset.map(parse_image_decoder, num_parallel_calls=64)

    dataset = dataset.shuffle(buffer_size=buffer_size)

    # Only go through the data once.
    num_repeat = 1

    # Repeat the dataset the given number of times and get a batch of data with the given size.
    dataset = dataset.repeat(num_repeat).apply(tf.contrib.data.batch_and_drop_remainder(batch_size))

    # Use pipelining to speed up things (see https://www.youtube.com/watch?v=SxOsJPaxHME)
    dataset = dataset.prefetch(2)

    # Create an iterator for the dataset and the above modifications.
    iterator = dataset.make_one_shot_iterator()

    # Get the next batch of images and labels.
    images, labels = iterator.get_next()

    # reshape images (they were flattened when transformed into bytes
    images = tf.reshape(images, [batch_size, im_size[0], im_size[1], 3])

    # this is not very elegant, but in the testing phase we don't need the reconstruction targets nor the shape labels.
    # but the model expects to receive them, so we fill them with zeros.
    feed_dict = {'X': images,
                 'y': labels}

    return feed_dict, labels


# if you want to check data at the end to make sure everything is ok.
def show_data(filename):

    with tf.Session() as sess:

            data_out, labels_out = image_decoder_input_function(filename)
            print(data_out)
            img, y = sess.run([data_out['X'], data_out['y']])

            print(y)
            print(y.shape)

            # Loop over each example in batch
            for i in range(10):
                plt.imshow(img[i, :, :, :])
                plt.title('Class label = ' + str(y[i]) + ', IM_SIZE='+str(img.shape))
                plt.show()


