import tensorflow as tf
import sys
import os
from batchMaker import StimMaker
import matplotlib.pyplot as plt

data_write_path = './data'
if not os.path.exists(data_write_path):
    os.mkdir(data_write_path)

with tf.device('/cpu:0'):

    # first we create a dataset object with our data
    check_data = False  # set to true if you want to take a look at the processed data at the end
    n_samples = 1000  # number of stimuli in an epoch
    n_epochs = 10       # number of epochs

    im_size = (30, 60)  # IF USING THE DECONVOLUTION DECODER NEED TO BE EVEN NUMBERS (NB. this suddenly changed. before that, odd was needed... that's odd.)
    shape_size = 10  # size of a single shape in pixels
    bar_width = 1  # thickness of elements' bars
    shape_types = [1, 2, 9]  # see batchMaker.drawShape for number-shape correspondences
    group_last_shapes = 1  # attributes the same label to the last n shapeTypes
    batch_size = 16
    n_simultaneous_shapes = 2
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
    def make_tfRecords(stim_maker, shape_types, n_samples, samples_per_file, out_path):
        # Args:
        # n_samples             the total number of stimuli.
        # samples_per_file      We may create several tfRecords files - choose the number here.
        # out_path              File-path for the TFRecords output file.

        print("Converting: " + out_path)

        # Number of images. Used when printing the progress.

        # Open a TFRecordWriter for the output-file.
        with tf.python_io.TFRecordWriter(out_path) as writer:
            # Create images one by one using stimMaker and save them
            for i in range(n_samples):
                # Print the percentage-progress.
                print_progress(count=i, total=n_samples - 1)

                # Load the image-file using matplotlib's imread function.
                multi_shape_img, single_shape_img, labels, vernier_label, n_elements = stim_maker.makeMultiShapeBatch(1, shape_types, n_shapes=n_simultaneous_shapes)

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


    # variables are in byte form in the tfrecord file. this converts back to the right format.
    def parse(serialized):
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

        # The type is now uint8 but we need it to be float.
        # multi_shape_img = tf.cast(multi_shape_img, tf.float32)
        # single_shape_img = tf.cast(single_shape_img, tf.float32)
        # labels = tf.cast(labels, tf.int64)
        # vernier_label = tf.cast(vernier_label, tf.int64)
        # n_elements = tf.cast(n_elements, tf.int64)

        # The image and label are now correct TensorFlow types.
        return multi_shape_img, single_shape_img, labels, vernier_label, n_elements


    # needed for estimators in the main script
    def input_fn(filenames, train, n_epochs=1, batch_size=32, buffer_size=2048):
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
        dataset = dataset.map(parse)

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

    # these functions are needed to accomodate the estimator API
    def train_input_fn():
        return input_fn(filenames=os.path.join(data_write_path, "train.tfrecords"), train=True)

    def test_input_fn():
        return input_fn(filenames=os.path.join(data_write_path, "test.tfrecords"), train=False)




    #  create en tfRecord file
    make_tfRecords(stim_maker, shape_types, n_samples, 1, os.path.join(data_write_path, "train.tfrecords"))


    # if you want to check data at the end to make sure everything is ok.
    if check_data:

        data_out = input_fn('./data/train.tfrecords', 0)

        with tf.Session() as sess:
            img, single_shape_img, labels, n_elements, vernier_offsets = sess.run([data_out['X'], data_out['reconstruction_targets'], data_out['y'], data_out['n_shapes'], data_out['vernier_offsets']])

            print(img.shape)
            print(single_shape_img.shape)
            print(labels)

            # Loop over each example in batch
            for i in range(10):
                plt.imshow(img[i, :, :, 0])
                plt.title('Class label = ' + str(labels[i, :]) + ', vernier = ' + str(vernier_offsets[i]) + ', n_elements = ' + str(n_elements[i,:]))
                plt.show()
                plt.imshow(single_shape_img[i, :, :, 0])
                plt.show()
                plt.imshow(single_shape_img[i, :, :, 1])
                plt.show()