import sys
from capser_7_input_fn import *
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

            multi_shape_img, single_shape_img, labels, vernier_label, n_elements = stim_maker.makeMultiShapeBatch(1, shape_types, n_shapes=simultaneous_shapes, noiseLevel=noise_level, group_last_shapes=group_last_shapes, vernierLabelEncoding=vernier_label_encoding, max_rows=max_rows, max_cols=max_cols, vernier_grids=vernier_grids, normalize=normalize_images, normalize_sets=normalize_sets, fixed_position=fixed_stim_position, random_size=random_size)

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

            if stim_matrices is None:
                n_stim_types = 1
            else:
                n_stim_types = len(stim_matrices)

            for stim in range(n_stim_types):
                config_img, vernier_label = stim_maker.makeConfigBatch(1, configMatrix=stim_matrices[stim], noiseLevel=noise_level, normalize=normalize_images, normalize_sets=normalize_sets,  fixed_position=fixed_stim_position, vernierLabelEncoding=vernier_label_encoding, random_size=random_size)

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


# if you want to check data at the end to make sure everything is ok.
def show_data(filename, type):
    # shows processed data from the filename file
    # type: # 'config' or 'multi_shape'

    with tf.Session() as sess:

        if type is 'multi_shape':

            data_out, labels_out = input_fn_multi_shape(filename, 0)
            img, single_shape_img, labels, n_elements, vernier_offsets = sess.run([data_out['X'], data_out['reconstruction_targets'], data_out['y'], data_out['n_shapes'], data_out['vernier_offsets']])

            print(labels)
            print(labels.shape)
            print(vernier_offsets)
            print(vernier_offsets.shape)
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
            img, labels = sess.run([data_out['X'], data_out['vernier_offsets']])

            print(labels)
            print(labels.shape)

            # Loop over each example in batch
            for i in range(10):
                plt.imshow(img[i, :, :, 0])
                plt.title('Class label = ' + str(labels[i]) + ', IM_SIZE='+str(img.shape))
                plt.show()
        else:
            raise Exception('Unknown data type. Enter "multi_shape" or "config" or set check_data = None in parameters')