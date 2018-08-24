from image_decoder_input_fn import make_image_decoder_tfrecords, show_data
from parameters import test_stimuli
from image_decoder_parameters import image_decoder_data_path, n_train_samples, n_test_samples, source_image_path
import os

training = 0
testing = 1

# choose which sets to create
show_samples = 1

if training:
    make_image_decoder_tfrecords(source_image_path + '/np_arrays_vernier_train_set/', n_train_samples, os.path.join(image_decoder_data_path, 'vernier_train_set.tfrecords'))
    if show_samples:
        show_data(os.path.join(image_decoder_data_path, 'vernier_train_set.tfrecords'))

if testing:
    for category in test_stimuli.keys():
        for this_stim in range(3):
            if not os.path.exists(image_decoder_data_path+'/'+category):
                os.makedirs(image_decoder_data_path+'/'+category)
            make_image_decoder_tfrecords(source_image_path+'/np_arrays_'+category+'/'+str(this_stim), n_test_samples, os.path.join(image_decoder_data_path, category+"/test_" + category + str(this_stim) + ".tfrecords"))
            if show_samples:
                show_data(os.path.join(image_decoder_data_path, category+"/test_" + category + str(this_stim) + ".tfrecords"))