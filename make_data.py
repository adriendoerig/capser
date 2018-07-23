from make_tf_dataset import *

# choose which sets to create
train = 0
vernier = 0
test_categories = 1
test_individual_stimuli = 1

# create tfRecord data file
if train:
    make_multi_shape_tfRecords(stim_maker, shape_types, n_train_samples, train_data_path)
if vernier:
    
