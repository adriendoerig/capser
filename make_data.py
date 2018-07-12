from make_tf_dataset import *

# create tfRecord data file
make_multi_shape_tfRecords(stim_maker, shape_types, n_train_samples, train_data_path)
