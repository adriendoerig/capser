import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from capsule_functions import safe_norm
from batchMaker import StimMaker
import tensorflow.contrib.framework

# data parameters
fixed_stim_position = None  # put top left corner of all stimuli at fixed_position
normalize_images = False    # make each image mean=0, std=1
max_rows, max_cols = 1, 3   # max number of rows, columns of shape grids
vernier_grids = False       # if true, verniers come in grids like other shapes. Only single verniers otherwise.
im_size = (40, 60)         # IF USING THE DECONVOLUTION DECODER NEED TO BE EVEN NUMBERS (NB. this suddenly changed. before that, odd was needed... that's odd.)
shape_size = 15             # size of a single shape in pixels
simultaneous_shapes = 2     # number of different shapes in an image. NOTE: more than 2 is not supported at the moment
bar_width = 2               # thickness of elements' bars
noise_level = 0  # 10       # add noise
shape_types = [0, 1, 2]  # see batchMaker.drawShape for number-shape correspondences
group_last_shapes = 1       # attributes the same label to the last n shapeTypes
label_to_shape = {0: 'vernier', 1: 'squares', 2:'circles'}
shape_to_label = dict( [ [v, k] for k, v in label_to_shape.items() ] )

batch_size = 10

stim_maker = StimMaker(im_size, shape_size, bar_width)  # handles data generation
test_stimuli = {'squares':       [None, [[1]], [[1, 1, 1]]],
                'circles':       [None, [[2]], [[2, 2, 2]]]}

batch_data, batch_single_shape_images, batch_labels, vernier_offset_labels, n_elements = stim_maker.makeMultiShapeBatch(
    batch_size, shape_types, n_shapes=simultaneous_shapes, noiseLevel=noise_level, group_last_shapes=group_last_shapes,
    max_rows=max_rows, max_cols=max_cols, vernier_grids=vernier_grids,
    normalize=normalize_images, fixed_position=fixed_stim_position)


sorted_labels = tensorflow.contrib.framework.sort(
    batch_labels,
    axis=-1,
    direction='ASCENDING',
    name=None
)

with tf.Session() as sess:

    sorted = sorted_labels.eval()
    print(sorted)

print(batch_labels)
print(sorted_labels)
