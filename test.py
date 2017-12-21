import tensorflow as tf
import numpy as np
from scipy import ndimage
import os
import random
from __future__ import division, print_function, unicode_literals
import matplotlib.pyplot as plt
import numpy as np

from create_sprite import images_to_sprite, invert_grayscale
from data_handling_functions import make_crowding_sets, make_shape_sets
from capsule_functions import primary_caps_layer, primary_to_fc_caps_layer, \
    fc_to_fc_caps_layer, caps_prediction, compute_margin_loss, create_masked_decoder_input, \
    decoder_with_mask, compute_reconstruction_loss
a = tf.constant(2)
