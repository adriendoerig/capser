import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from batchMaker import StimMaker

# data parameters
im_size = (70, 140)
im_size_prod = im_size[0] * im_size[1]
shape_size = 19
bar_width = 2
noise_level = 10
shape_types = [1, 2, 6, 7]  # see batchMaker.drawShape for number-shape correspondences
group_last_shapes = 1       # attributes the same label to the last n shapeTypes
label_to_shape = {0: 'vernier', 1: 'squares', 2: 'circles', 3: '7stars', 4: 'stuff'}
shape_to_label = dict( [ [v, k] for k, v in label_to_shape.items() ] )

stim_maker = StimMaker(im_size, shape_size, bar_width)  # handles data generation
batch_data, batch_labels = stim_maker.makeBatch(10, shape_types, noise_level, group_last_shapes)

X = tf.placeholder(shape=[None, im_size[0], im_size[1], 1], dtype=tf.float32, name="X")

X_flat = tf.reshape(X, [-1, im_size_prod], name="X_flat")
squared_difference = tf.square(X_flat - X_flat,
                               name="squared_difference")
squared_difference_sum = tf.reduce_sum(squared_difference, axis=1)
scales = tf.reduce_sum(X_flat, axis=1)
diff_rescale = squared_difference_sum/scales
error = tf.reduce_sum(squared_difference,
                                    name="reconstruction_loss")
error_rescale = tf.reduce_sum(diff_rescale,
                                    name="reconstruction_loss")



with tf.Session() as sess:

    flat, diff, diff_sum, scales, diff_rescale, error, error_rescale = sess.run([X_flat, squared_difference, squared_difference_sum, scales, diff_rescale, error, error_rescale], feed_dict={X: batch_data})

    print('')
    print(flat.shape)
    print('')
    print(diff.shape)
    print('')
    print(diff_sum.shape, diff_sum)
    print('')
    print(scales)
    print('')
    print(diff_rescale.shape, diff_rescale)
    print('')
    print(error)
    print('')
    print(error_rescale)
    print('')

    for i in range(5):
        plt.figure()
        plt.imshow(batch_data[i,:,:].reshape(im_size[0], im_size[1]))
        plt.title('Scale = '+str(scales[i])+', Error = '+str(diff_sum[i])+', Rescaled = '+str(diff_rescale[i]))
        plt.show()