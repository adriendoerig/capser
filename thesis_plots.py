# -*- coding: utf-8 -*-
"""
Plots for thesis:
Created on Fri Feb 22 17:00:04 2019
@author: Lynn
"""

from my_batchmaker import stim_maker_fn
import matplotlib.pyplot as plt
import numpy as np

imSize = [20, 72]
#imSize = [30, 75]
n_shapes = [1, 3, 5]
shapeSize = 14
barWidth = 1

test = stim_maker_fn(imSize, shapeSize, barWidth)

# For training datasets:
chosen_shape = 6
[shape_images_0, shape_images_1, _, _, _, _, _, _, _, _] = test.makeTrainBatch(
 chosen_shape, n_shapes, 1, 'vernier_shape', overlap=True, centralize=False, reduce_df=True)

plt.figure()
plt.imshow(np.clip(np.squeeze(shape_images_1 + np.random.normal(0, 0.08, [imSize[0], imSize[1], 1])), 0, 1), cmap='gray')
plt.axis('off')

#shapes = [0, 1, 2, 3, 4, 5, 6]
#[shape_images_0, shape_images_1, _, _, _, _, _, _, _, _] = test.makeTrainBatch(
# shapes, n_shapes, 1, 'random_random', overlap=True, centralize=False, reduce_df=True)
#
#plt.figure()
#plt.imshow(np.clip(np.squeeze(shape_images_0 + shape_images_1 + np.random.normal(0, 0.08, [imSize[0], imSize[1], 1])), 0, 1), cmap='gray')
#plt.axis('off')




#[vernier_images, shape_images, _, _, _, _, _, _, _, _] = test.makeTestBatch(416, n_shapes, 1, 2, False, True)
#plt.figure()
#plt.imshow(np.clip(np.squeeze(vernier_images + shape_images + np.random.normal(0, 0.1, [imSize[0], imSize[1], 1])), 0, 1), cmap='gray')
#plt.axis('off')

