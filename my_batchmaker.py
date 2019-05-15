# -*- coding: utf-8 -*-
"""
Publishing results: my_batchmaker!
@author: Lynn

Last update on 09.05.2019
-> First draft of all stimuli is finished
-> Everything should be set
-> In training set, stimuli face each other now
-> Multiple lines appear now in the training dataset
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import draw


##################################
#      stim_maker class fn:      #
##################################
class stim_maker_fn:

    def __init__(self, imSize, shapeSize, barWidth, offset=1):
        shapeDepth = shapeSize[2]
        depthW = int(np.floor((np.tan(45 / 180 * np.pi) * shapeDepth)))
        depthH = int(np.floor((np.sin(45 / 180 * np.pi) * shapeDepth)))

        self.imSize = imSize
        self.depthW = depthW
        self.depthH = depthH
        # I want all the patches to have the same height
        self.patchHeight = shapeSize[1] + depthH

        self.shapeWidth = shapeSize[0]
        self.shapeHeight = shapeSize[1]
        self.barWidth = barWidth
        #        self.vernierOffsetWidth = 1
        self.vernierOffsetHeight = 1
        self.offset = offset
        
        # Some parameters that I set manually for the new project:
        self.shape_repetitions = 2
        self.face_each_other = 1
        self.line_repetitions = [2, 4, 6, 8]
        self.max_offset_line = self.offset * 6
        self.max_offset_stim = self.offset * 6

        if not np.mod(shapeSize[1] + self.vernierOffsetHeight, 2) == 0:
            raise SystemExit('\nshapeHeight + vernierOffsetHeight has to be even!')

    def drawVernier(self, offset, offset_direction):
        # Inputs:
        # offset_direction: 0=r, 1=l
        patchHeight = self.patchHeight
        height = self.shapeHeight
        depthH = self.depthH
        barW = self.barWidth
        offsetW = offset
        offsetH = self.vernierOffsetHeight

        vernierSize = int((height - offsetH) / 2)
        patch = np.zeros([patchHeight, 2 * barW + offsetW], dtype=np.float32)
        patch[depthH:depthH + vernierSize, 0:barW] = 1
        patch[depthH + offsetH + vernierSize:depthH + offsetH + vernierSize * 2, barW + offsetW:] = 1

        if offset_direction:
            patch = np.fliplr(patch)
        return patch

    def drawLines(self, offset=0):
        patchHeight = self.patchHeight
        height = self.shapeHeight
        depthH = self.depthH
        barW = self.barWidth

        patch = np.zeros([patchHeight, barW + 2 * offset], dtype=np.float32)
        patch[depthH:depthH + height, offset:offset + barW] = 1
        return patch

    def drawRectangles(self, offset=0):
        patchHeight = self.patchHeight
        height = self.shapeHeight
        width = self.shapeWidth
        depthH = self.depthH
        barW = self.barWidth

        patch = np.zeros([patchHeight, width + 2 * offset], dtype=np.float32)
        patch[depthH:depthH + height, offset + width - barW:offset + width] = 1
        patch[depthH:depthH + height, offset:offset + barW] = 1
        patch[depthH:depthH + barW, offset:offset + width] = 1
        patch[depthH + height - barW:depthH + height, offset:offset + width] = 1
        return patch

    def drawCuboidsL(self, offset=0):
        # Unfortunately, the drawing function is sometimes out of borders
        adjust = 1
        patchHeight = self.patchHeight
        height = self.shapeHeight
        width = self.shapeWidth
        depthW = self.depthW
        depthH = self.depthH
        barW = self.barWidth

        patch = np.zeros([patchHeight, width + depthW + 2 * offset], dtype=np.float32)
        patch[depthH:depthH + height, depthW + offset + width - barW:depthW + offset + width] = 1
        patch[depthH:depthH + height, depthW + offset:depthW + offset + barW] = 1
        patch[depthH:depthH + barW, depthW + offset:depthW + offset + width] = 1
        patch[depthH + height - barW:depthH + height, depthW + offset:depthW + offset + width] = 1

        patch[0:barW, offset:offset + width] = 1
        patch[height - adjust:height + barW - adjust, offset:offset + width] = 1
        patch[0:height, offset:offset + barW] = 1
        patch[0:height, offset + width:offset + width + barW] = 1

        row1, col1 = draw.line(0, offset, depthH, offset + depthW)
        row2, col2 = draw.line(height - adjust, offset, height + depthH - adjust, offset + depthW)
        row3, col3 = draw.line(0, width + offset - adjust, depthH, width + depthW + offset - adjust)
        row4, col4 = draw.line(height - adjust, width + offset - adjust, height + depthH - adjust,
                               width + depthW + offset - adjust)
        patch[row1, col1] = 1
        patch[row2, col2] = 1
        patch[row3, col3] = 1
        patch[row4, col4] = 1
        return patch

    def drawCuboidsR(self, offset):
        patch = self.drawCuboidsL(offset)
        patch = np.fliplr(patch)
        return patch

    def drawShuffledCuboidsL(self, offset=0):
        patchHeight = self.patchHeight
        height = self.shapeHeight
        width = self.shapeWidth
        depthW = self.depthW
        depthH = self.depthH
        barW = self.barWidth
        patchWidth = width + depthW + 2 * offset

        patch = np.zeros([patchHeight, patchWidth], dtype=np.float32)
        # The line close to the vernier should always stay the same:
        patch[depthH:depthH + height, depthW + offset + width - barW:depthW + offset + width] = 1
        # All others should be random
        rnd1 = np.random.randint(0, patchHeight - height)
        rnd2 = np.random.randint(offset, patchWidth - offset - barW)
        patch[rnd1:rnd1 + height, rnd2:rnd2 + barW] = 1

        rnd1 = np.random.randint(0, patchHeight - barW)
        rnd2 = np.random.randint(offset, patchWidth - offset - width)
        patch[rnd1:rnd1 + barW, rnd2:rnd2 + width] = 1

        rnd1 = np.random.randint(0, patchHeight - barW)
        rnd2 = np.random.randint(offset, patchWidth - offset - width)
        patch[rnd1:rnd1 + barW, rnd2:rnd2 + width] = 1

        rnd1 = np.random.randint(0, patchHeight - barW)
        rnd2 = np.random.randint(offset, patchWidth - offset - width)
        patch[rnd1:rnd1 + barW, rnd2:rnd2 + width] = 1

        rnd1 = np.random.randint(0, patchHeight - barW)
        rnd2 = np.random.randint(offset, patchWidth - offset - width)
        patch[rnd1:rnd1 + barW, rnd2:rnd2 + width] = 1

        rnd1 = np.random.randint(0, patchHeight - height)
        rnd2 = np.random.randint(offset, patchWidth - offset - barW)
        patch[rnd1:rnd1 + height, rnd2:rnd2 + barW] = 1

        rnd1 = np.random.randint(0, patchHeight - height)
        rnd2 = np.random.randint(offset, patchWidth - offset - barW)
        patch[rnd1:rnd1 + height, rnd2:rnd2 + barW] = 1

        rnd1 = np.random.randint(0, patchHeight - depthH)
        rnd2 = np.random.randint(offset, patchWidth - offset - depthW)
        row1, col1 = draw.line(rnd1, rnd2, rnd1 + depthH, rnd2 + depthW)

        rnd1 = np.random.randint(0, patchHeight - depthH)
        rnd2 = np.random.randint(offset, patchWidth - offset - depthW)
        row2, col2 = draw.line(rnd1, rnd2, rnd1 + depthH, rnd2 + depthW)

        rnd1 = np.random.randint(0, patchHeight - depthH)
        rnd2 = np.random.randint(offset, patchWidth - offset - depthW)
        row3, col3 = draw.line(rnd1, rnd2, rnd1 + depthH, rnd2 + depthW)

        rnd1 = np.random.randint(0, patchHeight - depthH)
        rnd2 = np.random.randint(offset, patchWidth - offset - depthW)
        row4, col4 = draw.line(rnd1, rnd2, rnd1 + depthH, rnd2 + depthW)

        patch[row1, col1] = 1
        patch[row2, col2] = 1
        patch[row3, col3] = 1
        patch[row4, col4] = 1
        return patch

    def drawShuffledCuboidsR(self, offset=0):
        patch = self.drawShuffledCuboidsL(offset)
        patch = np.fliplr(patch)
        return patch

    def drawShape(self, shapeID, offset, offset_direction=0):
        if shapeID == 0:
            patch = self.drawVernier(offset, offset_direction)
        if shapeID == 1:
            patch = self.drawLines(offset)
        if shapeID == 6:
            patch = self.drawRectangles(offset)
        if shapeID == 2:
            patch = self.drawCuboidsL(offset)
        if shapeID == 3:
            patch = self.drawShuffledCuboidsL(offset)
        if shapeID == 4:
            patch = self.drawCuboidsR(offset)
        if shapeID == 5:
            patch = self.drawShuffledCuboidsR(offset)
        return patch

    def plotStim(self, shape_types, offset=0, noise=0.):
        '''Visualize all chosen shape_types in one plot'''
        imSize = self.imSize
        patchHeight = self.patchHeight
        patchesWidth = 0

        image = np.zeros(imSize, dtype=np.float32) + np.random.normal(0, noise, size=imSize)
        row = np.random.randint(0, imSize[0] - patchHeight)
        for i in range(len(shape_types)):
            shape = shape_types[i]
            patch = self.drawShape(shape, offset)
            patchesWidth += np.size(patch, 1)

        col = np.random.randint(0, imSize[1] - patchesWidth + 1)
        for i in range(len(shape_types)):
            shape = shape_types[i]
            patch = self.drawShape(shape, offset)
            patchWidth = np.size(patch, 1)
            image[row:row + patchHeight, col:col + patchWidth] += patch
            col += patchWidth
        plt.figure()
        plt.imshow(image)
        return

    def makeTestBatch(self, crowding_config, n_shapes, batch_size, stim_idx=None, centralize=True, reduce_df=False):
        '''Create one batch of test dataset according to stim_idx'''
        # Inputs:
        # selected_shape
        # n_shapes: list of shape repetitions
        # batch_size
        # stim_idx: decides whether to create vernier(0), crowding (1) or uncrowding (2) stimulus
        # noise: Random gaussian noise between [0,noise] gets added

        # Outputs:
        # vernier_images, shape_images, shapelabels, nshapeslabels, vernierlabels

        imSize = self.imSize
        patchHeight = self.patchHeight
        selected_shape = crowding_config[0]
        offset = self.offset
        shape_repetitions = self.shape_repetitions

        vernier_images = np.zeros(shape=[batch_size, imSize[0], imSize[1]], dtype=np.float32)
        shape_images = np.zeros(shape=[batch_size, imSize[0], imSize[1]], dtype=np.float32)
        shapelabels_idx = np.zeros(shape=[batch_size, 2], dtype=np.float32)
        vernierlabels_idx = np.zeros(shape=[batch_size, 1], dtype=np.float32)
        nshapeslabels = np.zeros(shape=[batch_size, 1], dtype=np.float32)
        nshapeslabels_idx = np.zeros(shape=[batch_size, 1], dtype=np.float32)
        x_vernier = np.zeros(shape=[batch_size, 1], dtype=np.float32)
        y_vernier = np.zeros(shape=[batch_size, 1], dtype=np.float32)
        x_shape = np.zeros(shape=[batch_size, 1], dtype=np.float32)
        y_shape = np.zeros(shape=[batch_size, 1], dtype=np.float32)

        for idx_batch in range(batch_size):
            vernier_image = np.zeros(imSize, dtype=np.float32)
            shape_image = np.zeros(imSize, dtype=np.float32)
            row = np.random.randint(0, imSize[0] - patchHeight)

            if stim_idx is None:
                idx = np.random.randint(0, 2)
            else:
                idx = stim_idx

            # For the reduction of the dfs, we need to know all the widths:
            # In this case, the cuboid patch width is biggest
            offset_direction = np.random.randint(0, 2)
            vernier_patch = self.drawShape(0, offset, offset_direction)
            vernierpatch_width = np.size(vernier_patch, 1)
            maxWidth = 2 * (self.shapeWidth + self.depthW + offset * 2) + vernierpatch_width

            if idx == 0:
                # Vernier test stimuli:
                selected_repetitions = 1
                nshapes_label = 0
                totalWidth = vernierpatch_width

                if reduce_df:
                    # We want to make the degrees of freedom for position on the x axis fair.
                    # For this condition, we have to reduce the image size depending on the actual patch width
                    imSize_adapted = imSize[1] - maxWidth + totalWidth
                    imStart = int((imSize[1] - imSize_adapted) / 2)
                    col = np.random.randint(imStart, imStart + imSize_adapted - totalWidth + 1)

                else:
                    col = np.random.randint(0, imSize[1] - totalWidth)

                vernier_image[row:row + patchHeight, col:col + vernierpatch_width] += vernier_patch
                x_vernier_ind, y_vernier_ind = col, row
                x_shape_ind, y_shape_ind = col, row

            elif idx == 1:
                # Crowded test stimuli:
                selected_repetitions = shape_repetitions
                nshapes_label = 1

                totalWidth = 0
                for i in range(len(crowding_config)):
                    shape = crowding_config[i]
                    shape_patch = self.drawShape(shape, offset)
                    totalWidth += np.size(shape_patch, 1)

                if reduce_df:
                    # We want to make the degrees of freedom for position on the x axis fair.
                    # For this condition, we have to reduce the image size depending on the actual patch width
                    imSize_adapted = imSize[1] - maxWidth + totalWidth
                    imStart = int((imSize[1] - imSize_adapted) / 2)
                    col = np.random.randint(imStart, imStart + imSize_adapted - totalWidth + 1)

                else:
                    col = np.random.randint(0, imSize[1] - totalWidth)

                # The shape is always first, thats y we can already take the coordinates:
                x_shape_ind, y_shape_ind = col, row

                # Loop through the configuration for the crowding stimulus
                for i in range(len(crowding_config)):
                    shape = crowding_config[i]
                    shape_patch = self.drawShape(shape, offset)
                    patchWidth = np.size(shape_patch, 1)
                    if shape == 0:
                        vernier_image[row:row + patchHeight, col:col + patchWidth] += vernier_patch
                        x_vernier_ind, y_vernier_ind = col, row
                    else:
                        shape_image[row:row + patchHeight, col:col + patchWidth] += shape_patch
                    col += patchWidth

            vernier_images[idx_batch, :, :] = vernier_image
            shape_images[idx_batch, :, :] = shape_image
            shapelabels_idx[idx_batch, 0] = 0
            shapelabels_idx[idx_batch, 1] = selected_shape
            nshapeslabels[idx_batch] = selected_repetitions
            nshapeslabels_idx[idx_batch] = nshapes_label
            vernierlabels_idx[idx_batch] = offset_direction
            x_vernier[idx_batch] = x_vernier_ind
            y_vernier[idx_batch] = y_vernier_ind
            x_shape[idx_batch] = x_shape_ind
            y_shape[idx_batch] = y_shape_ind

        # add the color channel for tensorflow:
        vernier_images = np.expand_dims(vernier_images, -1)
        shape_images = np.expand_dims(shape_images, -1)
        return [vernier_images, shape_images, shapelabels_idx, vernierlabels_idx,
                nshapeslabels, nshapeslabels_idx, x_vernier, y_vernier, x_shape, y_shape]

    def makeTrainBatch(self, shape_types, n_shapes, batch_size, train_procedure, overlap,
                       centralize, reduce_df=False):
        '''Create one batch of training dataset with each one vernier and
        one shape_type repeated n_shapes[random] times'''

        imSize = self.imSize
        patchHeight = self.patchHeight
        shape_repetitions = self.shape_repetitions
        face_each_other = self.face_each_other
        line_reps = self.line_repetitions
        # Set the max random offset between the stimuli:
        max_offset_line = self.max_offset_line
        max_offset_stim = self.max_offset_stim
        
        # I am setting the offset to 0 here, since I want to introduce a rd_offset afterwards
        offset = 0
        maxPatchWidth = self.shapeWidth + self.depthW + offset * 2

        shape_1_images = np.zeros(shape=[batch_size, imSize[0], imSize[1]], dtype=np.float32)
        shape_2_images = np.zeros(shape=[batch_size, imSize[0], imSize[1]], dtype=np.float32)
        shapelabels_idx = np.zeros(shape=[batch_size, 2], dtype=np.float32)
        vernierlabels_idx = np.zeros(shape=[batch_size, 1], dtype=np.float32)
        nshapeslabels = np.zeros(shape=[batch_size, 2], dtype=np.float32)
        nshapeslabels_idx = np.zeros(shape=[batch_size, 2], dtype=np.float32)
        x_shape_1 = np.zeros(shape=[batch_size, 1], dtype=np.float32)
        y_shape_1 = np.zeros(shape=[batch_size, 1], dtype=np.float32)
        x_shape_2 = np.zeros(shape=[batch_size, 1], dtype=np.float32)
        y_shape_2 = np.zeros(shape=[batch_size, 1], dtype=np.float32)

        for idx_batch in range(batch_size):
            shape_1_image = np.zeros(imSize, dtype=np.float32)
            shape_2_image = np.zeros(imSize, dtype=np.float32)

            try:
                # I want every second image with a vernier:
                if np.random.rand(1) < 0.5:
                    selected_shape_1 = 0
                else:
                    selected_shape_1 = np.random.randint(0, len(shape_types))
                selected_shape_2 = np.random.randint(1, len(shape_types))
            except:
                # if len(shape_types)=1, then just use this given shape_type
                selected_shape_1 = 0
                selected_shape_2 = shape_types

            # Create shape images:
            if selected_shape_1 == 0:
                # if the first shape is a vernier, only repeat once and use offset_direction 0=r or 1=l
                idx_n_shapes_1 = 0
                selected_repetitions_1 = 1
                rd_offset = 0
                offset_direction = np.random.randint(0, 2)
                shape_1_patch = self.drawShape(selected_shape_1, self.offset, offset_direction)
            elif selected_shape_1 == 1:
                # the line stimulus should be the only one that can be repeated more often:
                idx_n_shapes_1 = np.random.randint(0, len(line_reps)) + 1  # atm, idx_n_shapes_1=1 means 2 reps
                selected_repetitions_1 = line_reps[np.random.randint(0, len(line_reps))]
                rd_offset = np.random.randint(self.offset, max_offset_line + 1)
                offset_direction = np.random.randint(0, 2)
                shape_1_patch = self.drawShape(selected_shape_1, offset)
            else:
                # if not, repeat shape random times but at least once and set offset_direction to 2=no vernier
                idx_n_shapes_1 = 1
                selected_repetitions_1 = shape_repetitions
                rd_offset = np.random.randint(self.offset, max_offset_stim + 1)
                #                rd_offset = np.random.randint(self.offset, imSize[1]-maxPatchWidth*shape_repetitions)
                offset_direction = np.random.randint(0, 2)
                shape_1_patch = self.drawShape(selected_shape_1, offset)

            # In our case, shape2 is super irrelevant, because we will work only with one shape per image
            idx_n_shapes_2 = 0
            selected_repetitions_2 = 1
            shape_2_patch = self.drawShape(0, offset)
            row_shape_2 = 0
            col_shape_2_init = 0
            col_shape_2 = col_shape_2_init

            # For the reduction of the dfs, we need to know the patch widths:
            # In this case, the cuboid patch width is biggest
            shape1patch_width = np.size(shape_1_patch, 1)
            shape2patch_width = np.size(shape_2_patch, 1)

            row_shape_1 = np.random.randint(0, imSize[0] - patchHeight)
            if reduce_df:
                # We want to make the degrees of freedom for position on the x axis fair.
                # For this condition, we have to reduce the image size depending on the actual patch width
                if idx_n_shapes_1 == 0:
                    imSize_adapted = imSize[1] - maxPatchWidth * shape_repetitions + shape1patch_width * selected_repetitions_1 - 1
                else:
                    imSize_adapted = imSize[1] - maxPatchWidth * shape_repetitions + shape1patch_width * selected_repetitions_1
                imStart = int((imSize[1] - imSize_adapted) / 2)
                col_shape_1_init = np.random.randint(imStart, imStart + imSize_adapted - shape1patch_width * selected_repetitions_1 - rd_offset)
                col_shape_1 = col_shape_1_init

            else:
                col_shape_1_init = np.random.randint(0, imSize[1] - shape1patch_width * selected_repetitions_1 - rd_offset)
                col_shape_1 = col_shape_1_init

            if selected_shape_1 == 0:
                # If vernier, we only want one shape per image:
                shape_1_image[row_shape_1:row_shape_1 + patchHeight,
                col_shape_1:col_shape_1 + shape1patch_width] += shape_1_patch
            else:
                # Repeat shape_1 selected_repetitions times if not vernier:
                for i in range(selected_repetitions_1):
                    shape_1_image[row_shape_1:row_shape_1 + patchHeight,
                    col_shape_1:col_shape_1 + shape1patch_width] += shape_1_patch
                    col_shape_1 += shape1patch_width + rd_offset
                    if face_each_other == 1:
                        shape_1_patch = np.fliplr(shape_1_patch)

            # Repeat shape_2 selected_repetitions times:
            for i in range(selected_repetitions_2):
                shape_2_image[row_shape_2:row_shape_2 + patchHeight,
                col_shape_2:col_shape_2 + shape2patch_width] += shape_2_patch
                col_shape_2 += shape1patch_width + rd_offset

            shape_1_images[idx_batch, :, :] = shape_1_image
            shape_2_images[idx_batch, :, :] = shape_2_image
            shapelabels_idx[idx_batch, 0] = selected_shape_1
            shapelabels_idx[idx_batch, 1] = selected_shape_2
            vernierlabels_idx[idx_batch] = offset_direction
            nshapeslabels[idx_batch, 0] = selected_repetitions_1
            nshapeslabels[idx_batch, 1] = selected_repetitions_2
            nshapeslabels_idx[idx_batch, 0] = idx_n_shapes_1
            nshapeslabels_idx[idx_batch, 1] = idx_n_shapes_2
            x_shape_1[idx_batch] = col_shape_1_init
            y_shape_1[idx_batch] = row_shape_1
            x_shape_2[idx_batch] = col_shape_2_init
            y_shape_2[idx_batch] = row_shape_2

        # add the color channel for tensorflow:
        shape_1_images = np.expand_dims(shape_1_images, -1)
        shape_2_images = np.expand_dims(shape_2_images, -1)
        return [shape_1_images, shape_2_images, shapelabels_idx, vernierlabels_idx,
                nshapeslabels, nshapeslabels_idx, x_shape_1, y_shape_1, x_shape_2, y_shape_2]

#############################################################
#          HAVE A LOOK AT WHAT THE CODE DOES                #
#############################################################
#from my_parameters import parameters
#imSize = parameters.im_size
##imSize = [16, 48]
#shapeSize = parameters.shape_size
##shapeSize = [14, 11, 6]
#barWidth = parameters.bar_width
#offset = parameters.offset
#n_shapes = parameters.n_shapes
##batch_size = parameters.batch_size
#batch_size = 100
##shape_types = parameters.shape_types
#shape_types = [0, 1, 2, 3, 4]
#crowding_config = [3, 0, 5]
#train_procedure = parameters.train_procedure
#overlap = parameters.overlapping_shapes
#centralize = parameters.centralized_shapes
##reduce_df = parameters.reduce_df
#reduce_df = True
#test = stim_maker_fn(imSize, shapeSize, barWidth, offset)

# plt.imshow(test.drawShape(1))
# test.plotStim([3, 0, 5], offset, 0.01)

#[shape_1_images, shape_2_images, shapelabels_idx, vernierlabels_idx,
#nshapeslabels, nshapeslabels_idx, x_shape_1, y_shape_1, x_shape_2, y_shape_2] = test.makeTrainBatch(
#shape_types, n_shapes, batch_size, train_procedure, overlap, centralize, reduce_df)
#for i in range(batch_size):
#    if train_procedure=='random':
#        plt.imshow(np.squeeze(shape_1_images[i, :, :]))
#    else:
#        plt.imshow(np.squeeze(shape_1_images[i, :, :] + shape_2_images[i, :, :]))
#    plt.pause(0.5)

# [vernier_images, shape_images,  shapelabels_idx, vernierlabels_idx,
# nshapeslabels, nshapeslabels_idx, x_vernier, y_vernier, x_shape, y_shape] = test.makeTestBatch(
# crowding_config, n_shapes, batch_size, None, centralize, reduce_df)
# for i in range(batch_size):
#    plt.imshow(np.squeeze(vernier_images[i, :, :] + shape_images[i, :, :]))
#    plt.pause(0.5)
