# -*- coding: utf-8 -*-
"""
My capsnet: my_batchmaker!
Involving all basic shapes (verniers, squares, circles, polygons, stars)
@author: Lynn

Last update on 11.01.2019
-> added requirements for nshape and location loss
-> nshapeslabels now with index labels [0, len(n_shapes)]
-> added overlapping_shapes parameter
-> lets rather use a rotated square instead of stars, so we can decrease imSize
-> new validation and testing procedures
-> use train_procedures 'vernier_shape', 'random_random' or 'random'
-> implemented a variety of uncrowding stimuli (412+)
-> implemented the possibility to have centralized shapes only
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import draw


##################################
#      stim_maker class fn:      #
##################################
class stim_maker_fn:

    def __init__(self, imSize, shapeSize, barWidth):
        self.imSize    = imSize
        self.shapeSize = shapeSize
        self.barWidth  = barWidth

    
    def drawVernier(self, offset_direction, zoom=0):
        # Inputs:
        # zoom: neg/pos number to de-/increase shape size
        # offset_direction: 0=r, 1=l

        barHeight = int((self.shapeSize+zoom)/4 - (self.barWidth)/4)
        offsetHeight = 1
        # minimum distance between verniers should be one pixel
        if barHeight/2 < 2:
            offset_size = 1
        else:
            offset_size = np.random.randint(1, barHeight/2)
        patch = np.zeros((2*barHeight+offsetHeight, 2*self.barWidth+offset_size), dtype=np.float32)
        patch[0:barHeight, 0:self.barWidth] = 1
        patch[barHeight+offsetHeight:, self.barWidth+offset_size:] = 1
        
        if offset_direction:
            patch = np.fliplr(patch)
        fullPatch = np.zeros((self.shapeSize, self.shapeSize), dtype=np.float32)
        firstRow  = int((self.shapeSize-patch.shape[0])/2)
        firstCol  = int((self.shapeSize-patch.shape[1])/2)
        fullPatch[firstRow:firstRow+patch.shape[0], firstCol:firstCol+patch.shape[1]] = patch
        return fullPatch


    def drawSquare(self, zoom=0):
        # Inputs:
        # zoom: neg/pos number to de-/increase shape size
        zoom = np.abs(zoom)
        patch = np.zeros((self.shapeSize, self.shapeSize), dtype=np.float32)
        patch[zoom:self.barWidth+zoom, zoom:self.shapeSize-zoom] = 1
        patch[zoom:self.shapeSize-zoom, zoom:self.barWidth+zoom] = 1
        patch[self.shapeSize-self.barWidth-zoom:self.shapeSize-zoom, zoom:self.shapeSize-zoom] = 1
        patch[zoom:self.shapeSize-zoom, self.shapeSize-self.barWidth-zoom:self.shapeSize-zoom] = 1
        return patch


    def drawCircle(self, zoom=0, eps=1):
        # Inputs:
        # zoom: neg/pos number to de-/increase shape size
        # eps: you might have to increase this to take care of empty spots in the shape
        patch = np.zeros((self.shapeSize, self.shapeSize), dtype=np.float32)
        radius = (self.shapeSize+zoom)/2
        t = np.linspace(0, np.pi*2, self.shapeSize*4)
        for i in range(1,self.barWidth*eps+1):
            row = np.floor((radius-i/eps) * np.cos(t)+radius - zoom/2)
            col = np.floor((radius-i/eps) * np.sin(t)+radius - zoom/2)
            patch[row.astype(np.int), col.astype(np.int)] = 1
        return patch

    
    def drawPolygon(self, nSides, phi, zoom=0, eps=1):
        # Inputs:
        # nSides: number of sides;
        # phi: angle for rotation;
        # zoom: neg/pos number to de-/increase shape size
        # eps: you might have to increase this to take care of empty spots in the shape
        
        patch = np.zeros((self.shapeSize, self.shapeSize), dtype=np.float32)
        slide_factor = -1  # you might want to move the shape a little to fit the vernier inside
        radius = (self.shapeSize+zoom)/2
        t = np.linspace(0+phi, np.pi*2+phi, nSides+1)
        for i in range(1,self.barWidth*eps+1):
            rowCorner = (np.round((radius-i/eps) * np.sin(t)+radius - zoom/2 + slide_factor))
            colCorner = (np.round((radius-i/eps) * np.cos(t)+radius - zoom/2 + slide_factor))
            for n in range(len(rowCorner)-1):
                rowLines, colLines = draw.line(rowCorner.astype(np.int)[n], colCorner.astype(np.int)[n],
                                               rowCorner.astype(np.int)[n+1], colCorner.astype(np.int)[n+1])
                patch[rowLines, colLines] = 1
        return patch

    
    def drawStar(self, nSides, phi1, phi2, depth, zoom=1, eps=1):
        # Inputs:
        # nSides: number of sides;
        # phi1: angle for rotation of outer corners;
        # phi2: angle for rotation of inner corners;
        # depth: control the distance between inner and outer corners;
        # zoom: neg/pos number to de-/increase shape size
        # eps: you might have to increase this to take care of empty spots in the shape

        patch = np.zeros((self.shapeSize, self.shapeSize), dtype=np.float32)
        slide_factor = -1  # you might want to move the shape a little to fit the vernier inside
        radius_big = (self.shapeSize + zoom)/2
        radius_small = self.shapeSize/depth
        tExt = np.linspace(0+phi1, np.pi*2+phi1, nSides+1)
        tInt = np.linspace(0+phi2, np.pi*2+phi2, nSides+1)
        for i in range(1,self.barWidth*eps+1):
            rowCornerExt = (np.round((radius_big-i/eps) * np.sin(tExt)+radius_big - zoom/2 + slide_factor))
            colCornerExt = (np.round((radius_big-i/eps) * np.cos(tExt)+radius_big - zoom/2 + slide_factor))
            rowCornerInt = (np.round((radius_small-i/eps) * np.sin(tInt)+radius_big - zoom/2 + slide_factor))
            colCornerInt = (np.round((radius_small-i/eps) * np.cos(tInt)+radius_big - zoom/2 + slide_factor))
            for n in range(0, len(rowCornerExt)-1, 1):
                rowLines, colLines = draw.line(rowCornerExt.astype(np.int)[n], colCornerExt.astype(np.int)[n],
                                               rowCornerInt.astype(np.int)[n], colCornerInt.astype(np.int)[n])
                patch[rowLines, colLines] = 1
                rowLines, colLines = draw.line(rowCornerExt.astype(np.int)[n+1], colCornerExt.astype(np.int)[n+1],
                                               rowCornerInt.astype(np.int)[n], colCornerInt.astype(np.int)[n])
                patch[rowLines, colLines] = 1
        return patch
    
    
    def drawStuff(self, n_lines):
        patch = np.zeros((self.shapeSize, self.shapeSize), dtype=np.float32)

        for n in range(n_lines):
            (r1, c1, r2, c2) = np.random.randint(self.shapeSize, size=4)
            rr, cc = draw.line(r1, c1, r2, c2)
            patch[rr, cc] = 1
        return patch

    
    def drawShape(self, shapeID, offset_direction=0):
        if shapeID == 0:
            patch = self.drawVernier(offset_direction, -2)
        if shapeID == 1:
            patch = self.drawSquare(-1)
        if shapeID == 2:
            patch = self.drawCircle(-1)
        if shapeID == 3:
            patch = self.drawPolygon(4, 0)
        if shapeID == 4:
            patch = self.drawStar(4, np.pi/4, np.pi/2, 2.5, 5)
        if shapeID == 5:
            patch = self.drawPolygon(6, 0)
        if shapeID == 6:
            patch = self.drawStuff(5)
#        if shapeID == 7:
#            patch = self.drawStar(6, 0, np.pi/6, 3, 0)
        return patch

    
    def plotStim(self, shape_types, noise=0.):
        '''Visualize all chosen shape_types in one plot'''
        image = np.zeros(self.imSize, dtype=np.float32) + np.random.normal(0, noise, size=self.imSize)
        row = np.random.randint(0, self.imSize[0] - self.shapeSize)
        col = np.random.randint(0, self.imSize[1] - self.shapeSize*(len(shape_types)))
        for i in range(len(shape_types)):
            ID = shape_types[i]
            patch = self.drawShape(shapeID=ID)
            image[row:row+self.shapeSize, col:col+self.shapeSize] += patch
            col += self.shapeSize
        plt.figure()
        plt.imshow(image)
        return

    
    def makeTestBatch(self, selected_shape, n_shapes, batch_size, stim_idx=None, centralize=False):
        '''Create one batch of test dataset according to stim_idx'''
        # Inputs:
        # selected_shape
        # n_shapes: list of shape repetitions
        # batch_size
        # stim_idx: decides whether to create vernier(0), crowding (1) or uncrowding (2) stimulus
        # noise: Random gaussian noise between [0,noise] gets added
        
        # Outputs:
        # vernier_images, shape_images, shapelabels, nshapeslabels, vernierlabels

        vernier_images = np.zeros(shape=[batch_size, self.imSize[0], self.imSize[1]], dtype=np.float32)
        shape_images = np.zeros(shape=[batch_size, self.imSize[0], self.imSize[1]], dtype=np.float32)
        shapelabels = np.zeros(shape=[batch_size, 2], dtype=np.float32)
        vernierlabels = np.zeros(shape=[batch_size, 1], dtype=np.float32)
        nshapeslabels = np.zeros(shape=[batch_size, 1], dtype=np.float32)
        x_vernier = np.zeros(shape=[batch_size, 1], dtype=np.float32)
        y_vernier = np.zeros(shape=[batch_size, 1], dtype=np.float32)
        x_shape = np.zeros(shape=[batch_size, 1], dtype=np.float32)
        y_shape = np.zeros(shape=[batch_size, 1], dtype=np.float32)
        
        for idx_batch in range(batch_size):
            
            vernier_image = np.zeros(self.imSize, dtype=np.float32)
            shape_image = np.zeros(self.imSize, dtype=np.float32)
            if stim_idx is None:
                idx = np.random.randint(0, 3)
            else:
                idx = stim_idx
            
            if centralize:
                # Put each shape in the center of the image:
                row = int((self.imSize[0] - self.shapeSize) / 2)
            else:
                row = np.random.randint(0, self.imSize[0] - self.shapeSize)
            offset_direction = np.random.randint(0, 2)
            vernier_patch = self.drawShape(shapeID=0, offset_direction=offset_direction)

            # the 42-category is for creating squares_stars
            if selected_shape==412:
                shape_patch = self.drawShape(shapeID=1)
                uncrowding_patch = self.drawShape(shapeID=2)
            elif selected_shape==421:
                shape_patch = self.drawShape(shapeID=2)
                uncrowding_patch = self.drawShape(shapeID=1)
            elif selected_shape==413:
                shape_patch = self.drawShape(shapeID=1)
                uncrowding_patch = self.drawShape(shapeID=3)
            elif selected_shape==431:
                shape_patch = self.drawShape(shapeID=3)
                uncrowding_patch = self.drawShape(shapeID=1)
            elif selected_shape==423:
                shape_patch = self.drawShape(shapeID=2)
                uncrowding_patch = self.drawShape(shapeID=3)
            elif selected_shape==432:
                shape_patch = self.drawShape(shapeID=3)
                uncrowding_patch = self.drawShape(shapeID=2)
            else:
                shape_patch = self.drawShape(shapeID=selected_shape)
            
            if idx==0:
                # Vernier test stimuli:
                selected_repetitions = 0
                nshapes_label = 0
                if centralize:
                    # Put each shape in the center of the image:
                    col = int((self.imSize[1] - self.shapeSize) / 2)
                else:
                    col = np.random.randint(0, self.imSize[1] - self.shapeSize)
                vernier_image[row:row+self.shapeSize, col:col+self.shapeSize] += vernier_patch
                x_vernier_ind, y_vernier_ind = col, row
                x_shape_ind, y_shape_ind = col, row

            elif idx==1:
                # Crowded test stimuli:
                selected_repetitions = 1
                nshapes_label = n_shapes.index(selected_repetitions)
                if centralize:
                    # Put each shape in the center of the image:
                    col = int((self.imSize[1] - self.shapeSize) / 2)
                else:
                    col = np.random.randint(0, self.imSize[1] - self.shapeSize)
                vernier_image[row:row+self.shapeSize, col:col+self.shapeSize] += vernier_patch
                shape_image[row:row+self.shapeSize, col:col+self.shapeSize] += shape_patch
                x_vernier_ind, y_vernier_ind = col, row
                x_shape_ind, y_shape_ind = col, row

            elif idx==2:
                # Uncrowded test stimuli:
                selected_repetitions = np.max(n_shapes)
                nshapes_label = n_shapes.index(selected_repetitions)
                if centralize:
                    # Put each shape in the center of the image:
                    col = int((self.imSize[1] - self.shapeSize*selected_repetitions) / 2)
                else:
                    col = np.random.randint(0, self.imSize[1] - self.shapeSize*selected_repetitions)
                x_shape_ind, y_shape_ind = col, row
                
                if (selected_repetitions-1)/2 % 2 == 0:
                    trigger = 0
                else:
                    trigger = 1

                for n_repetitions in range(selected_repetitions):
                    if selected_shape>=400:
                        if n_repetitions == (selected_repetitions-1)/2:
                            vernier_image[row:row+self.shapeSize, col:col+self.shapeSize] += vernier_patch
                            x_vernier_ind, y_vernier_ind = col, row
                        if trigger == 0:
                            shape_image[row:row+self.shapeSize, col:col+self.shapeSize] += shape_patch
                            col += self.shapeSize
                            trigger = 1
                        else:
                            shape_image[row:row+self.shapeSize, col:col+self.shapeSize] += uncrowding_patch
                            col += self.shapeSize
                            trigger = 0

                    else:
                        shape_image[row:row+self.shapeSize, col:col+self.shapeSize] += shape_patch
                        if n_repetitions == (selected_repetitions-1)/2:
                            vernier_image[row:row+self.shapeSize, col:col+self.shapeSize] += vernier_patch
                            x_vernier_ind, y_vernier_ind = col, row
                        col += self.shapeSize

            vernier_images[idx_batch, :, :] = vernier_image #+ np.random.normal(0, noise, size=self.imSize)
            shape_images[idx_batch, :, :] = shape_image #+ np.random.normal(0, noise, size=self.imSize)
            shapelabels[idx_batch, 0] = 0
            shapelabels[idx_batch, 1] = selected_shape 
            nshapeslabels[idx_batch] = nshapes_label
            vernierlabels[idx_batch] = offset_direction
            x_vernier[idx_batch] = x_vernier_ind
            y_vernier[idx_batch] = y_vernier_ind
            x_shape[idx_batch] = x_shape_ind
            y_shape[idx_batch] = y_shape_ind

        # add the color channel for tensorflow:
        vernier_images = np.expand_dims(vernier_images, -1)
        shape_images = np.expand_dims(shape_images, -1)
        return vernier_images, shape_images, shapelabels, vernierlabels, nshapeslabels, x_vernier, y_vernier, x_shape, y_shape


    def makeTrainBatch(self, shape_types, n_shapes, batch_size, train_procedure='vernier_shape',
                       overlap=False, centralize=False):
        '''Create one batch of training dataset with each one vernier and
        one shape_type repeated n_shapes[random] times'''

        shape_1_images = np.zeros(shape=[batch_size, self.imSize[0], self.imSize[1]], dtype=np.float32)
        shape_2_images = np.zeros(shape=[batch_size, self.imSize[0], self.imSize[1]], dtype=np.float32)
        shapelabels = np.zeros(shape=[batch_size, 2], dtype=np.float32)
        vernierlabels = np.zeros(shape=[batch_size, 1], dtype=np.float32)
        nshapeslabels = np.zeros(shape=[batch_size, 2], dtype=np.float32)
        x_shape_1 = np.zeros(shape=[batch_size, 1], dtype=np.float32)
        y_shape_1 = np.zeros(shape=[batch_size, 1], dtype=np.float32)
        x_shape_2 = np.zeros(shape=[batch_size, 1], dtype=np.float32)
        y_shape_2 = np.zeros(shape=[batch_size, 1], dtype=np.float32)

        for idx_batch in range(batch_size):
            shape_1_image = np.zeros(self.imSize, dtype=np.float32)
            shape_2_image = np.zeros(self.imSize, dtype=np.float32)

            try:
                # choose shape_type(s) based on train_procedure
                if train_procedure=='vernier_shape':
                    selected_shape_1 = 0
                    selected_shape_2 = np.random.randint(1, len(shape_types))
                elif train_procedure=='random_random' or 'random':
                    # I want every second image with a vernier:
                    if np.random.rand(1)<0.5:
                        selected_shape_1 = 0
                    else:
                        selected_shape_1 = np.random.randint(0, len(shape_types))
                    selected_shape_2 = np.random.randint(1, len(shape_types))
                else:
                    raise SystemExit('\nThe chosen train_procedure is unknown!\n')

            except:
                # if len(shape_types)=1, then just use this given shape_type
                selected_shape_1 = 0
                selected_shape_2 = shape_types

            
            # Create shape images:
            if selected_shape_1==0:
                # if the first shape is a vernier, only repeat once and use offset_direction 0=r or 1=l
                selected_repetitions_1 = 1
                idx_n_shapes_1 = n_shapes.index(selected_repetitions_1)
                offset_direction = np.random.randint(0, 2)
                shape_1_patch = self.drawShape(shapeID=selected_shape_1, offset_direction=offset_direction)
            else:
                # if not, repeat shape random times but at least once and set offset_direction to 2=no vernier
                idx_n_shapes_1 = np.random.randint(1, len(n_shapes))
                selected_repetitions_1 = n_shapes[idx_n_shapes_1]
                offset_direction = 2
                shape_1_patch = self.drawShape(shapeID=selected_shape_1)

            idx_n_shapes_2 = np.random.randint(0, len(n_shapes))
            selected_repetitions_2 = n_shapes[idx_n_shapes_2]
            shape_2_patch = self.drawShape(shapeID=selected_shape_2)

            if centralize:
                # Put each shape in the center of the image:
                row_shape_1 = int((self.imSize[0] - self.shapeSize) / 2)
                col_shape_1_init = int((self.imSize[1] - self.shapeSize*selected_repetitions_1) / 2)
                col_shape_1 = col_shape_1_init
                row_shape_2 = int((self.imSize[0] - self.shapeSize) / 2)
                col_shape_2_init = int((self.imSize[1] - self.shapeSize*selected_repetitions_2) / 2)
                col_shape_2 = col_shape_2_init
            else:
                row_shape_1 = np.random.randint(0, self.imSize[0] - self.shapeSize)
                col_shape_1_init = np.random.randint(0, self.imSize[1] - self.shapeSize*selected_repetitions_1)
                col_shape_1 = col_shape_1_init
                row_shape_2 = np.random.randint(0, self.imSize[0] - self.shapeSize)
                col_shape_2_init = np.random.randint(0, self.imSize[1] - self.shapeSize*selected_repetitions_2)
                col_shape_2 = col_shape_2_init
            
            # Repeat shape_1 selected_repetitions times if not vernier:
            if selected_shape_1!=0:
                for i in range(selected_repetitions_1):
                    shape_1_image[row_shape_1:row_shape_1+self.shapeSize,
                                  col_shape_1:col_shape_1+self.shapeSize] += shape_1_patch
                    col_shape_1 += self.shapeSize
            
            # Repeat shape_2 selected_repetitions times:
            for i in range(selected_repetitions_2):
                shape_2_image[row_shape_2:row_shape_2+self.shapeSize,
                              col_shape_2:col_shape_2+self.shapeSize] += shape_2_patch
                col_shape_2 += self.shapeSize

            # If vernier, do we allow for overlap between vernier and shape image?
            if selected_shape_1==0:
                if overlap or centralize:
                        shape_1_image[row_shape_1:row_shape_1+self.shapeSize,
                                      col_shape_1:col_shape_1+self.shapeSize] += shape_1_patch
                else:
                    while np.sum(shape_2_image[row_shape_1:row_shape_1+self.shapeSize,
                                               col_shape_1:col_shape_1+self.shapeSize] + shape_1_patch) > np.sum(shape_1_patch):
                        row_shape_1 = np.random.randint(0, self.imSize[0] - self.shapeSize)
                        col_shape_1 = np.random.randint(0, self.imSize[1] - self.shapeSize)
                    shape_1_image[row_shape_1:row_shape_1+self.shapeSize,
                                  col_shape_1:col_shape_1+self.shapeSize] += shape_1_patch
                    col_shape_1_init = col_shape_1
            

            shape_1_images[idx_batch, :, :] = shape_1_image
            shape_2_images[idx_batch, :, :] = shape_2_image
            shapelabels[idx_batch, 0] = selected_shape_1
            shapelabels[idx_batch, 1] = selected_shape_2
            vernierlabels[idx_batch] = offset_direction
            nshapeslabels[idx_batch, 0] = idx_n_shapes_1
            nshapeslabels[idx_batch, 1] = idx_n_shapes_2
            x_shape_1[idx_batch] = col_shape_1_init
            y_shape_1[idx_batch] = row_shape_1
            x_shape_2[idx_batch] = col_shape_2_init
            y_shape_2[idx_batch] = row_shape_2

        # add the color channel for tensorflow:
        shape_1_images = np.expand_dims(shape_1_images, -1)
        shape_2_images = np.expand_dims(shape_2_images, -1)
        return [shape_1_images, shape_2_images, shapelabels, vernierlabels,
                nshapeslabels, x_shape_1, y_shape_1, x_shape_2, y_shape_2]


#############################################################
#          HAVE A LOOK AT WHAT THE CODE DOES                #
#############################################################
#from my_parameters import parameters
#imSize = parameters.im_size
##imSize = [20, 75]
#shapeSize = parameters.shape_size
##shapeSize = 14
#barWidth = parameters.bar_width
#n_shapes = parameters.n_shapes
##n_shapes = [1, 3, 5]
##batch_size = parameters.batch_size
#batch_size = 10
#shape_types = parameters.shape_types
##shape_types = [0, 1, 2, 3]
#train_procedure = parameters.train_procedure
##train_procedure = 'random'
#overlap = parameters.overlapping_shapes
##overlap = True
#centralize = parameters.centralized_shapes
##centralize = True
#test = stim_maker_fn(imSize, shapeSize, barWidth)

#plt.imshow(test.drawShape(5))
#test.plotStim([1, 2, 4, 5], 0.01)

#[shape_1_images, shape_2_images, shapelabels, vernierlabels,
# nshapeslabels, x_shape_1, y_shape_1, x_shape_2, y_shape_2] = test.makeTrainBatch(
# shape_types, n_shapes, batch_size, train_procedure, overlap=overlap, centralize=centralize)
#for i in range(batch_size):
#    plt.imshow(np.squeeze(shape_1_images[i, :, :] + shape_2_images[i, :, :]))
#    plt.pause(0.5)

#[vernier_images, shape_images, shapelabels, vernierlabels,
# nshapeslabels, x_vernier, y_vernier, x_shape, y_shape] = test.makeTestBatch(3, n_shapes, batch_size, None, centralize)
#for i in range(batch_size):
#    plt.imshow(np.squeeze(vernier_images[i, :, :] + shape_images[i, :, :]))
#    plt.pause(0.5)
