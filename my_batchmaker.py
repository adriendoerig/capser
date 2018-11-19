# -*- coding: utf-8 -*-
"""
My capsnet: my_batchmaker!
Involving all basic shapes (verniers, squares, circles, polygons, stars)
@author: Lynn

Last update on 15.11.2018
-> added requirements for nshape and location loss
-> nshapeslabels now with index labels [0, len(n_shapes)]
-> added overlapping_shapes parameter
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import draw


##################################
#      stim_maker class fn:      #
##################################
class stim_maker_fn:

    def __init__(self, imSize, shapeSize, barWidth):
        print('-------------------------------------------------------')
        print('Creation of the StimMaker class')
        print('-------------------------------------------------------')
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
        
        patch  = np.zeros((self.shapeSize, self.shapeSize), dtype=np.float32)
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

        patch  = np.zeros((self.shapeSize, self.shapeSize), dtype=np.float32)
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

    
    def drawShape(self, shapeID, offset_direction=0):
        if shapeID == 0:
            patch = self.drawVernier(offset_direction, -2)
        if shapeID == 1:
            patch = self.drawSquare(-1)
        if shapeID == 2:
            patch = self.drawCircle(-1)
        if shapeID == 3:
            patch = self.drawPolygon(6, 0)
        if shapeID == 4:
            patch = self.drawStar(4, np.pi/4, np.pi/2, 2.8, 6)
        if shapeID == 5:
            patch = self.drawStar(6, 0, np.pi/6, 3, 0)
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

    
    def makeTestBatch(self, selected_shape, n_shapes, batch_size, stim_idx=None, noise=0.):
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
        x_shape = np.zeros(shape=[batch_size, 1], dtype=np.float32)
        y_shape = np.zeros(shape=[batch_size, 1], dtype=np.float32)
        x_vernier = np.zeros(shape=[batch_size, 1], dtype=np.float32)
        y_vernier = np.zeros(shape=[batch_size, 1], dtype=np.float32)
        
        for idx_batch in range(batch_size):
            
            vernier_image = np.zeros(self.imSize, dtype=np.float32)
            shape_image = np.zeros(self.imSize, dtype=np.float32)
            if stim_idx is None:
                idx = np.random.randint(0, 3)
            else:
                idx = stim_idx
            
            row = np.random.randint(0, self.imSize[0] - self.shapeSize)
            offset_direction = np.random.randint(0, 2)
            vernier_patch = self.drawShape(shapeID=0, offset_direction=offset_direction)

            # the 42-category is for creating squares_stars
            if selected_shape==42:
                shape_patch = self.drawShape(shapeID=1)
                star_patch = self.drawShape(shapeID=5)
            else:
                shape_patch = self.drawShape(shapeID=selected_shape)
            
            if idx==0:
                # Vernier test stimuli:
                selected_repetitions = 0
                nshapes_label = 0  # actually, this should not have a value but since we only use this for training, its ok!
                col = np.random.randint(0, self.imSize[1] - self.shapeSize)
                vernier_image[row:row+self.shapeSize, col:col+self.shapeSize] += vernier_patch
                x_vernier_ind, y_vernier_ind = col, row
                x_shape_ind, y_shape_ind = col, row

            elif idx==1:
                # Crowded test stimuli:
                selected_repetitions = 1
                nshapes_label = n_shapes.index(selected_repetitions)
                col = np.random.randint(0, self.imSize[1] - self.shapeSize)
                vernier_image[row:row+self.shapeSize, col:col+self.shapeSize] += vernier_patch
                shape_image[row:row+self.shapeSize, col:col+self.shapeSize] += shape_patch
                x_vernier_ind, y_vernier_ind = col, row
                x_shape_ind, y_shape_ind = col, row

            elif idx==2:
                # Uncrowded test stimuli:
                selected_repetitions = np.max(n_shapes)
                nshapes_label = n_shapes.index(selected_repetitions)
                col = np.random.randint(0, self.imSize[1] - self.shapeSize*selected_repetitions)
                x_shape_ind, y_shape_ind = col, row
                
                if (selected_repetitions-1)/2 % 2 == 0:
                    trigger = 0
                else:
                    trigger = 1

                for n_repetitions in range(selected_repetitions):
                    if selected_shape==42:
                        if n_repetitions == (selected_repetitions-1)/2:
                            vernier_image[row:row+self.shapeSize, col:col+self.shapeSize] += vernier_patch
                            x_vernier_ind, y_vernier_ind = col, row
                        if trigger == 0:
                            shape_image[row:row+self.shapeSize, col:col+self.shapeSize] += shape_patch
                            col += self.shapeSize
                            trigger = 1
                        else:
                            shape_image[row:row+self.shapeSize, col:col+self.shapeSize] += star_patch
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
            x_shape[idx_batch] = x_shape_ind
            y_shape[idx_batch] = y_shape_ind
            x_vernier[idx_batch] = x_vernier_ind
            y_vernier[idx_batch] = y_vernier_ind

        # add the color channel for tensorflow:
        vernier_images = np.expand_dims(vernier_images, -1)
        shape_images = np.expand_dims(shape_images, -1)
        return vernier_images, shape_images, shapelabels, nshapeslabels, vernierlabels, x_shape, y_shape, x_vernier, y_vernier


    def makeTrainBatch(self, shape_types, n_shapes, batch_size, noise=0., overlap=None):
        '''Create one batch of training dataset with each one vernier and
        one shape_type repeated n_shapes[random] times'''
        # Inputs: 
        # shape_types: one of these shape_type (ID>0) gets randomly chosen;
        # n_shapes: one of the listed repetitions gets randomly chosen;
        # batch_size
        # noise: Random gaussian noise between [0,noise] gets added
        # overlap: if True, the verniers can also be placed in/on the shapes
        
        # Outputs:
        # vernier_images, shape_images, shapelabels, nshapeslabels, vernierlabels

        vernier_images = np.zeros(shape=[batch_size, self.imSize[0], self.imSize[1]], dtype=np.float32)
        shape_images = np.zeros(shape=[batch_size, self.imSize[0], self.imSize[1]], dtype=np.float32)
        shapelabels = np.zeros(shape=[batch_size, 2], dtype=np.float32)
        vernierlabels = np.zeros(shape=[batch_size, 1], dtype=np.float32)
        nshapeslabels = np.zeros(shape=[batch_size, 1], dtype=np.float32)
        x_shape = np.zeros(shape=[batch_size, 1], dtype=np.float32)
        y_shape = np.zeros(shape=[batch_size, 1], dtype=np.float32)
        x_vernier = np.zeros(shape=[batch_size, 1], dtype=np.float32)
        y_vernier = np.zeros(shape=[batch_size, 1], dtype=np.float32)

        for idx_batch in range(batch_size):
            vernier_image = np.zeros(self.imSize, dtype=np.float32)
            shape_image = np.zeros(self.imSize, dtype=np.float32)

            # randomly choose one of the selected shape_types and the number of shape repetitions
            idx_shape_type = np.random.randint(1, len(shape_types))
            selected_shape = shape_types[idx_shape_type]
            idx_n_shapes = np.random.randint(0, len(n_shapes))
            selected_repetitions = n_shapes[idx_n_shapes]
            
            # Create shape image
            row_shape = np.random.randint(0, self.imSize[0] - self.shapeSize)
            col_shape = np.random.randint(0, self.imSize[1] - self.shapeSize*selected_repetitions)
            x_shape_ind, y_shape_ind = col_shape, row_shape
            shape_patch = self.drawShape(shapeID=selected_shape)
            for n_repetitions in range(selected_repetitions):
                shape_image[row_shape:row_shape+self.shapeSize, col_shape:col_shape+self.shapeSize] += shape_patch
                col_shape += self.shapeSize

            
            # Create vernier image
            row_vernier = np.random.randint(0, self.imSize[0] - self.shapeSize)
            col_vernier = np.random.randint(0, self.imSize[1] - self.shapeSize)
            
            offset_direction = np.random.randint(0, 2)
            vernier_patch = self.drawShape(shapeID=0, offset_direction=offset_direction)

            # Do we allow for overlap between vernier and shape image?
            if overlap:
                vernier_image[row_vernier:row_vernier+self.shapeSize, col_vernier:col_vernier+self.shapeSize] += vernier_patch
            else:
                while np.sum(shape_image[row_vernier:row_vernier+self.shapeSize, col_vernier:col_vernier+self.shapeSize] + vernier_patch) > np.sum(vernier_patch):
                    row_vernier = np.random.randint(0, self.imSize[0] - self.shapeSize)
                    col_vernier = np.random.randint(0, self.imSize[1] - self.shapeSize)
                vernier_image[row_vernier:row_vernier+self.shapeSize, col_vernier:col_vernier+self.shapeSize] += vernier_patch
            
            x_vernier_ind, y_vernier_ind = col_vernier, row_vernier
            
            vernier_images[idx_batch, :, :] = vernier_image #+ np.random.normal(0, noise, size=self.imSize)
            shape_images[idx_batch, :, :] = shape_image #+ np.random.normal(0, noise, size=self.imSize)
            shapelabels[idx_batch, 0] = 0
            shapelabels[idx_batch, 1] = selected_shape
            vernierlabels[idx_batch] = offset_direction
            nshapeslabels[idx_batch] = idx_n_shapes
            x_shape[idx_batch] = x_shape_ind
            y_shape[idx_batch] = y_shape_ind
            x_vernier[idx_batch] = x_vernier_ind
            y_vernier[idx_batch] = y_vernier_ind

        # add the color channel for tensorflow:
        vernier_images = np.expand_dims(vernier_images, -1)
        shape_images = np.expand_dims(shape_images, -1)
        return vernier_images, shape_images, shapelabels, nshapeslabels, vernierlabels, x_shape, y_shape, x_vernier, y_vernier


#############################################################
#          HAVE A LOOK AT WHAT THE CODE DOES                #
#############################################################

#from my_parameters import parameters
#imSize = parameters.im_size
#shapeSize = parameters.shape_size
#barWidth = parameters.bar_width
#n_shapes = parameters.n_shapes
#train_noise = parameters.train_noise
#test_noise = parameters.test_noise
#batch_size = parameters.batch_size
#shape_types = parameters.shape_types
#overlap = parameters.overlapping_shapes
#test = stim_maker_fn(imSize, shapeSize, barWidth)

#plt.imshow(test.drawShape(3))
#test.plotStim([1, 2, 3, 4, 5], 0.05)

#[train_vernier_images, train_shape_images, train_shapelabels, train_nshapeslabels, 
#train_vernierlabels, train_x_shape, train_y_shape, train_x_vernier, train_y_vernier] = test.makeTrainBatch(
#        shape_types, n_shapes, batch_size, train_noise, overlap=overlap)
#for i in range(batch_size):
#    plt.imshow(np.squeeze(train_vernier_images[i, :, :] + train_shape_images[i, :, :]))
#    plt.pause(0.5)

#[test_vernier_images, test_shape_images, test_shapelabels, test_nshapeslabels, 
#test_vernierlabels, test_x_shape, test_y_shape, test_x_vernier, test_y_vernier] = test.makeTestBatch(42, n_shapes, batch_size, None, test_noise)
#for i in range(batch_size):
#    plt.imshow(np.squeeze(test_vernier_images[i, :, :] + test_shape_images[i, :, :]))
#    plt.pause(0.5)

