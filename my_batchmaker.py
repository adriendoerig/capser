# -*- coding: utf-8 -*-
"""
My capsnet: my_batchmaker!
Version 1
Created on Fri Oct  5 11:00:15 2018
@author: Lynn
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

    
    def drawVernier(self, zoom=0, offset=None):
        '''Create a vernier patch of size shapesize
        
        Input:
            zoom:
                negative or positive number to decrease or increase vernier size
            offset:
                if True, also return the offset direction (0=r, 1=l)'''
        barHeight = int((self.shapeSize+zoom)/4 - (self.barWidth)/4)
        offsetHeight = 1
        # minimum distance between verniers should be one pixel
        if barHeight/2 < 2:
            offset_size = 1
        else:
            offset_size = np.random.randint(1, barHeight/2)
        patch = np.zeros((2*barHeight+offsetHeight, 2*self.barWidth+offset_size))
        patch[0:barHeight, 0:self.barWidth] = 1
        patch[barHeight+offsetHeight:, self.barWidth+offset_size:] = 1
        # randomly create flipped verniers
        offset_direction = np.random.randint(0, 2)
        if offset_direction:
            patch = np.fliplr(patch)
        fullPatch = np.zeros((self.shapeSize, self.shapeSize))
        firstRow  = int((self.shapeSize-patch.shape[0])/2)
        firstCol  = int((self.shapeSize-patch.shape[1])/2)
        fullPatch[firstRow:firstRow+patch.shape[0], firstCol:firstCol+patch.shape[1]] = patch
        if offset:
            return fullPatch, offset_direction
        else:
            return fullPatch


    def drawSquare(self, zoom=0):
        zoom = np.abs(zoom)
        patch = np.zeros((self.shapeSize, self.shapeSize))
        patch[zoom:self.barWidth+zoom, zoom:self.shapeSize-zoom] = 1
        patch[zoom:self.shapeSize-zoom, zoom:self.barWidth+zoom] = 1
        patch[self.shapeSize-self.barWidth-zoom:self.shapeSize-zoom, zoom:self.shapeSize-zoom] = 1
        patch[zoom:self.shapeSize-zoom, self.shapeSize-self.barWidth-zoom:self.shapeSize-zoom] = 1
        return patch


    def drawCircle(self, zoom=0, eps=1):
        patch = np.zeros((self.shapeSize, self.shapeSize))
        radius = (self.shapeSize+zoom)/2
        t = np.linspace(0, np.pi*2, self.shapeSize*4)
        for i in range(1,self.barWidth*eps+1):
            row = np.floor((radius-i/eps) * np.cos(t)+radius - zoom/2)
            col = np.floor((radius-i/eps) * np.sin(t)+radius - zoom/2)
            patch[row.astype(np.int), col.astype(np.int)] = 1
        return patch

    
    def drawPolygon(self, nSides, phi, zoom=0, eps=1):
        '''Function to draw a polygon with nSides
        
        Inputs:
            nSides:
                number of sides;
            phi:
                angle for rotation;
            zoom:
                increase or decrease size of the shape;
            eps:
                if shape size gets increased, you can increase eps to take care of empty spots'''
        patch  = np.zeros((self.shapeSize, self.shapeSize))
        # if the stimulus does not fill the whole patch, u can move it a little
        # Might be important to have the vernier in the middle for the test stimuli:
        slide_factor = -1
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
        '''Function to draw a star with nSides
        
        Inputs:
            nSides:
                number of sides;
            phi1:
                angle for rotation of outer corners;
            phi2:
                angle for rotation of inner corners;
            depth:
                control the distance between inner and outer corners;
            zoom:
                increase or decrease size of the shape;
            eps:
                if shapesize gets increased, you can increase eps to take care of empty spots'''
        patch  = np.zeros((self.shapeSize, self.shapeSize))
        # if the stimulus does not fill the whole patch, u can move it a little
        # Might be important to have the vernier in the middle for the test stimuli:
        slide_factor = -1
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

    
    def drawShape(self, shapeID, offset=None):
        vernier_zoom = -2
        if offset and shapeID == 0:
            patch, offset_direction = self.drawVernier(zoom=vernier_zoom, offset=True)
            return patch, offset_direction
        else:
            if shapeID == 0:
                patch = self.drawVernier(zoom=vernier_zoom)
            if shapeID == 1:
                patch = self.drawSquare(zoom=-1)
            if shapeID == 2:
                patch = self.drawCircle(zoom=-1)
            if shapeID == 3:
                patch = self.drawPolygon(6, 0)
            if shapeID == 4:
                patch = self.drawStar(4, np.pi/4, np.pi/2, 2.8, zoom=6)
            if shapeID == 5:
                patch = self.drawStar(6, 0, np.pi/6, 3, zoom=0)
            return patch

    
    def plotStim(self, shape_types, noise=0.):
        '''Visualize all chosen shape_types in one plot'''
        image = np.zeros(self.imSize) + np.random.normal(0, noise, size=self.imSize)
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

    
    def makeTestBatch(self, selected_shape, n_shapes, batch_size, noise=0.):
        '''Create one batch of test dataset with one vernier in the shape in
        the middle of the n_shapes[random] times repeated selected_shape
        
        Inputs:
            selected_shape:
                the shape you want to use (ID>0);
            n_shapes:
                one of the listed repetitions gets randomly chosen;
            batch_size:
                size of each batch;
            noise:
                Random gaussian noise between [0,noise] gets added
        
        Outputs:
            image_batch:
                matrix of shape [batch_size, imSize[0], imSize[1], 1] involving
                the creaed image test dataset
            vernierlabels_batch:
                matrix of shape [batch_size, 2] involving the offset direction
                as one hot encoded (1st col = r; 2nd col = l)'''
        image_batch = np.zeros(shape=[batch_size, self.imSize[0], self.imSize[1]])
        labels_batch = np.zeros(shape=[batch_size, 1])
        nshapes_batch = np.zeros(shape=[batch_size, 1])
        vernierlabels_batch = np.zeros(shape=[batch_size, 1])
        
        for idx_batch in range(batch_size):
            
            image = np.zeros(self.imSize)
            # choose the number of shape repetitions and randomly whether to
            # create vernier, crowded or uncrowded test stimuli
            idx = np.random.randint(0, 3)
            
            row = np.random.randint(0, self.imSize[0] - self.shapeSize)
            vernier_patch, offset_direction = self.drawShape(shapeID=0, offset=True)
            # the 42-category is for creating squares_stars
            if selected_shape==42:
                shape_patch = self.drawShape(shapeID=1)
                star_patch = self.drawShape(shapeID=5)
            else:
                shape_patch = self.drawShape(shapeID=selected_shape)
            
            if idx==0:
                # Vernier test stimuli:
                selected_repetitions = 0
                col = np.random.randint(0, self.imSize[1] - self.shapeSize)
                image[row:row+self.shapeSize, col:col+self.shapeSize] += vernier_patch

            elif idx==1:
                # Crowded test stimuli:
                selected_repetitions = np.min(n_shapes)
                col = np.random.randint(0, self.imSize[1] - self.shapeSize)
                full_patch = vernier_patch + shape_patch
                image[row:row+self.shapeSize, col:col+self.shapeSize] += full_patch

            elif idx==2:
                # Uncrowded test stimuli:
                selected_repetitions = np.max(n_shapes)
                col = np.random.randint(0, self.imSize[1] - self.shapeSize*selected_repetitions)
                if (selected_repetitions-1)/2 % 2 == 0:
                    trigger = 0
                else:
                    trigger = 1

                for n_repetitions in range(selected_repetitions):
                    if selected_shape==42:
                        if n_repetitions == (selected_repetitions-1)/2:
                            image[row:row+self.shapeSize, col:col+self.shapeSize] += vernier_patch
                        if trigger == 0:
                            image[row:row+self.shapeSize, col:col+self.shapeSize] += shape_patch
                            col += self.shapeSize
                            trigger = 1
                        else:
                            image[row:row+self.shapeSize, col:col+self.shapeSize] += star_patch
                            col += self.shapeSize
                            trigger = 0

                    else:
                        image[row:row+self.shapeSize, col:col+self.shapeSize] += shape_patch
                        if n_repetitions == (selected_repetitions-1)/2:
                            image[row:row+self.shapeSize, col:col+self.shapeSize] += vernier_patch
                        col += self.shapeSize

            image_batch[idx_batch, :, :] = image + np.random.normal(0, noise, size=self.imSize)
            labels_batch[idx_batch] = selected_shape
            nshapes_batch[idx_batch] = selected_repetitions
            vernierlabels_batch[idx_batch] = offset_direction

        # add the color channel for tensorflow:
        image_batch = np.expand_dims(image_batch, -1)
        return image_batch, labels_batch, nshapes_batch, vernierlabels_batch


    def makeTrainBatch(self, shape_types, n_shapes, batch_size, noise=0., overlap=None):
        '''Create one batch of training dataset with each one vernier and
        one shape_type repeated n_shapes[random] times
        
        Inputs:
            shape_types:
                one of these shape_type (ID>0) gets randomly chosen;
            n_shapes:
                one of the listed repetitions gets randomly chosen;
            batch_size:
                size of each batch;
            noise:
                Random gaussian noise between [0,noise] gets added
            overlap:
                if True, the verniers can also be placed in/on the shapes
        
        Outputs:
            image_batch:
                matrix of shape [batch_size, imSize[0], imSize[1], 1] involving
                the creaed image training dataset
            labels_batch:
                matrix of shape [batch_size, len(shape_types)] involving the
                shape label as one hot encoded'''
        image_batch = np.zeros(shape=[batch_size, self.imSize[0], self.imSize[1]])
        labels_batch = np.zeros(shape=[batch_size, 1])
        nshapes_batch = np.zeros(shape=[batch_size, 1])
        vernierlabels_batch = np.zeros(shape=[batch_size, 1])
        

        for idx_batch in range(batch_size):
            image = np.zeros(self.imSize)

            # randomly choose one of the selected shape_types and the number
            # of shape repetitions
            idx_shape_type = np.random.randint(1, len(shape_types))
            selected_shape = shape_types[idx_shape_type]
            idx_n_shapes = np.random.randint(0, len(n_shapes))
            selected_repetitions = n_shapes[idx_n_shapes]
            
            row = np.random.randint(0, self.imSize[0] - self.shapeSize)
            col = np.random.randint(0, self.imSize[1] - self.shapeSize*selected_repetitions)
            shape_patch = self.drawShape(shapeID=selected_shape)
            for n_repetitions in range(selected_repetitions):
                image[row:row+self.shapeSize, col:col+self.shapeSize] += shape_patch
                col += self.shapeSize
            
            row = np.random.randint(0, self.imSize[0] - self.shapeSize)
            col = np.random.randint(0, self.imSize[1] - self.shapeSize)
            vernier_patch, offset_direction = self.drawShape(shapeID=0, offset=True)
            if overlap:
                image[row:row+self.shapeSize, col:col+self.shapeSize] += vernier_patch
            else:
                while np.sum(image[row:row+self.shapeSize, col:col+self.shapeSize] + vernier_patch) > np.sum(vernier_patch):
                    row = np.random.randint(0, self.imSize[0] - self.shapeSize)
                    col = np.random.randint(0, self.imSize[1] - self.shapeSize)
                image[row:row+self.shapeSize, col:col+self.shapeSize] += vernier_patch
            
            image_batch[idx_batch, :, :] = image + np.random.normal(0, noise, size=self.imSize)
            labels_batch[idx_batch] = selected_shape
            nshapes_batch[idx_batch] = selected_repetitions
            vernierlabels_batch[idx_batch] = offset_direction

        # add the color channel for tensorflow:
        image_batch = np.expand_dims(image_batch, -1)
        return image_batch, labels_batch, nshapes_batch, vernierlabels_batch


#############################################################
#           JUST SOME TRIALS AND CHECKING                   #
#############################################################

#imSize = [60, 150]
#shapeSize = 20
#barWidth = 1
#n_shapes = [1, 3, 5, 7]
#noise = 0.05
#batch_size = 10
#shape_types = [0, 1, 2, 3, 4, 5]
#test = stim_maker_fn(imSize, shapeSize, barWidth)

#plt.imshow(test.drawShape(1))
#test.plotStim([1, 2, 3, 4, 5], 0.05)

#train_image_batch, train_labels_batch, train_nshapes_batch, train_vernierlabels_batch = test.makeTrainBatch(
#        shape_types, n_shapes, batch_size, noise, overlap=None)
#for i in range(batch_size):
#    plt.imshow(np.squeeze(train_image_batch[i, :, :]))
#    plt.pause(0.5)

#test_image_batch, test_labels_batch, test_nshapes_batch, test_vernierlabels_batch = test.makeTestBatch(42, n_shapes, batch_size, noise)
#for i in range(batch_size):
#    plt.imshow(np.squeeze(test_image_batch[i, :, :]))
#    plt.pause(0.5)
