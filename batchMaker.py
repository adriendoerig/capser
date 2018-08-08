
# Class to make a batch

import numpy, random, matplotlib.pyplot as plt
from skimage import draw
from scipy.ndimage import zoom
from datetime import datetime
from parameters import shape_types, simultaneous_shapes, noise_level, max_cols, max_rows, vernier_normalization_exp, random_pixels


def clipped_zoom(img, zoom_factor, **kwargs):

    h, w = img.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(numpy.round(h * zoom_factor))
        zw = int(numpy.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        # Zero-padding
        out = numpy.zeros_like(img)
        out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)

    # Zooming in
    elif zoom_factor > 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(numpy.round(h / zoom_factor))
        zw = int(numpy.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = zoom(img[top:top + zh, left:left + zw], zoom_tuple, **kwargs)

        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top + h, trim_left:trim_left + w]

        # If zoom_factor == 1, just return the input array
    else:
        out = img
    return out


def computeMeanAndStd(self, shapeTypes, batchSize=100, n_shapes=1, noiseLevel=0.0, max_rows=1, max_cols=3):
    # compute mean and std over a large batch. May be used to apply the same normalization to all images (cf. make_tf_dataset.py)

    batchImages = numpy.zeros(shape=(batchSize, self.imSize[0], self.imSize[1]), dtype=numpy.float32)
    batchSingleShapeImages = numpy.zeros(shape=(batchSize, self.imSize[0], self.imSize[1], n_shapes),
                                         dtype=numpy.float32)

    for n in range(batchSize):
        shapes = numpy.random.permutation(len(shapeTypes))[:n_shapes]

        for shape in range(n_shapes):
            if shapes[shape] == 0:  # 1/len(shapeTypes):
                thisOffset = random.randint(0, 1)
                batchSingleShapeImages[n, :, :, shape] = self.drawStim(False, shapeMatrix=[0], offset=thisOffset, offset_size=random.randint(1, int(self.barHeight / 2.0))) + numpy.random.normal(0, noiseLevel, size=self.imSize)
                batchImages[n, :, :] += batchSingleShapeImages[n, :, :, shape]

            else:
                thisType = shapes[shape]
                shapeType = shapeTypes[thisType]
                nRows = random.randint(1, max_rows)
                nCols = random.randint(1, max_cols)
                shapeConfig = shapeType * numpy.ones((nRows, nCols))
                batchSingleShapeImages[n, :, :, shape] = self.drawStim(0, shapeConfig) + numpy.random.normal(0, noiseLevel, size=self.imSize)
                batchImages[n, :, :] += batchSingleShapeImages[n, :, :, shape]

    batchMean = numpy.mean(batchImages)
    batchStd = numpy.std(batchImages)

    return batchMean, batchStd


class StimMaker:

    def __init__(self, imSize, shapeSize, barWidth):

        self.imSize    = imSize
        self.shapeSize = shapeSize
        self.barWidth  = barWidth
        self.barHeight = int(shapeSize/4-barWidth/4)
        self.offsetHeight = 1
        self.mean, self.std = computeMeanAndStd(self, shape_types, batchSize=100, n_shapes=simultaneous_shapes, noiseLevel=noise_level, max_rows=max_rows, max_cols=max_cols)


    def setShapeSize(self, shapeSize):

        self.shapeSize = shapeSize


    def drawSquare(self):

        resizeFactor = 1.2
        patch = numpy.zeros((self.shapeSize, self.shapeSize))

        firstRow = int((self.shapeSize - self.shapeSize/resizeFactor)/2)
        firstCol = firstRow
        sideSize = int(self.shapeSize/resizeFactor)

        patch[firstRow         :firstRow+self.barWidth,          firstCol:firstCol+sideSize+self.barWidth] = random.uniform(1-random_pixels, 1+random_pixels)
        patch[firstRow+sideSize:firstRow+self.barWidth+sideSize, firstCol:firstCol+sideSize+self.barWidth] = random.uniform(1-random_pixels, 1+random_pixels)
        patch[firstRow:firstRow+sideSize+self.barWidth, firstCol         :firstCol+self.barWidth         ] = random.uniform(1-random_pixels, 1+random_pixels)
        patch[firstRow:firstRow+sideSize+self.barWidth, firstRow+sideSize:firstRow+self.barWidth+sideSize] = random.uniform(1-random_pixels, 1+random_pixels)

        return patch


    def drawCircle(self):

        resizeFactor = 1.01
        radius = self.shapeSize/(2*resizeFactor)
        patch  = numpy.zeros((self.shapeSize, self.shapeSize))
        center = (int(self.shapeSize/2)-1, int(self.shapeSize/2)-1) # due to discretization, you maybe need add or remove 1 to center coordinates to make it look nice

        for row in range(self.shapeSize):
            for col in range(self.shapeSize):

                distance = numpy.sqrt((row-center[0])**2 + (col-center[1])**2)
                if radius-self.barWidth < distance < radius:
                    patch[row, col] = random.uniform(1-random_pixels, 1+random_pixels)

        return patch


    def drawPolygon(self, nSides, phi):

        resizeFactor = 1.0
        patch  = numpy.zeros((self.shapeSize, self.shapeSize))
        center = (int(self.shapeSize/2), int(self.shapeSize/2))
        radius = self.shapeSize/(2*resizeFactor)

        rowExtVertices = []
        colExtVertices = []
        rowIntVertices = []
        colIntVertices = []
        for n in range(nSides):
            rowExtVertices.append( radius               *numpy.sin(2*numpy.pi*n/nSides + phi) + center[0])
            colExtVertices.append( radius               *numpy.cos(2*numpy.pi*n/nSides + phi) + center[1])
            rowIntVertices.append((radius-self.barWidth)*numpy.sin(2*numpy.pi*n/nSides + phi) + center[0])
            colIntVertices.append((radius-self.barWidth)*numpy.cos(2*numpy.pi*n/nSides + phi) + center[1])

        RR, CC = draw.polygon(rowExtVertices, colExtVertices)
        rr, cc = draw.polygon(rowIntVertices, colIntVertices)
        patch[RR, CC] = random.uniform(1-random_pixels, 1+random_pixels)
        patch[rr, cc] = 0.0

        return patch


    def drawStar(self, nTips, ratio, phi):

        resizeFactor = 0.8
        patch  = numpy.zeros((self.shapeSize, self.shapeSize))
        center = (int(self.shapeSize/2), int(self.shapeSize/2))
        radius = self.shapeSize/(2*resizeFactor)

        rowExtVertices = []
        colExtVertices = []
        rowIntVertices = []
        colIntVertices = []
        for n in range(2*nTips):

            thisRadius = radius
            if not n%2:
                thisRadius = radius/ratio

            rowExtVertices.append(max(min( thisRadius               *numpy.sin(2*numpy.pi*n/(2*nTips) + phi) + center[0], self.shapeSize), 0.0))
            colExtVertices.append(max(min( thisRadius               *numpy.cos(2*numpy.pi*n/(2*nTips) + phi) + center[1], self.shapeSize), 0.0))
            rowIntVertices.append(max(min((thisRadius-self.barWidth)*numpy.sin(2*numpy.pi*n/(2*nTips) + phi) + center[0], self.shapeSize), 0.0))
            colIntVertices.append(max(min((thisRadius-self.barWidth)*numpy.cos(2*numpy.pi*n/(2*nTips) + phi) + center[1], self.shapeSize), 0.0))

        RR, CC = draw.polygon(rowExtVertices, colExtVertices)
        rr, cc = draw.polygon(rowIntVertices, colIntVertices)
        patch[RR, CC] = random.uniform(1-random_pixels, 1+random_pixels)
        patch[rr, cc] = 0.0

        return patch


    def drawIrreg(self, nSidesRough, repeatShape):

        if repeatShape:
            random.seed(1)

        patch  = numpy.zeros((self.shapeSize, self.shapeSize))
        center = (int(self.shapeSize/2), int(self.shapeSize/2))
        angle  = 0  # first vertex is at angle 0

        rowExtVertices = []
        colExtVertices = []
        rowIntVertices = []
        colIntVertices = []
        while angle < 2*numpy.pi:

            if numpy.pi/4 < angle < 3*numpy.pi/4 or 5*numpy.pi/4 < angle < 7*numpy.pi/4:
                radius = (random.random()+2.0)/3.0*self.shapeSize/2
            else:
                radius = (random.random()+1.0)/2.0*self.shapeSize/2

            rowExtVertices.append( radius               *numpy.sin(angle) + center[0])
            colExtVertices.append( radius               *numpy.cos(angle) + center[1])
            rowIntVertices.append((radius-self.barWidth)*numpy.sin(angle) + center[0])
            colIntVertices.append((radius-self.barWidth)*numpy.cos(angle) + center[1])

            angle += (random.random()+0.5)*(2*numpy.pi/nSidesRough)

        RR, CC = draw.polygon(rowExtVertices, colExtVertices)
        rr, cc = draw.polygon(rowIntVertices, colIntVertices)
        patch[RR, CC] = random.uniform(1-random_pixels, 1+random_pixels)
        patch[rr, cc] = 0.0

        if repeatShape:
            random.seed(datetime.now())

        return patch


    def drawStuff(self, nLines):

        patch  = numpy.zeros((self.shapeSize, self.shapeSize))

        for n in range(nLines):

            (r1, c1, r2, c2) = numpy.random.randint(self.shapeSize, size=4)
            rr, cc = draw.line(r1, c1, r2, c2)
            patch[rr, cc] = random.uniform(1-random_pixels, 1+random_pixels)

        return patch


    def drawVernier(self, offset=None, offset_size=None):

        if offset_size is None:
            offset_size = random.randint(1, int(self.barHeight/2.0))
        patch = numpy.zeros((2*self.barHeight+self.offsetHeight, 2*self.barWidth+offset_size))
        patch[0:self.barHeight, 0:self.barWidth] = 1.0
        patch[self.barHeight+self.offsetHeight:, self.barWidth+offset_size:] = random.uniform(1-random_pixels, 1+random_pixels)

        if offset is None:
            if random.randint(0, 1):
                patch = numpy.fliplr(patch)
        elif offset == 1:
            patch = numpy.fliplr(patch)

        fullPatch = numpy.zeros((self.shapeSize, self.shapeSize))
        firstRow  = int((self.shapeSize-patch.shape[0])/2)
        firstCol  = int((self.shapeSize-patch.shape[1])/2)
        fullPatch[firstRow:firstRow+patch.shape[0], firstCol:firstCol+patch.shape[1]] = patch

        return fullPatch


    def drawShape(self, shapeID, offset=None, offset_size=None):

        if shapeID == 0:
            patch = self.drawVernier(offset, offset_size)
        if shapeID == 1:
            patch = self.drawSquare()
        if shapeID == 2:
            patch = self.drawCircle()
        if shapeID == 3:
            patch = self.drawPolygon(6, 0)
        if shapeID == 4:
            patch = self.drawPolygon(8, numpy.pi/8)
        if shapeID == 5:
            patch = self.drawStar(4, 1.8, 0)
        if shapeID == 6:
            patch = self.drawStar(7, 1.7, -numpy.pi/14)
        if shapeID == 7:
            patch = self.drawIrreg(15, False)
        if shapeID == 8:
            patch = self.drawIrreg(15, True)
        if shapeID == 9:
            patch = self.drawStuff(5)

        return patch


    def drawStim(self, vernier, shapeMatrix, offset=None, offset_size=None, fixed_position=None):

        image        = numpy.zeros(self.imSize)
        critDist     = 0 # int(self.shapeSize/6)
        padDist      = int(self.shapeSize/6)
        shapeMatrix  = numpy.array(shapeMatrix)

        if len(shapeMatrix.shape) < 2:
            shapeMatrix = numpy.expand_dims(shapeMatrix, axis=0)

        if shapeMatrix.all() == None:  # this means we want only a vernier
            patch = numpy.zeros((self.shapeSize+5, self.shapeSize+5))
        else:
            patch = numpy.zeros((shapeMatrix.shape[0]*self.shapeSize + (shapeMatrix.shape[0]-1)*critDist + 1,
                                 shapeMatrix.shape[1]*self.shapeSize + (shapeMatrix.shape[1]-1)*critDist + 1))

            for row in range(shapeMatrix.shape[0]):
                for col in range(shapeMatrix.shape[1]):

                    firstRow = row*(self.shapeSize + critDist)
                    firstCol = col*(self.shapeSize + critDist)
                    patch[firstRow:firstRow+self.shapeSize, firstCol:firstCol+self.shapeSize] = self.drawShape(shapeMatrix[row,col], offset, offset_size)

        if vernier:

            firstRow = int((patch.shape[0]-self.shapeSize)/2) # + 1  # small adjustments may be needed depending on precise image size
            firstCol = int((patch.shape[1]-self.shapeSize)/2) # + 1
            patch[firstRow:firstRow+self.shapeSize, firstCol:firstCol+self.shapeSize] += self.drawVernier(offset, offset_size)
            patch[patch > 1.0] = 1.0

        if fixed_position is None:
            firstRow = random.randint(padDist, self.imSize[0] - (patch.shape[0]+padDist))  # int((self.imSize[0]-patch.shape[0])/2)
            firstCol = random.randint(padDist, self.imSize[1] - (patch.shape[1]+padDist))  # int((self.imSize[1]-patch.shape[1])/2)
        else:
            firstRow = fixed_position[0]
            firstCol = fixed_position[1]

        image[firstRow:firstRow+patch.shape[0], firstCol:firstCol+patch.shape[1]] = patch

        # make images with only -1 and 1
        image[image==0] = -0.
        image[image>0] = 1.

        return image


    def plotStim(self, vernier, shapeMatrix):

        plt.figure()
        plt.imshow(self.drawStim(vernier, shapeMatrix))
        plt.show()


    def makeBatchOld(self, batchSize, shapeTypes, noiseLevel=0.0, group_last_shapes=1, normalize=False, fixed_position=None):

        # group_last_types attributes the same label to the last n shapeTypes
        shapeLabels = numpy.arange(1, len(shapeTypes)+1)
        shapeLabels[-group_last_shapes:] = shapeLabels[-group_last_shapes]
        batchImages = numpy.ndarray(shape=(batchSize, self.imSize[0], self.imSize[1]), dtype=numpy.float32)
        batchLabels = numpy.zeros(batchSize, dtype=numpy.float32)

        for n in range(batchSize):

            if random.uniform(0,1) < 0.25:
                batchImages[n,:,:] = self.drawStim(False, shapeMatrix=[0], fixed_position=fixed_position) + numpy.random.normal(0, noiseLevel, size=self.imSize)
                batchLabels[n]     = 0

            else:
                thisType    = random.randint(0, len(shapeTypes)-1)
                shapeType   = shapeTypes[thisType]
                shapeConfig = shapeType*numpy.ones((random.randint(1, 3), random.randint(1, 7)))
                batchImages[n,:,:] = self.drawStim(0, shapeConfig, fixed_position=fixed_position) + numpy.random.normal(0, noiseLevel, size=self.imSize)
                if normalize:
                    batchImages[n, :, :] = (batchImages[n,:,:] - numpy.mean(batchImages[n,:,:])) / numpy.std(batchImages[n,:,:])
                batchLabels[n]     = shapeLabels[thisType]

        batchImages = numpy.expand_dims(batchImages, -1)  # need to a a fourth dimension for tensorflow

        return batchImages, batchLabels


    def makeBatch(self, batchSize, shapeTypes, noiseLevel=0.0, group_last_shapes=1, max_rows=1, max_cols=5, vernierLabelEncoding='lr_01', vernier_grids=False, normalize=False, fixed_position=None):

        # group_last_types attributes the same label to the last n shapeTypes
        shapeLabels = numpy.arange(len(shapeTypes))
        shapeLabels[-group_last_shapes:] = shapeLabels[-group_last_shapes]
        batchImages = numpy.zeros(shape=(batchSize, self.imSize[0], self.imSize[1]), dtype=numpy.float32)
        batchLabels = numpy.zeros(batchSize, dtype=numpy.float32)
        nElements = numpy.zeros(batchSize, dtype=numpy.int64)
        vernierLabels = numpy.zeros(batchSize, dtype=numpy.float32)  # 0 -> not a vernier, 1 -> left, 2 -> right

        if vernier_grids:  # verniers come in grids, like all other shapes.

            for n in range(batchSize):

                thisType = random.randint(0, len(shapeTypes)-1)
                shapeType = shapeTypes[thisType]
                nRows = random.randint(1, max_rows)
                nCols = random.randint(1, max_cols)
                shapeConfig = shapeType*numpy.ones((nRows, nCols))
                thisOffset = random.randint(0, 1)
                batchImages[n, :, :] += self.drawStim(0, shapeConfig, fixed_position=fixed_position, offset=thisOffset, offset_size=random.randint(1, int(self.barHeight/2.0))) + numpy.random.normal(0, noiseLevel, size=self.imSize)
                if normalize:
                    batchImages[n, :, :] = (batchImages[n, :, :] - numpy.mean(batchImages[n, :, :])) / numpy.std(batchImages[n, :, :])
                batchLabels[n, shape] = shapeLabels[thisType]
                nElements[n, shape] = nRows*nCols
                if vernierLabelEncoding is 'lr_01':
                    if thisType == 0:
                        vernierLabels[n] = random.randint(0, 1)
                    else:
                        vernierLabels[n] = -thisOffset
                elif vernierLabelEncoding is 'nothinglr_012':
                    if thisType == 0:
                        vernierLabels[n] = 0
                    else:
                        vernierLabels[n] = -thisOffset + 2

            batchImages = numpy.expand_dims(batchImages, -1)  # need to a a fourth dimension for tensorflow

        else:  # if vernier_grids is false, the vernier stimuli always comprise a single vernier

            for n in range(batchSize):

                if random.uniform(0, 1) < 0.5:  # 1/len(shapeTypes):
                    thisOffset = random.randint(0, 1)
                    batchImages[n, :, :] += self.drawStim(False, shapeMatrix=[0],  fixed_position=fixed_position, offset=thisOffset, offset_size=random.randint(1, int(self.barHeight/2.0))) + numpy.random.normal(0, noiseLevel, size=self.imSize)
                    if normalize:
                        batchImages[n, :, :] = (batchImages[n, :, :] - numpy.mean(batchImages[n, :, :])) / numpy.std(batchImages[n, :, :])
                    batchLabels[n] = 0
                    nElements[n] = 1
                    if vernierLabelEncoding is 'lr_01':
                        vernierLabels[n] = -thisOffset
                    elif vernierLabelEncoding is 'nothinglr_012':
                        vernierLabels[n] = -thisOffset + 2

                else:
                    thisType = random.randint(1, len(shapeTypes)-1)
                    shapeType = shapeTypes[thisType]
                    nRows = random.randint(1, max_rows)
                    nCols = random.randint(1, max_cols)
                    shapeConfig = shapeType*numpy.ones((nRows, nCols))
                    batchImages[n, :, :] += self.drawStim(0, shapeConfig, fixed_position=fixed_position) + numpy.random.normal(0, noiseLevel, size=self.imSize)
                    if normalize:
                        batchImages[n, :, :] = (batchImages[n, :, :] - numpy.mean(batchImages[n, :, :])) / numpy.std(batchImages[n, :, :])
                    batchLabels[n] = shapeLabels[thisType]
                    nElements[n] = nRows * nCols
                    if vernierLabelEncoding is 'lr_01':
                        vernierLabels[n] = random.randint(0, 1)
                    elif vernierLabelEncoding is 'nothinglr_012':
                        vernierLabels[n] = 0

        if normalize_sets:
            batchImages = (batchImages - self.mean) / self.std

        batchImages = numpy.expand_dims(batchImages, -1)  # need to a a fourth dimension for tensorflow

        return batchImages, batchLabels, vernierLabels, nElements


    def makeMultiShapeBatch(self, batchSize, shapeTypes, n_shapes=1, noiseLevel=0.0, group_last_shapes=1, max_rows=1, max_cols=3, vernierLabelEncoding='nothinglr_012', vernier_grids=False, normalize=False, normalize_sets=False, fixed_position=None, random_size=False):

        # group_last_types attributes the same label to the last n shapeTypes
        shapeLabels = numpy.arange(len(shapeTypes))
        shapeLabels[-group_last_shapes:] = shapeLabels[-group_last_shapes]
        batchImages = numpy.zeros(shape=(batchSize, self.imSize[0], self.imSize[1]), dtype=numpy.float32)
        batchSingleShapeImages =  numpy.zeros(shape=(batchSize, self.imSize[0], self.imSize[1], n_shapes), dtype=numpy.float32)
        batchLabels = numpy.zeros(shape=(batchSize, n_shapes), dtype=numpy.float32)
        nElements = numpy.zeros(shape=(batchSize, n_shapes), dtype=numpy.int64)
        if vernierLabelEncoding is 'nothinglr_012':
            vernierLabels = numpy.zeros(shape=(batchSize, n_shapes), dtype=numpy.float32)  # 0 -> not a vernier, 1 -> left, 2 -> right
        elif vernierLabelEncoding is 'lr_01':
            vernierLabels = numpy.zeros(shape=[batchSize], dtype=numpy.float32)  # 0 -> left, 1 -> right
        if vernier_grids:  # verniers come in grids, like all other shapes.

            for n in range(batchSize):

                shapes = numpy.random.permutation(len(shapeTypes))[:n_shapes]

                for shape in range(n_shapes):

                    thisType = shapes[shape]
                    shapeType = shapeTypes[thisType]
                    nRows = random.randint(1, max_rows)
                    nCols = random.randint(1, max_cols)
                    shapeConfig = shapeType*numpy.ones((nRows, nCols))
                    thisOffset = random.randint(0, 1)
                    batchSingleShapeImages[n, :, :, shape] = self.drawStim(0, shapeConfig, fixed_position=fixed_position, offset=thisOffset, offset_size=random.randint(1, int(self.barHeight/2.0))) + numpy.random.normal(0, noiseLevel, size=self.imSize)
                    if random_size:
                        zoom_factor = random.uniform(0.8, 1.2)
                        for shape in range(n_shapes):
                            tempImage = clipped_zoom(batchSingleShapeImages[n, :, :, shape], zoom_factor)
                            if tempImage.shape == batchSingleShapeImages[n, :, :,
                                                  shape].shape:  # because sometimes the zooming fucks up the image
                                batchSingleShapeImages[n, :, :, shape] = tempImage
                    batchImages[n, :, :] += batchSingleShapeImages[n, :, :, shape]
                    if normalize:
                        batchSingleShapeImages[n, :, :, shape] = (batchSingleShapeImages[n, :, :, shape] - numpy.mean(batchSingleShapeImages[n, :, :, shape])) / numpy.std(batchSingleShapeImages[n, :, :, shape])
                    batchLabels[n, shape] = shapeLabels[thisType]
                    nElements[n, shape] = nRows*nCols
                    if vernierLabelEncoding is 'lr_01':
                        if thisType == 0:
                            vernierLabels[n, shape] = random.randint(0, 1)
                        else:
                            vernierLabels[n, shape] = -thisOffset + 1
                    elif vernierLabelEncoding is 'nothinglr_012':
                        if thisType == 0:
                            vernierLabels[n, shape] = 0
                        else:
                            vernierLabels[n, shape] = -thisOffset + 2
                if normalize:
                    batchImages[n, :, :] = (batchImages[n, :, :] - numpy.mean(batchImages[n, :, :])) / numpy.std(batchImages[n, :, :])

        else:  # if vernier_grids is false, the vernier stimuli always comprise a single vernier

            for n in range(batchSize):

                always_vernier = True
                if always_vernier:
                    # always a vernier with another shape
                    shapes = numpy.zeros(shape=[n_shapes], dtype=numpy.int64)
                    shapes[0] = 0
                    shapes[1:] = numpy.random.permutation(range(1, len(shapeTypes)))[:n_shapes-1]
                else:
                    # use a vernier every other trial
                    if n % 2:
                        shapes = numpy.random.permutation(len(shapeTypes))[:n_shapes]
                        shapes[0] = 0
                    else:
                        shapes = numpy.random.permutation(len(shapeTypes))[:n_shapes]

                for shape in range(n_shapes):
                    if shapes[shape] == 0:  # 1/len(shapeTypes):
                        thisOffset = random.randint(0, 1)
                        batchSingleShapeImages[n, :, :, shape] = self.drawStim(False, shapeMatrix=[0],  fixed_position=fixed_position, offset=thisOffset, offset_size=random.randint(1, int(self.barHeight/2.0))) + numpy.random.normal(0, noiseLevel, size=self.imSize)
                        batchSingleShapeImages[batchSingleShapeImages > 0.2] = 1
                        batchSingleShapeImages[batchSingleShapeImages < 0] = 0
                        if normalize:
                            batchSingleShapeImages[n, :, :, shape] = (batchSingleShapeImages[n, :, :, shape] - numpy.mean(batchSingleShapeImages[n, :, :, shape])) / (numpy.std(batchSingleShapeImages[n, :, :, shape]))**vernier_normalization_exp
                            # batchSingleShapeImages[batchSingleShapeImages < 0] = 0

                        if random_size:
                            zoom_factor = random.uniform(0.8, 1.2)
                            tempImage = clipped_zoom(batchSingleShapeImages[n, :, :, shape], zoom_factor)
                            if tempImage.shape == batchSingleShapeImages[n, :, :, shape].shape:  # because sometimes the zooming fucks up the image
                                batchSingleShapeImages[n, :, :, shape] = tempImage

                        batchImages[n, :, :] += batchSingleShapeImages[n, :, :, shape]

                        # note, we could normalize batchSingleShapeImages AFTER adding it to the multishape image, to avoid normalizing the multishape image several times

                        batchLabels[n, shape] = 0
                        nElements[n, shape] = 1
                        if vernierLabelEncoding is 'lr_01':
                            vernierLabels[n] = -thisOffset + 1
                        elif vernierLabelEncoding is 'nothinglr_012':
                            vernierLabels[n, shape] = -thisOffset + 2

                    else:
                        thisType = shapes[shape]
                        shapeType = shapeTypes[thisType]
                        nRows = random.randint(1, max_rows)
                        nCols = random.randint(1, max_cols)
                        shapeConfig = shapeType*numpy.ones((nRows, nCols))
                        batchSingleShapeImages[n, :, :, shape] = self.drawStim(0, shapeConfig, fixed_position=fixed_position) + numpy.random.normal(0, noiseLevel, size=self.imSize)
                        batchSingleShapeImages[batchSingleShapeImages > 0.2] = 1
                        batchSingleShapeImages[batchSingleShapeImages < 0] = 0
                        if normalize:
                            batchSingleShapeImages[n, :, :, shape] = (batchSingleShapeImages[n, :, :, shape] - numpy.mean(batchSingleShapeImages[n, :, :, shape])) / numpy.std(batchSingleShapeImages[n, :, :, shape])**vernier_normalization_exp
                            # batchSingleShapeImages[batchSingleShapeImages < 0] = 0

                        if random_size:
                            zoom_factor = random.uniform(0.8, 1.2)
                            tempImage = clipped_zoom(batchSingleShapeImages[n, :, :, shape], zoom_factor)
                            if tempImage.shape == batchSingleShapeImages[n, :, :, shape].shape:  # because sometimes the zooming fucks up the image
                                batchSingleShapeImages[n, :, :, shape] = tempImage

                        batchImages[n, :, :] += batchSingleShapeImages[n, :, :, shape]

                        # note, we could normalize batchSingleShapeImages AFTER adding it to the multishape image, to avoid normalizing the multishape image several times

                        batchLabels[n, shape] = shapeLabels[thisType]
                        nElements[n, shape] = nRows * nCols
                        if vernierLabelEncoding is 'nothinglr_012':
                            vernierLabels[n, shape] = 0


        if normalize_sets:
            # batchImages[batchImages > 1.2] = 1.2  # to avoid overlapping pixels to by twice as high as the rest
            batchImages = (batchImages - self.mean) / self.std
            batchSingleShapeImages = (batchSingleShapeImages - self.mean) / self.std

        if vernierLabelEncoding is 'nothinglr_012':
            vernierLabels = numpy.amax(vernierLabels, axis=1)

        batchImages = numpy.expand_dims(batchImages, -1)  # need to a a fourth dimension for tensorflow

        return batchImages, batchSingleShapeImages, batchLabels, vernierLabels, nElements


    def makeVernierBatch(self, batchSize, noiseLevel=0.0, normalize=False, normalize_sets=False, fixed_position=None, vernierLabelEncoding='nothinglr_012'):

        batchImages = numpy.ndarray(shape=(batchSize, self.imSize[0], self.imSize[1]), dtype=numpy.float32)
        batchLabels = numpy.zeros(batchSize, dtype=numpy.float32)

        for n in range(batchSize):

            offset = random.randint(0, 1)
            batchImages[n,:,:] = self.drawStim(False, shapeMatrix=[0], offset=offset, fixed_position=fixed_position) + numpy.random.normal(0, noiseLevel, size=self.imSize)
            if normalize:
                batchImages[n, :, :] = (batchImages[n, :, :] - numpy.mean(batchImages[n, :, :])) / numpy.std(batchImages[n, :, :])
            if vernierLabelEncoding is 'nothinglr_012':
                batchLabels[n] = -offset + 2
            elif vernierLabelEncoding is 'lr_01':
                batchLabels[n] = -offset + 1

        if normalize_sets:
            batchImages = (batchImages - self.mean) / self.std

        batchImages = numpy.expand_dims(batchImages, -1)  # need to a a fourth dimension for tensorflow

        return batchImages, batchLabels


    def makeConfigBatch(self, batchSize, configMatrix, noiseLevel=0.0, normalize=False, normalize_sets=False, fixed_position=None, vernierLabelEncoding='nothinglr_012', random_size=False):

        batchImages   = numpy.ndarray(shape=(batchSize, self.imSize[0], self.imSize[1]), dtype=numpy.float32)
        vernierLabels = numpy.zeros(batchSize, dtype=numpy.float32)

        for n in range(batchSize):

            offset = random.randint(0, 1)
            batchImages[n, :, :] = self.drawStim(True, shapeMatrix=configMatrix, fixed_position=fixed_position, offset=offset) + numpy.random.normal(0, noiseLevel, size=self.imSize)
            if normalize:
                # batchImages[batchImages < 0] = 0
                # batchImages[batchImages > 0.2] = 1
                batchImages[n, :, :] = (batchImages[n, :, :] - numpy.mean(batchImages[n, :, :])) / numpy.std(batchImages[n, :, :])**vernier_normalization_exp
            if vernierLabelEncoding is 'nothinglr_012':
                vernierLabels[n] = -offset + 2
            elif vernierLabelEncoding is 'lr_01':
                vernierLabels[n] = -offset + 1

            if random_size:
                zoom_factor = random.uniform(0.8, 1.2)
                tempImage = clipped_zoom(batchImages[n, :, :], zoom_factor)
                tempImage[tempImage == 0] = -numpy.mean(tempImage)  # because when using random_sizes, small images get padded with 0 but the background may be <= because of normalization
                if tempImage.shape == batchImages[n, :, :].shape:
                    batchImages[n, :, :] = tempImage

                # batchImages[batchImages > 0.2] = 1
                # batchImages[batchImages < 0] = 0

        if normalize_sets:
            batchImages = (batchImages - self.mean) / self.std

        batchImages = numpy.expand_dims(batchImages, -1)  # need to a a fourth dimension for tensorflow

        return batchImages, vernierLabels


    # this returns the batch along with patches for each image (e.g. a square patch if the image is made of squares)
    def makeBatchWithShape(self, batchSize, shapeTypes, noiseLevel=0.0, group_last_shapes=1, normalize=False, normalize_sets=False, fixed_position=None):

        # group_last_types attributes the same label to the last n shapeTypes
        shapeLabels = numpy.arange(1, len(shapeTypes) + 1)
        shapeLabels[-group_last_shapes:] = shapeLabels[-group_last_shapes]
        batchImages = numpy.ndarray(shape=(batchSize, self.imSize[0], self.imSize[1]), dtype=numpy.float32)
        batchLabels = numpy.zeros(batchSize, dtype=numpy.float32)
        batchShapes = numpy.ndarray(shape=(batchSize, self.shapeSize, self.shapeSize), dtype=numpy.float32)

        for n in range(batchSize):

            if not n % 4:
                offset = random.randint(0, 1)
                batchImages[n, :, :] = self.drawStim(False, shapeMatrix=[0], offset=offset, fixed_position=fixed_position) + numpy.random.normal(0, noiseLevel, size=self.imSize)
                batchShapes[n, :, :] = self.drawVernier(offset=offset)
                if normalize:
                    batchImages[n, :, :] = (batchImages[n,:,:] - numpy.mean(batchImages[n,:,:])) / numpy.std(batchImages[n,:,:])
                batchLabels[n] = 0

            else:
                thisType = random.randint(0, len(shapeTypes) - 1)
                shapeType = shapeTypes[thisType]
                shapeConfig = shapeType * numpy.ones((random.randint(1, 3), random.randint(1, 7)))
                batchImages[n, :, :] = self.drawStim(0, shapeConfig, fixed_position=fixed_position) + numpy.random.normal(0, noiseLevel, size=self.imSize)
                batchShapes[n, :, :] = self.drawShape(shapeType)
                if normalize:
                    batchImages[n, :, :] = (batchImages[n,:,:] - numpy.mean(batchImages[n,:,:])) / numpy.std(batchImages[n,:,:])
                batchLabels[n] = shapeLabels[thisType]

        if normalize_sets:
            batchImages = (batchImages - self.mean) / self.std

        batchImages = numpy.expand_dims(batchImages, -1)  # need to a a fourth dimension for tensorflow
        batchShapes = numpy.expand_dims(batchShapes, -1)  # need to a a fourth dimension for tensorflow

        return batchImages, batchLabels, batchShapes


    def showBatch(self, batchSize, shapeTypes, n_shapes=1, showVernier=False, showPatch=False, showConfig='no_config', noiseLevel=0.0, group_last_shapes=1, normalize=False, fixed_position=None, random_size=False, vernierLabelEncoding='lr_01'):

        if showPatch:
            batchImages, batchLabels, batchShapes = self.makeBatchWithShape(batchSize, shapeTypes, noiseLevel=0.0, group_last_shapes=1, normalize=normalize, fixed_position=fixed_position)

            for n in range(batchSize):
                plt.figure()
                plt.imshow(batchImages[n, :, :, 0])
                plt.title('Label, mean, stdev = ' + str(batchLabels[n]) + ', ' + str(numpy.mean(batchImages[n, :, :, 0])) + ', ' +  str(numpy.std(batchImages[n, :, :, 0])))
                plt.show()
                plt.imshow(batchShapes[n, :, :, 0])
                plt.title('Single shape from previous stimulus')
                plt.show()

        elif showVernier:
            batchImages, batchLabels = self.makeVernierBatch(batchSize, noiseLevel, normalize, fixed_position)

            for n in range(batchSize):
                plt.figure()
                plt.imshow(batchImages[n, :, :, 0])
                plt.title('Label, mean, stdev = ' + str(batchLabels[n]) + ', ' + str(numpy.mean(batchImages[n, :, :, 0])) + ', ' + str(numpy.std(batchImages[n, :, :, 0])))
                plt.show()

        elif showConfig is not 'no_config':
            # input a configuration to display
            batchImages, batchLabels = self.makeConfigBatch(batchSize, showConfig, noiseLevel=noiseLevel, normalize=normalize, fixed_position=fixed_position, vernierLabelEncoding=vernierLabelEncoding)

            for n in range(batchSize):
                plt.figure()
                plt.imshow(batchImages[n, :, :, 0])
                plt.title('Label, mean, stdev = ' + str(batchLabels[n]) + ', ' + str(
                    numpy.mean(batchImages[n, :, :, 0])) + ', ' + str(numpy.std(batchImages[n, :, :, 0])))
                plt.show()

        elif n_shapes>1:
            batchImages, batchSingleShapeImages, batchLabels, vernierLabels, nElements = self.makeMultiShapeBatch(batchSize, shapeTypes, n_shapes=n_shapes, noiseLevel=noiseLevel, group_last_shapes=group_last_shapes, normalize=normalize, fixed_position=fixed_position, random_size=random_size, vernierLabelEncoding=vernierLabelEncoding)
            for n in range(batchSize):
                plt.figure()
                plt.imshow(batchImages[n, :, :, 0])
                plt.title('Label, vernier label, mean, stdev = ' + str(batchLabels[n]) + ', ' + str(vernierLabels[n]) + ', ' + str(numpy.mean(batchImages[n, :, :, 0])) + ', ' + str(numpy.std(batchImages[n, :, :, 0])))
                plt.show()
                for m in range(n_shapes):
                    plt.figure()
                    plt.imshow(batchSingleShapeImages[n, :, :, m])
                    plt.title('Single shape ' + str(m) + '. Label = ' + str(batchLabels[n,m]))
                    plt.show()
        else:
            batchImages, batchLabels, vernierLabels, nElements = self.makeBatch(batchSize, shapeTypes, noiseLevel=noiseLevel, group_last_shapes=group_last_shapes, normalize=normalize, fixed_position=fixed_position)
            for n in range(batchSize):
                plt.figure()
                plt.imshow(batchImages[n, :, :, 0])
                plt.title('Label, vernier label, mean, stdev = ' + str(batchLabels[n]) + ', ' + str(vernierLabels[n]) + ', ' + str(numpy.mean(batchImages[n, :, :, 0])) + ', ' + str(numpy.std(batchImages[n, :, :, 0])))
                plt.show()


if __name__ == "__main__":

    rufus = StimMaker((45, 100), 18, 2)
    # rufus.plotStim(1, [[1, 2, 3], [4, 5, 6], [6, 7, 0]])
    rufus.showBatch(20, [0, 1, 2, 6, 7], n_shapes=2, showPatch=False, showVernier=False, showConfig='no_config', noiseLevel=0., group_last_shapes=1, normalize=False, random_size=True, vernierLabelEncoding='lr_01')
