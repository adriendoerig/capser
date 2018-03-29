
# Class to make a batch

import numpy, random, turtle, matplotlib.pyplot as plt
from skimage import draw
from datetime import datetime


class StimMaker:

    def __init__(self, imSize, shapeSize, barWidth):

        self.imSize    = imSize
        self.shapeSize = shapeSize
        self.barWidth  = barWidth
        self.barHeight = int(shapeSize/3.3-barWidth/2)
        self.offsetHeight = 0


    def setShapeSize(self, shapeSize):

        self.shapeSize = shapeSize


    def drawSquare(self):

        resizeFactor = 1.2
        patch = numpy.zeros((self.shapeSize, self.shapeSize))

        firstRow = int((self.shapeSize - self.shapeSize/resizeFactor)/2)
        firstCol = firstRow
        sideSize = int(self.shapeSize/resizeFactor)

        patch[firstRow         :firstRow+self.barWidth,          firstCol:firstCol+sideSize+self.barWidth] = 254.0
        patch[firstRow+sideSize:firstRow+self.barWidth+sideSize, firstCol:firstCol+sideSize+self.barWidth] = 254.0
        patch[firstRow:firstRow+sideSize+self.barWidth, firstCol         :firstCol+self.barWidth         ] = 254.0
        patch[firstRow:firstRow+sideSize+self.barWidth, firstRow+sideSize:firstRow+self.barWidth+sideSize] = 254.0

        return patch


    def drawCircle(self):

        resizeFactor = 1.1
        radius = self.shapeSize/(2*resizeFactor)
        patch  = numpy.zeros((self.shapeSize, self.shapeSize))
        center = (int(self.shapeSize/2), int(self.shapeSize/2))

        for row in range(self.shapeSize):
            for col in range(self.shapeSize):

                distance = numpy.sqrt((row-center[0])**2 + (col-center[1])**2)
                if radius-self.barWidth < distance < radius:
                    patch[row, col] = 254.0

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
        patch[RR, CC] = 254.0
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
        patch[RR, CC] = 254.0
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
        patch[RR, CC] = 254.0
        patch[rr, cc] = 0.0

        if repeatShape:
            random.seed(datetime.now())

        return patch


    def drawStuff(self, nLines):

        patch  = numpy.zeros((self.shapeSize, self.shapeSize))

        for n in range(nLines):

            (r1, c1, r2, c2) = numpy.random.randint(self.shapeSize, size=4)
            rr, cc = draw.line(r1, c1, r2, c2)
            patch[rr, cc] = 254.0

        return patch


    def drawVernier(self, offset=None, offset_size=None):

        if offset_size is None:
            offset_size = random.randint(1, int(self.barHeight/2.0))
        patch       = numpy.zeros((2*self.barHeight+self.offsetHeight, 2*self.barWidth+offset_size))
        patch[0                     :self.barHeight, 0           :self.barWidth] = 254.0
        patch[self.barHeight+self.offsetHeight:    , self.barWidth+offset_size:] = 254.0

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
            patch = self.drawStuff(20)

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

            firstRow = int((patch.shape[0]-self.shapeSize)/2) + 1
            firstCol = int((patch.shape[1]-self.shapeSize)/2) + 1
            patch[firstRow:firstRow+self.shapeSize, firstCol:firstCol+self.shapeSize] += self.drawVernier(offset)
            patch[patch > 254.0] = 254.0

        if fixed_position is None:
            firstRow = random.randint(padDist, self.imSize[0] - (patch.shape[0]+padDist))  # int((self.imSize[0]-patch.shape[0])/2)
            firstCol = random.randint(padDist, self.imSize[1] - (patch.shape[1]+padDist))  # int((self.imSize[1]-patch.shape[1])/2)
        else:
            firstRow = fixed_position[0]
            firstCol = fixed_position[1]

        image[firstRow:firstRow+patch.shape[0], firstCol:firstCol+patch.shape[1]] = patch

        # make images with only -1 and 1
        image[image==0] = -254.
        image[image>0] = 254.

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


    def makeBatch(self, batchSize, shapeTypes, noiseLevel=0.0, group_last_shapes=1, normalize=False, fixed_position=None, return_n_elements=False):

        # group_last_types attributes the same label to the last n shapeTypes
        shapeLabels = numpy.arange(len(shapeTypes))
        shapeLabels[-group_last_shapes:] = shapeLabels[-group_last_shapes]
        batchImages = numpy.ndarray(shape=(batchSize, self.imSize[0], self.imSize[1]), dtype=numpy.float32)
        batchLabels = numpy.zeros(batchSize, dtype=numpy.float32)

        for n in range(batchSize):

            thisType = random.randint(0, len(shapeTypes)-1)
            shapeType = shapeTypes[thisType]
            nRows = random.randint(1, 1)
            nCols = random.randint(1, 7)
            shapeConfig = shapeType*numpy.ones((nRows, nCols))
            batchImages[n, :, :] = self.drawStim(0, shapeConfig, fixed_position=fixed_position, offset=random.randint(0, 1), offset_size=random.randint(1, int(self.barHeight/2.0))) + numpy.random.normal(0, noiseLevel, size=self.imSize)
            if normalize:
                batchImages[n, :, :] = (batchImages[n, :, :] - numpy.mean(batchImages[n, :, :])) / numpy.std(batchImages[n, :, :])
            batchLabels[n] = shapeLabels[thisType]

        batchImages = numpy.expand_dims(batchImages, -1)  # need to a a fourth dimension for tensorflow

        if return_n_elements is True:
            return batchImages, batchLabels, nRows*nCols  # returns the number of elements in the last stimulus of the batch
        else:
            return batchImages, batchLabels


    def makeVernierBatch(self, batchSize, noiseLevel=0.0, normalize=False, fixed_position=None):
        batchImages = numpy.ndarray(shape=(batchSize, self.imSize[0], self.imSize[1]), dtype=numpy.float32)
        batchLabels = numpy.zeros(batchSize, dtype=numpy.float32)

        for n in range(batchSize):

            offset = random.randint(0, 1)
            batchImages[n,:,:] = self.drawStim(False, shapeMatrix=[0], offset=offset, fixed_position=fixed_position) + numpy.random.normal(0, noiseLevel, size=self.imSize)
            if normalize:
                batchImages[n, :, :] = (batchImages[n, :, :] - numpy.mean(batchImages[n, :, :])) / numpy.std(batchImages[n, :, :])
            batchLabels[n] = offset

        batchImages = numpy.expand_dims(batchImages, -1)  # need to a a fourth dimension for tensorflow

        return batchImages, batchLabels


    def makeConfigBatch(self, batchSize, configMatrix, noiseLevel=0.0, normalize=False, fixed_position=None):

        batchImages   = numpy.ndarray(shape=(batchSize, self.imSize[0], self.imSize[1]), dtype=numpy.float32)
        vernierLabels = numpy.zeros(batchSize, dtype=numpy.float32)

        for n in range(batchSize):

            offset = random.randint(0, 1)
            batchImages[n, :, :] = self.drawStim(True, shapeMatrix=configMatrix, fixed_position=fixed_position, offset=offset, offset_size=random.randint(1, int(self.barHeight/2.0))) + numpy.random.normal(0, noiseLevel, size=self.imSize)
            if normalize:
                batchImages[n, :, :] = (batchImages[n, :, :] - numpy.mean(batchImages[n, :, :])) / numpy.std(batchImages[n, :, :])
            vernierLabels[n] = offset

        batchImages = numpy.expand_dims(batchImages, -1)  # need to a a fourth dimension for tensorflow

        return batchImages, vernierLabels


    # this returns the batch along with patches for each image (e.g. a square patch if the image is made of squares)
    def makeBatchWithShape(self, batchSize, shapeTypes, noiseLevel=0.0, group_last_shapes=1, normalize=False, fixed_position=None):

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

        batchImages = numpy.expand_dims(batchImages, -1)  # need to a a fourth dimension for tensorflow
        batchShapes = numpy.expand_dims(batchShapes, -1)  # need to a a fourth dimension for tensorflow

        return batchImages, batchLabels, batchShapes


    def showBatch(self, batchSize, shapeTypes, showPatch=False, noiseLevel=0.0, group_last_shapes=1, normalize=False, fixed_position=None):

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
        else:
            batchImages, batchLabels = self.makeBatch(batchSize, shapeTypes, noiseLevel, group_last_shapes, normalize=normalize, fixed_position=fixed_position)

            for n in range(batchSize):
                plt.figure()
                plt.imshow(batchImages[n, :, :, 0])
                plt.title('Label, mean, stdev = ' + str(batchLabels[n]) + ', ' + str(numpy.mean(batchImages[n, :, :, 0])) + ', ' + str(numpy.std(batchImages[n, :, :, 0])))
                plt.show()


if __name__ == "__main__":

    rufus = StimMaker((100, 200), 19, 2)
    # rufus.plotStim(1, [[1, 2, 3], [4, 5, 6], [6, 7, 0]])
    rufus.showBatch(20, [0, 1, 2, 6, 7, 9], showPatch=False, noiseLevel=0.0, group_last_shapes=2, normalize=False)
