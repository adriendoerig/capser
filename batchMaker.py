
# Class to make a batch

import numpy, random, turtle, matplotlib.pyplot as plt
from skimage import draw

class StimMaker:

	def __init__(self, imSize, shapeSize, barWidth):
	    
		self.imSize    = imSize
		self.shapeSize = shapeSize
		self.barWidth  = barWidth
		self.barHeight = int(shapeSize/3-barWidth/2)
		self.offsetHeight = 0

	def drawSquare(self):

		patch = numpy.zeros((self.shapeSize, self.shapeSize))
		patch[0:self.barWidth, :] = 254.0
		patch[-self.barWidth:, :] = 254.0
		patch[:, 0:self.barWidth] = 254.0
		patch[:, -self.barWidth:] = 254.0

		return patch

	def drawCircle(self):

		patch  = numpy.zeros((self.shapeSize, self.shapeSize))
		center = (int(self.shapeSize/2), int(self.shapeSize/2))
		for row in xrange(self.shapeSize):
			for col in xrange(self.shapeSize):

				distance = numpy.sqrt((row-center[0])**2 + (col-center[1])**2)
				if self.shapeSize/2-self.barWidth < distance < self.shapeSize/2:
					patch[row, col] = 254.0

		return patch

	def drawPolygon(self, nSides, phi):

		patch  = numpy.zeros((self.shapeSize, self.shapeSize))
		center = (int(self.shapeSize/2), int(self.shapeSize/2))
		radius = self.shapeSize/2

		rowExtVertices = []
		colExtVertices = []
		rowIntVertices = []
		colIntVertices = []
		for n in xrange(nSides):
			rowExtVertices.append( radius               *numpy.sin(2*numpy.pi*n/nSides + phi) + center[0])
			colExtVertices.append( radius               *numpy.cos(2*numpy.pi*n/nSides + phi) + center[1])
			rowIntVertices.append((radius-self.barWidth)*numpy.sin(2*numpy.pi*n/nSides + phi) + center[0])
			colIntVertices.append((radius-self.barWidth)*numpy.cos(2*numpy.pi*n/nSides + phi) + center[1])

		RR, CC = draw.polygon(rowExtVertices, colExtVertices)
		rr, cc = draw.polygon(rowIntVertices, colIntVertices)
		patch[RR, CC] = 254.0
		patch[rr, cc] = 0.0

		return patch

	# def draw_star(self, nTips, ratio):
	    
	# 	patch = numpy.zeros((self.shapeSize, self.shapeSize))

	#     angle = 120
	#     turtle.fillcolor(color)
	#     turtle.begin_fill()

	#     for side in range(nTips):
	        
	#         turtle.forward(size)
	#         turtle.right(angle)
	#         turtle.forward(size)
	#         turtle.right(360.0/nTips-angle)
	    
	#     turtle.end_fill()
    	
 #    	return patch

	def drawVernier(self):
		
		offsetWidth  = random.randint(1, int(self.barHeight/1.5))
		patch        = numpy.zeros((2*self.barHeight+self.offsetHeight, 2*self.barWidth+offsetWidth))
		patch[0                     :self.barHeight, 0           :self.barWidth] = 254.0
		patch[self.barHeight+self.offsetHeight:    , self.barWidth+offsetWidth:] = 254.0

		if random.randint(0,1):
			patch = numpy.fliplr(patch)

		fullPatch = numpy.zeros((self.shapeSize, self.shapeSize))
		firstRow  = int((self.shapeSize-patch.shape[0])/2)
		firstCol  = int((self.shapeSize-patch.shape[1])/2)
		fullPatch[firstRow:firstRow+patch.shape[0], firstCol:firstCol+patch.shape[1]] = patch

		return fullPatch

	def drawShape(self, shapeID):

		if shapeID == 0:
			patch = self.drawSquare()

		if shapeID == 1:
			patch = self.drawCircle()

		if shapeID == 2:
			patch = self.drawPolygon(6, 0)

		if shapeID == 3:
			patch = self.drawPolygon(8, numpy.pi/8)

		if shapeID == 6:
			patch = self.drawVernier()

		return patch

	def drawStim(self, vernier, shapeMatrix):

		image        = numpy.zeros(self.imSize)
		critDist     = int(self.shapeSize/6)
		shapeMatrix  = numpy.array(shapeMatrix)
		if len(shapeMatrix.shape) < 2:
			shapeMatrix = numpy.expand_dims(shapeMatrix, axis=0)
		patch        = numpy.zeros((shapeMatrix.shape[0]*self.shapeSize + (shapeMatrix.shape[0]-1)*critDist,
									shapeMatrix.shape[1]*self.shapeSize + (shapeMatrix.shape[1]-1)*critDist))

		for row in xrange(shapeMatrix.shape[0]):
			for col in xrange(shapeMatrix.shape[1]):

				firstRow = row*(self.shapeSize + critDist)
				firstCol = col*(self.shapeSize + critDist)
				patch[firstRow:firstRow+self.shapeSize, firstCol:firstCol+self.shapeSize] = self.drawShape(shapeMatrix[row,col])

		if vernier:

			firstRow = int((patch.shape[0]-self.shapeSize)/2)
			firstCol = int((patch.shape[1]-self.shapeSize)/2)
			patch[firstRow:firstRow+self.shapeSize, firstCol:firstCol+self.shapeSize] += self.drawVernier()

		firstRow = random.randint(critDist, self.imSize[0] - (patch.shape[0]+critDist))  # int((self.imSize[0]-patch.shape[0])/2)
		firstCol = random.randint(critDist, self.imSize[1] - (patch.shape[1]+critDist))  # int((self.imSize[1]-patch.shape[1])/2)
		image[firstRow:firstRow+patch.shape[0], firstCol:firstCol+patch.shape[1]] = patch
		
		return image

	def plotStim(self, vernier, shapeMatrix):

		plt.figure()
		plt.imshow(self.drawStim(vernier, shapeMatrix))
		plt.show()


class BatchMaker(StimMaker):

	def __init__(self, imSize, batchSize, shapeTypes):
	    
	    StimMaker.__init__(imSize)
	    self.shapeTypes = shapeTypes
	    self.batchSize  = batchSize


if __name__ == "__main__":

	rufus = StimMaker((200, 300), 41, 2)
	rufus.plotStim(1, [[0, 1, 2], [3, 0, 1], [2, 3, 6]])
