import torchfile
import config 
import numpy 
import numpy as np
import math
import scipy.misc

def loadSensorData():
	data = torchfile.load('data.t7')
	print data.shape[0]
	#replace vv with data.shape[0]
	vv = 1000
	width = (config.gridMaxX - config.gridMinX)/config.gridStep + 1
	height = (config.gridMaxY - config.gridMinY)/config.gridStep + 1

	distance = numpy.zeros((height,width),dtype='float')
	index = numpy.zeros((height,width),dtype = 'int')

	for y in range(height):
		for x in range(width):
			tmpX = (x)*config.gridStep + config.gridMinX
			tmpY = (y)*config.gridStep + config.gridMinY

			angle = numpy.rad2deg(numpy.arctan2(tmpX,tmpY))
			distance[y][x] = numpy.sqrt(tmpX*tmpX + tmpY*tmpY)
			index[y][x] = numpy.floor((angle - config.gridDepth)/ config.gridDepthStep + 0.5) 

	index = index.reshape(width*height)
	table1 = np.zeros((vv,height,width),dtype = 'float')
	input1 = np.zeros((vv,2,height,width),dtype = 'float')
	for j in range(vv):
		k = 0
		for i in index:
			ii = int(math.floor(k/width))
			jj = int(k-width*math.floor(k/width))
			
			table1[j][ii][jj] = (data[j][i])
			k +=1
		a = abs(table1[j] - distance)
		input1[j][0] = (a<0.7071*config.gridStep) 
		input1[j][1] = table1[j]+0.7071*config.gridStep>distance
			
	return input1


table1 = loadSensorData()
print table1.shape
'''
for i in range(0, 99):
	data_np = np.asarray(table1[i], np.float32)
	scipy.misc.imsave('video2/inp' + str(i) + '.png', (data_np[1] / 2) + data_np[0])
'''