import os
#number of training iterations
iter = 100000

#length of training sequence
N = 100

#load pre trained model from path
path = './model'

#data file path
data = 'data.t7'

learning_rate = 0.01

#initialize net with weights in range
initial_weights = 0




#####################
#For loading sensor data
gridMinX = -25

gridMaxX = 25

gridMinY = -45

gridMaxY = 5

gridStep = 1

gridDepth = -180

gridDepthStep = 0.5 
