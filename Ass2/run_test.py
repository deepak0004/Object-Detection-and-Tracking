import numpy as np
import tensorflow as tf
from RWACell import *	# Modified code to load RWACell
import torchfile
import matplotlib.pyplot as plt
import scipy.misc
from data_gen import *
import random

N = 100
filee = loadSensorData()
print filee.shape
num_cells = 1
# Training parameters
num_iterations = 10
batch_size = 100
learning_rate = 0.001

print 'filee.shape: ', filee.shape

def dropoutInput(target):
	input = []
	for i in range (0,len(target)):
		if( (i%20) >= 10 ):
			input.append(np.zeros(target[i].shape))
		else:
			input.append(target[i])
    
	return input

def getSequence(i):
    input = []
    for j in range(0, N):
        input.append( filee[(i-1) * N + j] )

    return input

def evalModel():
    inputt = getSequence(1)
    
    xs_sequence = np.reshape(inputt, [1, batch_size, 2*51*51])	# Convert image into a sequence of pixels.
    ys = np.reshape(inputt, [batch_size, 2, 51, 51])
    feed_test = {x: xs_sequence, y: ys}

    graph = tf.Graph()
    with tf.Session(graph=graph) as session:
	      ckpt = tf.train.get_checkpoint_state('./bin/')
	      saver.restore(session, ckpt.model_checkpoint_path)
	      feed_dictt = feed_test
	      predictions = session.run( feed_dict=feed_dictt )

    op = len(input)
    for i in range(0, op):
    	data_np = np.asarray(input[i], np.float32)
    	scipy.misc.imsave('video2/inp' + str(i) + '.png', (data_np[1] / 2) + data_np[0])

def costt(input_, target_):
   su = 0.0
   for i in range(100): 
        eps = tf.constant(1e-12)

        input_1 = tf.ones([1, 51, 51], dtype = tf.float32)
        target = target_[i][0]
        weights = target_[i][1]
        target1 = tf.subtract(input_1, target)

        l1 = tf.log(tf.add(eps, input_[i]))
        l2 = tf.log(tf.add(eps, tf.subtract(input_1, input_[i])))
        loss = tf.add(tf.multiply(tf.multiply(weights, target), l1), tf.multiply(tf.multiply(weights, target1), l2))
        y_pred = tf.scalar_mul(1.0/51.0/51.0, loss)
        y_pred = tf.reduce_mean(loss)
        su += y_pred
        #y_pred = tf.reduce_mean(loss)
    
   su = -su 
   return su

def update_grad(input_, target_):
	eps = tf.constant(1e-12)
	input_1 = tf.ones([1, 51, 51], dtype = tf.float32)
	target = target_[0]
	weights = target_[1]
		
	l0 = tf.subtract(target, input_)
	l0 = tf.multiply(weights, l0)
	l1 = tf.add(eps, input_)
		
	l2 = tf.add(eps, tf.subtract(input_1, input_))
	l3 = tf.multiply(l1, l2)

	loss = tf.div(l0, l3)
	y_pred = tf.reduce_mean(loss)

# Inputs
x = tf.placeholder(tf.float32, [1, batch_size, 2*51*51])
y = tf.placeholder(tf.float32, [100, 2, 51, 51])

# Model
with tf.variable_scope('recurrent_layer_1'):
    cell = RWACell(1)
	#state = np.zeros(shape=(1, 51, 51, 32))
	#state = tf.convert_to_tensor(state, np.float32)
    state = cell.zero_state(1, tf.float32)
	#print 'cell.shape: ', cell.shape
    #print 'state: ', state
    out1, h = tf.nn.dynamic_rnn(cell, x, initial_state=state, dtype=tf.float32)

'''
print '################################################'
print 'Done'
print '################################################'
'''

with tf.variable_scope('output_layer'):
    ly = out1
    ly = tf.reshape(ly, [100, 1, 51, 51])  

gpu_options = tf.GPUOptions()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
# Cost function and optimizer
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=ly, labels=y))	# Cross-entropy cost function.
cost = costt(ly, y)
tf.summary.scalar("cost_function", cost)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Merge all summaries into a single operator
merged_summary_op = tf.summary.merge_all()
#writer.add_graph(sess.graph)

# Create operator to initialize session
initializer = tf.global_variables_initializer()
sess.run(initializer)
# Create operator for saving the model and its parameters
saver = tf.train.Saver()
###################################################################
# Session
###################################################################
#print 'PPPPPPPPPPPPPPPPPPrrrrrrrooooooooooooooooooooooooooooooo'
#with tf.Session() as session:
#print 'PPPPPPPPPPPPPPPPPPrrrrrrrooooooooooooooooooooooooooooooo2'
# Initialize variables
#session.run(initializer)
#print 'PPPPPPPPPPPPPPPPPPrrrrrrrooooooooooooooooooooooooooooooo3'
# Each training session represents one batch
summary_writer = tf.summary.FileWriter('./bin/', graph_def=sess.graph_def)
for iteration in range(num_iterations):
    print 'Iterations: ', iteration
    randomm = random.randint(1, 10)
    print randomm
    # Grab a batch of training data
    #print 'PPPPPPPPPPPPPPPPPPrrrrrrrooooooooooooooooooooooooooooooo4'
    xinp = getSequence(randomm)
    xinp = dropoutInput(xinp)
    #print 'PPPPPPPPPPPPPPPPPPrrrrrrrooooooooooooooooooooooooooooooo5'
        
    xs_sequence = np.reshape(xinp, [1, batch_size, 2*51*51])	# Convert image into a sequence of pixels.
    ys = np.reshape(xinp, [batch_size, 2, 51, 51])
    #print 'PPPPPPPPPPPPPPPPPPrrrrrrrooooooooooooooooooooooooooooooo6'
    #print('xs_sequence: ', xs_sequence)
    feed_train = {x: xs_sequence, y: ys}
    # Update parameters
    sess.run(optimizer, feed_dict={x: xs_sequence, y: ys})
    cost_value = sess.run(cost, feed_dict=feed_train)
    #print 'Cost:'
    #print cost_value[0]
    print('Iteration:', iteration, 'Cost:', cost_value/np.log(2.0))
    summary_str = sess.run(merged_summary_op, feed_dict={x: xs_sequence, y: ys})
    #print 'Summ: ', summary_str
    summary_writer.add_summary(summary_str, iteration)

# Save the trained model
saver.save(sess, 'bin/train.ckpt')
#evalModel()