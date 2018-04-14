#https://github.com/Guanghan/ROLO/blob/master/3rd%20party/YOLO_network.py
import logging
import gensim, os, sys
import helper
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.ops.nn import rnn_cell
import numpy as np
import cv2
import random
import pickle
from string import ascii_lowercase

slim = tf.contrib.slim
def leaky_relu(alpha):
	print 'Mo'
	def op(inputs):
		return tf.nn.leaky_relu(inputs, alpha=alpha, name='leaky_relu')
	return op

class YOLO_TF:
	classes =  ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
	"chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", 
	"sheep", "sofa", "train","tvmonitor"]
	tofile_img = 'test/output.jpg'
	tofile_txt = 'test/output.txt'
	#imshow = True
	weights_file = 'weights/YOLO_small.ckpt'
	alpha = 0.1
	num_class = 20
	num_box = 2
	num_heatmap = 1024
	threshold = 0.08
	grid_size = 7
	w_img, h_img = [352, 240]
	num_feat = 4096
	num_predict = 6 

	def __init__(self):
		self.build_networks()
					
    # https://github.com/hizhangp/yolo_tensorflow/blob/88aba9d5569c04170f294a093455390a90f2686e/yolo/yolo_net.py
	def build_networks(self):
		print self.grid_size
		print 'Building'
		with tf.variable_scope('yolo'):
			with slim.arg_scope(
	            [slim.conv2d, slim.fully_connected],
	            activation_fn=leaky_relu(0.1),
	            weights_regularizer=slim.l2_regularizer(0.0005),
	            weights_initializer=tf.truncated_normal_initializer(0.0, 0.01)
	        ):
				print 'Hiiiiiiii'
				self.x = tf.placeholder('float32',[None,448,448,3])
				#self.net = tf.pad(self.x, np.array([[0, 0], [3, 3], [3, 3], [0, 0]]), name='pad_1')
				self.net = slim.conv2d(self.x, 64, 7, 2, padding='VALID', scope='conv_2')
				self.net = slim.max_pool2d(self.net, 2, padding='SAME', scope='pool_3')
				self.net = slim.conv2d(self.net, 192, 3, scope='conv_4')
				self.net = slim.max_pool2d(self.net, 2, padding='SAME', scope='pool_5')
				self.net = slim.conv2d(self.net, 128, 1, scope='conv_6')
				self.net = slim.conv2d(self.net, 256, 3, scope='conv_7')
				self.net = slim.conv2d(self.net, 256, 1, scope='conv_8')
				self.net = slim.conv2d(self.net, 512, 3, scope='conv_9')
				self.net = slim.max_pool2d(self.net, 2, padding='SAME', scope='pool_10')
				self.net = slim.conv2d(self.net, 256, 1, scope='conv_11')
				self.net = slim.conv2d(self.net, 512, 3, scope='conv_12')
				self.net = slim.conv2d(self.net, 256, 1, scope='conv_13')
				self.net = slim.conv2d(self.net, 512, 3, scope='conv_14')
				self.net = slim.conv2d(self.net, 256, 1, scope='conv_15')
				self.net = slim.conv2d(self.net, 512, 3, scope='conv_16')
				self.net = slim.conv2d(self.net, 256, 1, scope='conv_17')
				self.net = slim.conv2d(self.net, 512, 3, scope='conv_18')
				self.net = slim.conv2d(self.net, 512, 1, scope='conv_19')
				self.net = slim.conv2d(self.net, 1024, 3, scope='conv_20')
				self.net = slim.max_pool2d(self.net, 2, padding='SAME', scope='pool_21')
				self.net = slim.conv2d(self.net, 512, 1, scope='conv_22')
				self.net = slim.conv2d(self.net, 1024, 3, scope='conv_23')
				self.net = slim.conv2d(self.net, 512, 1, scope='conv_24')
				self.net = slim.conv2d(self.net, 1024, 3, scope='conv_25')
				self.net = slim.conv2d(self.net, 1024, 3, scope='conv_26')
				self.net = tf.pad(
		            self.net, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]),
		            name='pad_27')
				self.net = slim.conv2d(
		            self.net, 1024, 3, 2, padding='VALID', scope='conv_28')
				self.net = slim.conv2d(self.net, 1024, 3, scope='conv_29')
				self.net = slim.conv2d(self.net, 1024, 3, scope='conv_30')
				self.net = tf.transpose(self.net, [0, 3, 1, 2], name='trans_31')
				self.net = slim.flatten(self.net, scope='flat_32')
				self.net = slim.fully_connected(self.net, 512, scope='fc_33')
				self.fc_30 = slim.fully_connected(self.net, 4096, scope='fc_34')
				#self.net = slim.dropout(self.net, keep_prob=keep_prob, is_training=is_training, scope='dropout_35')
				self.fc_32 = slim.fully_connected(self.fc_30, 1470, activation_fn=None, scope='fc_36')
		        self.sess = tf.Session()
		        self.sess.run(tf.global_variables_initializer())
		        self.saver = tf.train.Saver()
		        self.saver.restore(self.sess,self.weights_file)
		        print 'Done'

	def comp(self, a, b):
		if( a>b ):
	   		return 1
	  	return 0

	def newfunc(self, fil, pro):
		thresh = 0.5
		lengfi = len(fil)
		for i in range(lengfi):
			if( pro[i]!=0 ): 
				for j in range(i+1, lengfi):
					if self.comp(self.iou(fil[i], fil[j]), thresh): 
						pro[j] = 0.0
		return pro

	def calcres(self, bfil, clf, pro):
		res = []
		leng = len(bfil)
		for i in range(leng):
			res.append([self.classes[clf[i]],bfil[i][0],bfil[i][1],bfil[i][2],bfil[i][3],pro[i]])
		return res

	def mult(self, procla, sca):
		resu = np.zeros((7,7,2,20))
		up = 2
		lo = 20
		for i in range(up):
			for j in range(lo):
				resu[:,:,i,j] = np.multiply(procla[:,:,j], sca[:,:,i])
		return resu 	

	def outp_getting(self,output):
		widthh = self.w_img
		heightt = self.h_img

		probs = np.zeros((7, 7, 2, 20))
		class_probs = np.reshape(output[0:980],(7,7,20))
		scales = np.reshape(output[980:1078],(7,7,2))
		boundingg = np.reshape(output[1078:],(7,7,2,4))
		offset = np.transpose(np.reshape(np.array([np.arange(7)]*14),(2,7,7)),(1,2,0))

		boundingg[:,:,:,0] = boundingg[:,:,:,0] + offset
		boundingg[:,:,:,1] = boundingg[:,:,:,1] + np.transpose(offset,(1,0,2))
		boundingg[:,:,:,0:2] = boundingg[:,:,:,0:2] / 7.0
		boundingg[:,:,:,2] = np.multiply(boundingg[:,:,:,2],boundingg[:,:,:,2])
		boundingg[:,:,:,3] = np.multiply(boundingg[:,:,:,3],boundingg[:,:,:,3])
		
		boundingg[:,:,:,0] = boundingg[:,:,:,0] * widthh
		boundingg[:,:,:,1] = boundingg[:,:,:,1] * heightt
		boundingg[:,:,:,2] = boundingg[:,:,:,2] * widthh
		boundingg[:,:,:,3] = boundingg[:,:,:,3] * heightt

		probs = self.mult(class_probs, scales)

		filter_mat_probs = np.array(probs>=self.threshold,dtype='bool')
		filter_mat_boundingg = np.nonzero(filter_mat_probs)
		bfil = boundingg[filter_mat_boundingg[0],filter_mat_boundingg[1],filter_mat_boundingg[2]]
		probs_filtered = probs[filter_mat_probs]
		clf = np.argmax(filter_mat_probs,axis=3)[filter_mat_boundingg[0],filter_mat_boundingg[1],filter_mat_boundingg[2]] 

		argsort = np.array(np.argsort(probs_filtered))[::-1]
		bfil = bfil[argsort]
		probs_filtered = probs_filtered[argsort]
		clf = clf[argsort]
		
		probs_filtered = self.newfunc(bfil, probs_filtered)
		
		filter_iou = np.array(probs_filtered>0.0,dtype='bool')
		bfil = bfil[filter_iou]
		probs_filtered = probs_filtered[filter_iou]
		clf = clf[filter_iou]

		res = self.calcres(bfil, clf, probs_filtered)

		return res

	def iou(self, B1, B2):
		    bbox1 = [float(x) for x in B1]
		    bbox2 = [float(x) for x in B2]

		    (cx_11, cy_11, w1, h1) = bbox1
		    (cx_22, cy_22, w2, h2) = bbox2

		    (x0_1, y0_1, x1_1, y1_1) = bbox1
		    (x0_2, y0_2, x1_2, y1_2) = bbox2

		    x0_1 = cx_11 - (w1/2)
		    y0_1 = cy_11 - (h1/2)
		    x1_1 = cx_11 + (w1/2)
		    y1_1 = cy_11 + (h1/2)

		    x0_2 = cx_22 - (w2/2)
		    y0_2 = cy_22 - (h2/2)
		    x1_2 = cx_22 + (w2/2)
		    y1_2 = cy_22 + (h2/2)

		    # get the overlap rectangle
		    overlap_x0 = max(x0_1, x0_2)
		    overlap_y0 = max(y0_1, y0_2)
		    overlap_x1 = min(x1_1, x1_2)
		    overlap_y1 = min(y1_1, y1_2)

		    # check if there is an overlap
		    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
		        return 0

		    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
		    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
		    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
		    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
		    size_union = size_1 + size_2 - size_intersection

		    return size_intersection / size_union

	def find_best_location(self, locations, gt_location):
			for i in range(1):
				x1, y1, w, h = gt_location[i], gt_location[i+1], gt_location[i+2], gt_location[i+3]
			max_ious = 0 
			gt_location_revised = [x1 + w/2, y1 + h/2, w, h]
			for id, location in enumerate(locations):
					location_revised = location[1:5]
					ious = self.iou(location_revised, gt_location_revised)
					if ious >= max_ious:
							max_ious = ious
							index = id
							best_location = locations[index]
							class_index = self.classes.index(best_location[0])
							best_location[0] = class_index

			if max_ious != 0:
				return best_location
			return [0, 0, 0, 0, 0, 0]

	def img_draww(self, x, y, w, h, backupimg, location): 
		subx = x-w
		suby = y-h
		addx = x+w
		addy = y+h
		poss = location[0]

		cv2.rectangle(backupimg,(subx, suby),(addx, addy),(0,255,0),2)
		cv2.rectangle(backupimg, (subx, suby-20), (addx, suby), (125,125,125), -1)
		cv2.putText(backupimg, str(poss) + ' : %.2f' % location[5], (subx+5, suby-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
		#cv2.imshow('YOLO_small detection', backupimg)
		cv2.waitKey(1)

	def func22(self, img):
		img = cv2.resize(img, (448, 448))
		img = np.asarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
		inputs = np.zeros((1, 448, 448, 3), dtype='float32')
		inputs[0] = (img/255.0) * 2.0 - 1.0
		
		feature = self.sess.run(self.fc_30, feed_dict = {self.x : inputs})
		output = self.sess.run(self.fc_32, feed_dict = {self.x : inputs}) 
        
		return feature, output


	def prepare_training_data(self, imagefold, gt_file, out_fold):  #[or]prepare_training_data(self, list_file, gt_file, out_fold):
		paths = os.listdir(imagefold)
		paths = [os.path.join(imagefold, p) for p in paths]
		paths = sorted(paths)

		txtfile = open(gt_file, "r")
		gt_locations = txtfile.read().split('\n')  

		for id, path in enumerate(paths):
			filename = os.path.basename(path)
			print 'Image: ', id
			img = cv2.imread(path)

			self.h_img, self.w_img, _ = img.shape
			feature, output = self.func22(img)

			locations = self.outp_getting(output[0])
			lineeid = gt_locations[id]
			elems = lineeid.split('\t')   
			if len(elems) < 4:
				elems = lineeid.split(',') 
			x1, y1, w, h = elems[0], elems[1], elems[2], elems[3]
			gt_location = [int(x1), int(y1), int(w), int(h)]

			#find the ROI that has the maximum IOU with the ground truth
			location = self.find_best_location(locations, gt_location) 
	
			backupimg = img.copy()
			x, y, w, h = int(location[1]), int(location[2]), int(location[3])//2, int(location[4])//2
			
			widthh = self.w_img
			heightt = self.w_img

			for i in xrange(1,5):
				if i % 2 == 0:
					location[i] /= heightt
				else:
					location[i] /= widthh

			feature = np.reshape(feature, [-1, self.num_feat])
			location = np.reshape(location, [-1, self.num_predict])
			
			yolo_output=  np.concatenate((feature, location), axis = 1)

			path = os.path.join(out_fold, os.path.splitext(filename)[0])
			np.save(path, yolo_output)

def main():
	yolo = YOLO_TF()
	checker = 10
	[yolo.w_img, yolo.h_img, sequence_name, _, __] = helper.getVideoSequenceMetadata(checker)

	folderdir = '/home/deepak/Desktop/ROLO/DATA/'
	imagefold = '/home/deepak/Desktop/ROLO/DATA/' + sequence_name + '/img/'

	gt_file = folderdir + sequence_name + '/groundtruth_rect.txt'
	out_fold = folderdir + sequence_name + '/yolo_out/'
	heat_fold = folderdir + sequence_name + '/yolo_heat/'
	if not os.path.exists(out_fold):
		os.makedirs(out_fold)
	if not os.path.exists(heat_fold):
		os.makedirs(heat_fold)

	#if heatmap is True:
	#	yolo.prepare_training_data_heatmap(imagefold, gt_file, heat_fold)
	#else:
		#if (test >= 0 and test <= 29):
	yolo.prepare_training_data(imagefold, gt_file, out_fold)

main()