from __future__ import print_function
import os
import numpy as np
import tensorflow as tf

import sys
sys.path.insert(0, "./anshul/")
import helper

class ROLO():

	def __init__(self):
		#Set it to 0 to enable GPU
		os.environ['CUDA_VISIBLE_DEVICES'] = ''

		self.rolo_weights_dir = "/home/sharat/Documents/assignment2_new/ROLO/weights/trainExp2a"
		self.rolo_weights_intermediate_dir = "/home/sharat/Documents/assignment2_new/ROLO/weights/trainExp2a/intermediate"
		self.summary_dir = "/home/sharat/Documents/assignment2_new/ROLO/summaryNew/"
		self.use_weights = False

		if not os.path.exists(self.rolo_weights_dir):
			os.mkdir(self.rolo_weights_dir)

		if not os.path.exists(self.rolo_weights_intermediate_dir):
			os.mkdir(self.rolo_weights_intermediate_dir)

		if not os.path.exists(self.summary_dir):
			os.mkdir(self.summary_dir)

		self.rolo_weights_dir += "model.ckpt"
		self.rolo_weights_intermediate_dir += "model.ckpt"

		self.time_steps = 3
		self.feature_size = 4096
		self.predict_size = 6
		self.gt_size = 4
		self.yolo_output_size = self.feature_size + self.predict_size

		self.lr = 1e-5
		self.batch_size = 1

		self.num_videos = 30
		self.offset = 0
		self.num_epoches = self.num_videos * 100
		self.display_count = 1
		self.summary_count = 50

		self.config = tf.ConfigProto()
		#self.config.gpu_options.allow_growth = True

		self.createPlaceholders()
		self.model()
		self.loss()


	def createPlaceholders(self):
		self.x = tf.placeholder(tf.float32, [None, self.time_steps, self.yolo_output_size])
		self.y = tf.placeholder(tf.float32, [None, self.gt_size])
		self.state = tf.placeholder(tf.float32, [None, 2 * self.yolo_output_size])


	def model(self):
		Xi = tf.transpose(self.x, [1, 0, 2])
		Xi = tf.reshape(Xi, [self.time_steps * self.batch_size, self.yolo_output_size])
		Xi = tf.split(Xi, self.time_steps, 0)

		lstm = tf.nn.rnn_cell.LSTMCell(self.yolo_output_size, self.yolo_output_size, state_is_tuple = False)
		istate = self.state

		for time_step in xrange(self.time_steps):
			output, istate = tf.nn.static_rnn(lstm, [Xi[time_step]], istate)
			tf.get_variable_scope().reuse_variables()

		self.lstm_output = output
		self.box_prediction = self.lstm_output[0][:, 4097:4101]


	def loss(self):
		loss_sqr = tf.square(self.box_prediction - self.y)
		self.loss = tf.reduce_mean(loss_sqr) * 100
		tf.summary.scalar('Loss', self.loss)


	def trainModel(self):
		self.merged = tf.summary.merge_all()

		with tf.variable_scope(tf.get_variable_scope(), reuse = tf.AUTO_REUSE):
			self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

		initialize = tf.global_variables_initializer()

		with tf.Session(config = self.config) as sess:
			self.saver = tf.train.Saver(max_to_keep=1)
			self.saver_intermediate = tf.train.Saver(max_to_keep=0)
			self.train_writer = tf.summary.FileWriter(self.summary_dir, sess.graph)

			sess.run(initialize)
			
			if self.use_weights:
				self.saver.restore(sess, self.rolo_weights_dir)
				

			iterator = 1

			for epoch in xrange(self.num_epoches):
				video_id = epoch % self.num_videos + self.offset
				img_w, img_h, video_seq, train_iters, _ = helper.getVideoSequenceMetadata(video_id)
				
				x_dir = os.path.join("DATA", video_seq, "yolo_out")
				y_dir = os.path.join("DATA", video_seq, "groundtruth_rect.txt")
				out_dir = os.path.join("DATA", video_seq, "rolo_out_train")

				if not os.path.exists(out_dir):
					os.mkdir(out_dir)

				total_loss = 0
				index = 0

				while index < train_iters - self.time_steps:
					inputs_x = helper.loadYoloPredictions(index, self.time_steps, self.batch_size, self.yolo_output_size, x_dir)
					inputs_x = np.reshape(inputs_x, [self.batch_size, self.time_steps, self.yolo_output_size])

					inputs_y = helper.loadRoloGroundTruth(index, self.time_steps, self.batch_size, y_dir)
					inputs_y = helper.getNormalizedCoordinates(img_w, img_h, inputs_y)
					inputs_y = np.reshape(inputs_y, [self.batch_size, self.gt_size])

					input_dict = {self.x: inputs_x, self.y: inputs_y, self.state: np.zeros((self.batch_size, 2 * self.yolo_output_size))}

					sess.run(self.optimizer, feed_dict = input_dict)

					if index % self.display_count == 0:
						loss = sess.run(self.loss, feed_dict = input_dict)
						print("Epoch: {}, Index: {}, Loss: {}, Video Sequence Name: {}".format(epoch, index, "{:.6f}".format(loss), video_seq))
						total_loss += loss

					if index % self.summary_count == 0:
						summary = sess.run(self.merged, feed_dict = input_dict)
						self.train_writer.add_summary(summary, iterator)
						iterator += 1

					index += 1

				avg_loss = total_loss/index
				print("Epoch: {}, Video Name: {}, Average Loss: {}, Sequence Name: {}".format(epoch, video_seq, avg_loss, video_seq))

				model_save_path = self.saver.save(sess, self.rolo_weights_dir, global_step = epoch)
				print("Model Checkpoint files dumped at {}:{}".format(epoch, model_save_path))

				if epoch % 50:
					save_path = self.saver_intermediate.save(sess, self.rolo_weights_intermediate_dir, global_step = epoch)
					print("Intermediate Model Checkpoint files dumped at {}:{}".format(epoch, save_path))


	def testModel(self, video_id, out_dir = "rolo_out_train"):
		with tf.Session(config = self.config) as sess:
			self.saver = tf.train.Saver()
			
			self.saver.restore(sess, self.rolo_weights_dir)
			#self.saver.restore(sess, "/home/sharat/Documents/assignment2_new/ROLO/weights/train/model_step6_exp1.ckpt")
				
			img_w, img_h, video_seq, _, test_iters = helper.getVideoSequenceMetadata(video_id)

			x_dir = os.path.join("DATA", video_seq, "yolo_out")
			y_dir = os.path.join("DATA", video_seq, "groundtruth_rect.txt")
			out_dir = os.path.join("DATA", video_seq, out_dir)

			if not os.path.exists(out_dir):
				os.mkdir(out_dir)

			total_loss = 0
			index = 0

			while index < test_iters - self.time_steps:
				inputs_x = helper.loadYoloPredictions(index, self.time_steps, self.batch_size, self.yolo_output_size, x_dir)
				inputs_x = np.reshape(inputs_x, [self.batch_size, self.time_steps, self.yolo_output_size])

				inputs_y = helper.loadRoloGroundTruth(index, self.time_steps, self.batch_size, y_dir)
				inputs_y = helper.getNormalizedCoordinates(img_w, img_h, inputs_y)
				inputs_y = np.reshape(inputs_y, [self.batch_size, self.gt_size])

				input_dict = {self.x: inputs_x, self.y: inputs_y, self.state: np.zeros((self.batch_size, 2 * self.yolo_output_size))}

				location_pred = sess.run(self.box_prediction, feed_dict = input_dict)

				location_pred = helper.getRealCoordinates(img_w, img_h, location_pred)
				location_pred = np.reshape(location_pred, [self.batch_size, self.gt_size])

				helper.dumpRoloPredictions(location_pred, index, self.time_steps, self.batch_size, out_dir)

				if index % self.display_count == 0:
					loss = sess.run(self.loss, feed_dict = input_dict)
					print("Index: {}, Loss: {}, Predictions: {}, Video Sequence Name: {}".format(index, "{:.6f}".format(loss), location_pred, video_seq))
					total_loss += loss

				index += 1

			avg_loss = total_loss/index
			print("Video Name: {}, Average Loss: {}".format(video_seq, avg_loss))


def main():
	rolo = ROLO()
	#rolo.testModel(video_id = 21)
	rolo.trainModel()

if __name__ == '__main__':
	main()