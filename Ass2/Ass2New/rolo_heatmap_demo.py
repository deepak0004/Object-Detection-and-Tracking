from __future__ import print_function
import cv2
import os
import sys
sys.path.insert(0, "./anshul/")
import helper
import numpy as np
import matplotlib.pyplot as plt


def drawHeatMap(heatmap):
	fig = plt.figure(1, figsize=(10,10))
	ax2 = fig.add_subplot(222)
	ax2.imshow(heatmap, origin='lower', aspect='auto')
	ax2.set_title("heatmap")
	plt.show()


def demoDumpImages(out_dump_dir, time_steps, (img_width, img_height, video_seq, iter1, iter2), imshow = False):
	
	img_dir = "DATA/" + video_seq + "/img/"
	ground_truth_dir = "DATA/" + video_seq + "/groundtruth_rect.txt"
	yolo_dump_dir = "DATA/" + video_seq + "/yolo_out/"
	rolo_dump_dir = "DATA/" + video_seq + "/rolo_heat_test/"

	images_files = sorted(os.listdir(img_dir))
	rolo_files = sorted(os.listdir(rolo_dump_dir))

	images_files = [os.path.join(img_dir, file) for file in images_files]
	rolo_files = [os.path.join(rolo_dump_dir, file) for file in rolo_files]

	with open(ground_truth_dir, "r") as file:
		ground_truth = file.readlines()

	out_dump_dir += video_seq

	count = 0
	rolo_iou = 0
	yolo_iou = 0

	for index in xrange(len(rolo_files) - time_steps):
		id = index + 1
		img_id = id + time_steps - 2

		img_path = images_files[img_id]
		img = cv2.imread(img_path)

		
		#yolo_predictions = helper.loadYoloPredictionsDemo(img_id, yolo_dump_dir)
		#yolo_predictions = yolo_predictions[0][4097:4101]
		#yolo_predictions = helper.getRealCoordinatesDemo(img_width, img_height, yolo_predictions)
		#print(yolo_predictions)
		#img = helper.drawBBox(img, yolo_predictions, color=(255, 0, 0), id=0)

		heatmapVector = helper.loadRoloPredictions(img_id, rolo_dump_dir)
		heatmapVector = helper.heatmapVectorToHeatMap(heatmapVector)
		drawHeatMap(heatmapVector)
		


		#rolo_predictions = helper.loadRoloPredictions(img_id, rolo_dump_dir)
		#rolo_predictions = helper.getRealCoordinatesDemo(img_width, img_height, rolo_predictions)
		#print(rolo_predictions)
		#img = helper.drawBBox(img, rolo_predictions, color=(0, 255, 0), id=1)

		#gt_predictions = helper.loadGroundTruthDemo(ground_truth, img_id-1)
		#print(gt_predictions)
		#img = helper.drawBBox(img, gt_predictions, color=(0, 0, 255), id=2)
		print(heatmapVector)

		
		if not os.path.exists(out_dump_dir):
			os.mkdir(out_dump_dir)

		filename = out_dump_dir + "/" + str(img_id) + ".jpg"
		print(filename)
		#cv2.imwrite(filename, img)
		cv2.imwrite(filename, heatmapVector)

		"""
		if imshow:
			cv2.imshow('frame', img)
			cv2.waitKey(100)


		rolo_iou += helper.getIOUPredictions(rolo_predictions, gt_predictions)
		yolo_iou += helper.getIOUPredictions(yolo_predictions, gt_predictions)
		"""
		count += 1

	"""
	rolo_iou /= float(count)
	yolo_iou /= float(count)
	print("YOLO average IOU is {}".format(yolo_iou))
	print("ROLO average IOU is {}".format(rolo_iou))
	"""


def main():
	video_id = 21
	time_steps = 6
	out_dump_dir = "output/frames_new/"
	demoDumpImages(out_dump_dir, time_steps, helper.getVideoSequenceMetadata(video_id), imshow=False)


if __name__ == '__main__':
	main()