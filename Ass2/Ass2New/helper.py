import os
import cv2
import numpy as np


def getRealCoordinates(width, height, locs):
	for index in xrange(len(locs)):
		locs[index][0] *= float(width)
		locs[index][1] *= float(height)
		locs[index][2] *= float(width)
		locs[index][3] *= float(height)
	return locs


def getRealCoordinatesDemo(width, height, locs):
	locs[0] *= float(width)
	locs[1] *= float(height)
	locs[2] *= float(width)
	locs[3] *= float(height)
	return locs


def getNormalizedCoordinates(width, height, locs):
	for index in xrange(len(locs)):
		locs[index][0] += locs[index][2] / 2.0
		locs[index][1] += locs[index][3] / 2.0

		locs[index][0] /= float(width)
		locs[index][1] /= float(height)
		locs[index][2] /= float(width)
		locs[index][3] /= float(height)

	return locs


def dumpRoloPredictions(predictions, index, timestep, batch_size, out_dir):
	offset = index * batch_size * timestep - 2

	for batch in xrange(batch_size):
		img_id = offset + (batch + 1)*timestep + 1
		np.save(os.path.join(out_dir, str(img_id)), predictions[batch])


def loadRoloGroundTruth(index, time_steps, batch_size, gt_dir):
	with open(gt_dir, "r") as file:
		ground_truth = file.readlines()

	offset = time_steps - 2
	start_id = index + offset
	end_id = index + batch_size * time_steps + offset

	ground_truth_locations = []
	for index in xrange(start_id, end_id, time_steps):
		line = ground_truth[index].split("\t")
		if len(line) < 4:
			line = ground_truth[index].split(",")

		location = [int(line[0]), int(line[1]), int(line[2]), int(line[3])]
		ground_truth_locations.append(location)

	return ground_truth_locations


def loadGroundTruthDemo(ground_truth, index):
	line = ground_truth[index].split("\t")
	if len(line) < 4:
		line = ground_truth[index].split(",")

	location = [int(line[0]), int(line[1]), int(line[2]), int(line[3])]
	return location
		

def loadYoloPredictions(index, time_steps, batch_size, yolo_pred_size, yolo_dir):
	files = os.listdir(yolo_dir)
	files = [os.path.join(yolo_dir, file) for file in files]
	files = sorted(files)

	start_id = index
	end_id = index + batch_size * time_steps

	yolo_predictions = []
	for file in files[start_id:end_id]:
		yolo_pred = np.load(file)
		yolo_pred = np.reshape(yolo_pred, yolo_pred_size)
		yolo_predictions.append(yolo_pred)

	yolo_predictions = np.reshape(yolo_predictions, [batch_size * time_steps, yolo_pred_size])
	return yolo_predictions


def getVideoSequenceMetadata(video_id):
	#OTB30 Dataset Video Sequences 0-29
	#MOT Dataset Video Sequences 30-43
	#For Testing Only (Outside OTB30) 90-97
	video_sequence = {0: [480, 640, 'Human2', 250, 1128], 1: [320, 240, 'Human9', 70, 302], 2: [320, 240, 'Suv', 314, 943], 3: [640, 480, 'BlurBody', 111, 334], 4: [640, 480, 'BlurCar1', 247, 742], 5: [352, 240, 'Dog', 42, 127], 6: [624, 352, 'Singer2', 121, 366], 7: [352, 288, 'Woman', 198, 597], 8: [640, 480, 'David3', 83, 252], 9: [320, 240, 'Human7', 83, 250], 10: [720, 400, 'Bird1', 135, 408], 
					11: [360, 240, 'Car4', 219, 659], 12: [320, 240, 'CarDark', 130, 393], 13: [320, 240, 'Couple', 46, 140], 14: [400, 224, 'Diving', 71, 214], 15: [480, 640, 'Human3', 565, 1698], 16: [480, 640, 'Human6', 263, 792], 17: [624, 352, 'Singer1', 116, 351], 18: [384, 288, 'Walking2', 166, 500], 19: [640, 480, 'BlurCar3', 117, 356], 20: [640, 480, 'Girl2', 499, 1500], 
					21: [640, 360, 'Skating1', 133, 400], 22: [320, 240, 'Skater', 50, 160], 23: [320, 262, 'Skater2', 144, 435], 24: [320, 246, 'Dancer', 74, 225], 25: [320, 262, 'Dancer2', 49, 150], 26: [640, 272, 'CarScale', 81, 252], 27: [426, 234, 'Gym', 255, 767], 28: [320, 240, 'Human8', 42, 128], 29: [416, 234, 'Jump', 40, 122],
					
    				30: [1920, 1080, 'MOT17-02', 199, 600], 31: [1920, 1080, 'MOT17-04', 349, 1050], 32: [640, 480, 'MOT17-05', 278, 837], 33: [1920, 1080, 'MOT17-09', 174, 525], 34: [1920, 1080, 'MOT17-10', 217, 654], 35: [1920, 1080, 'MOT17-11', 299, 900], 36: [1920, 1080, 'MOT17-13', 249, 750], 
    				37: [1920, 1080, 'MOT17-01', 149, 450], 38: [1920, 1080, 'MOT17-03', 499, 1500], 39: [640, 480, 'MOT17-06', 397, 1194], 40: [1920, 1080, 'MOT17-07', 166, 500], 41: [1920, 1080, 'MOT17-08', 208, 625], 42: [1920, 1080, 'MOT17-12', 299, 900], 43: [1920, 1080, 'MOT17-14', 249, 750],
    				
    				90: [352, 288, 'Jogging_1', 100, 300], 91: [352, 288, 'Jogging_2', 100, 300], 92: [640, 480, 'Boy', 199, 602], 93: [352, 288, 'Jumping', 103, 313], 94: [480, 360, 'Surfer', 125, 376], 
    				95: [640, 332, 'Trans', 41, 124], 96: [640, 360, 'DragonBaby', 37, 113], 97: [640, 480, 'Liquor', 580, 1741]}
	return video_sequence[video_id]


def locToHeatmapCoordinates(loc):
	loc = [i * 32 for i in loc]

	x1 = int(loc[0] - loc[2]/2)
	y1 = int(loc[1] - loc[3]/2)
	x2 = int(loc[0] + loc[2]/2)
	y2 = int(loc[1] + loc[3]/2)

	return [x1, y1, x2, y2]


def heatmapCoordinatesToHeatmapVector(coordinates):
	heatmap_vec = np.zeros(1024)

	[x1, y1, x2, y2] = coordinates
	for y in xrange(y1, y2):
		for x in xrange(x1, x2):
			heatmap_vec[x + y*32] = 1.0

	return heatmap_vec


def heatmapVectorToHeatMap(heatmap_vector):
	heatmap = np.zeros((32, 32))

	for y in xrange(32):
		for x in xrange(32):
			heatmap[y][x] = heatmap_vector[x + y*32]

	return heatmap


def dumpRoloPredictions(heatmap, index, timestep, batch_size, out_dir):
	offset = index - 2

	for batch in xrange(batch_size):
		img_id = offset + (batch + 1) * timestep + 1
		np.save(os.path.join(out_dir, str(img_id).zfill(4)), heatmap[batch])


def loadRoloPredictions(index, rolo_dir):
	return np.load(os.path.join(rolo_dir, str(index) + ".npy"))


def loadYoloPredictionsDemo(index, yolo_dir):
	return np.load(os.path.join(yolo_dir, str(index).zfill(4) + ".npy"))


def getIOU(B1, B2):
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


def getIOUPredictions(pred1, pred2):
	pred1[0] = pred1[0] - pred1[2]/2
	pred1[1] = pred1[1] - pred1[3]/2

	return getIOU(pred1, pred2)


def drawBBox(img, bbox, color, id):
	x = int(bbox[0])
	y = int(bbox[1])
	w = int(bbox[2])
	h = int(bbox[3])

	if id == 2:
		cv2.rectangle(img, (x,y), (x+w,y+h), color, 2)
	else:
		cv2.rectangle(img, (x-w//2, y-h//2),(x+w//2,y+h//2), color, 2)

	return img