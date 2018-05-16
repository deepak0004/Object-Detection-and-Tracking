# Ref https://github.com/abewley/sort/blob/master/sort.py
import cv2
import logging
import gensim, os, sys
import tensorflow as tf
from sklearn.utils.linear_assignment_ import linear_assignment
from tensorflow.contrib import rnn
from tensorflow.python.ops.nn import rnn_cell
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from tracker import re3_tracker
import matplotlib.patches as patches  
import cv2
import random
import pickle
from string import ascii_lowercase

dictt = {}
listtappendd = []
noof = 1
colours = np.random.rand(32, 3) 

def calc_iou(boxA, boxB):
  xA = max(boxA[0], boxB[0])
  yA = max(boxA[1], boxB[1])
  xB = min(boxA[2], boxB[2])
  yB = min(boxA[3], boxB[3])
 
  interArea = (xB - xA + 1) * (yB - yA + 1)
  boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
  boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
  iou = interArea / float(boxAArea + boxBArea - interArea)
 
  return iou

def compp(a, b):
  if( a<=b ):
     return 1
  return 0

def matchh(detections, trackers):
  i = 0
  j = 0
  matrixofiou = np.zeros((len(detections),len(trackers)), dtype=np.float32)
  for det in detections:
    j = 0
    for trk in trackers:
      matrixofiou[i, j] = calc_iou(det, trk)
      j += 1
    i += 1
  matched_indices = linear_assignment(-matrixofiou)

  return matched_indices, matrixofiou

def associate_detections_to_trackers(detections, trackers, iou_threshold = 0.01):  
  detections_unmatched = []
  trackers_unmatched = []      
  matches = []
  matched_indices, matrixofiou = matchh(detections, trackers)
 
  for m in matched_indices:
      if( compp( matrixofiou[ m[0], m[1] ], iou_threshold )==1 ):
        detections_unmatched.append( m[0] )
        trackers_unmatched.append( m[1] )
      else:
        matches.append( m.reshape(1, 2) )
  for d, det in enumerate(detections):
      if(d not in matched_indices[:, 0]):
        detections_unmatched.append(d)
      if(d not in matched_indices[:, 1]):
        trackers_unmatched.append(d)

  if(len(matches)==0):
    matches = np.empty((0,2), dtype=int)
  else:
    matches = np.concatenate(matches, axis=0)

  return matches, np.array(detections_unmatched), np.array(trackers_unmatched)
 
def draww(bounding_boxesss):
  startingpo = 0
  for bbox in bounding_boxesss:
      tup = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
      bbox = bbox.astype(np.float32)
      ax1.add_patch(patches.Rectangle((tup[0], tup[1]), tup[2]-tup[0], tup[3]-tup[1], fill=False, lw=3, ec=colours[listtappendd[startingpo]%32,:]))
      ax1.set_adjustable('box-forced')
      dictt[ listtappendd[startingpo] ] = tup
      startingpo += 1

objtracker = re3_tracker.Re3Tracker()
seq = '/home/deepak/Desktop/Ass2/Re3New/MOT17-01-DPM/det/'
seq_dets = np.loadtxt('/home/deepak/Desktop/Ass2/Re3New/MOT17-01-DPM/det/det.txt', delimiter=',')
length_of_file = int(seq_dets[:,0].max())
plt.ion()
fig = plt.figure()

#initial_bbox = [772.68, 455.43, 41.871, 127.61]
#tracker.track('ball', image_paths[0], initial_bbox)

for frame in range(length_of_file):
  frame += 1
  dets = seq_dets[seq_dets[:,0]==frame,2:6]
  dets[:, 2:4] += dets[:, 0:2] 

  ax1 = fig.add_subplot(111, aspect='equal')
  frameno = str(frame)
  if( len(frameno)<6 ):
     frame = ( (6-len(frameno))*'0' ) + frameno + '.jpg'
  fn = '/home/deepak/Desktop/Ass2/Re3New/MOT17-01-DPM/img1/' + frame
  im = io.imread(fn)
  ax1.imshow(im)
  imageRGB = im
  
  if( len(listtappendd) ):
    bounding_boxes = objtracker.multi_track(listtappendd, imageRGB)
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, bounding_boxes)
        
    dictt2 = {}
    for udets in unmatched_dets:
        listtappendd.append(noof)
        obj = dets[udets,:]
        dictionaryy_value = [obj[0], obj[1], obj[2], obj[3]]
        dictt2[noof] = dictionaryy_value
        dictt[noof] = dictionaryy_value
        noof += 1
    bounding_boxes = objtracker.multi_track(listtappendd, imageRGB, dictt2) 
    for t, trk in enumerate(bounding_boxes):
      if(t not in unmatched_trks):
        d = matched[np.where(matched[:,1]==t)[0],0]
        trk = dets[d,:]
    unmatched_trks = sorted(unmatched_trks, reverse=True) 
    for udets in unmatched_trks:
        indexx = listtappendd[udets]
        del listtappendd[udets]
        del dictt[indexx]
        bounding_boxes = np.delete(bounding_boxes, udets, axis=0)
  else:
    for obj in dets:
       listtappendd.append(noof)
       lisecond = [obj[0], obj[1], obj[2], obj[3]]
       dictt[noof] = lisecond 
       noof += 1
    bounding_boxes = objtracker.multi_track(listtappendd, imageRGB, dictt)

  draww(bounding_boxes)
   
  fig.canvas.flush_events()
  plt.draw()
  ax1.cla()