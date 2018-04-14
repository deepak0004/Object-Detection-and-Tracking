import cv2
import os

vidcap = cv2.VideoCapture('vid.mp4')
success,image = vidcap.read()
count = 0
success = True

if not os.path.exists('Break_Video'):
	os.makedirs('Break_Video')

while success:
  success, image = vidcap.read()
  cv2.imwrite("Break_Video/frame%d.jpg" % count, image)     # save frame as JPEG file
  count += 1