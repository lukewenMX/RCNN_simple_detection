#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

import rospy
from ros_faster_rcnn.msg import *
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import threading

RUNNING = False
IMAGE1 = Image()
IMAGE2 = Image()
	
def detect(image):
	"""Detect object classes in an image using pre-computed object proposals."""
	global NET
	
	# Detect all object classes and regress object bounds
	rospy.loginfo("Starting detection")
	timer = Timer()
	timer.tic()
	scores, boxes = im_detect(NET, image)
	timer.toc()
	rospy.loginfo('Detection took %f seconds for %d object proposals', timer.total_time, boxes.shape[0])
	return (scores, boxes)
			
'''def imageCallbackColor(im):
	global IMAGE1
	global RUNNING
	#rospy.loginfo('Color Image received')
	if (RUNNING):
		pass
		#rospy.logwarn('Color Image Detection already running, message omitted')
	else:
		RUNNING = True
		mutex.acquire()
		IMAGE1 = im
		mutex.release()'''
def imageCallbackThermal(im):
	global IMAGE2
	global RUNNING
	#rospy.loginfo("Thermal Image received!")
	if (RUNNING):
		pass
		#rospy.logwarn("Thermal Image Detection already running, message omitted")
	else:
		RUNNING = True
		mutex.acquire()
		IMAGE2 = im
		mutex.release()
		rospy.logwarn('IMAGE2 succeed! the header is %s',IMAGE2.header.seq)

		
		#IMAGE2 = cv2.cvtColor(, cv2.COLOR_BGR2RGB)
	

def parse_args():
	"""Parse input arguments."""
	# Filter roslaunch arguments
	sys.argv = filter(lambda arg: not arg.startswith('__'), sys.argv)
	sys.argc = len(sys.argv)
	
	# Parse the other arguments
	parser = argparse.ArgumentParser(description='Faster R-CNN demo')
	parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
						default=0, type=int)
	parser.add_argument('--cpu', dest='cpu_mode',
						help='Use CPU mode (overrides --gpu)',
						action='store_true')
	parser.add_argument('--tresh', dest='treshold', help='The treshold for the detection', 
						default=0.8, type=float)
	parser.add_argument('--prototxt', dest='prototxt', help='The proto file', 
						default='../libraries/py-faster-rcnn/models/pascal_voc/VGG16/faster_rcnn_alt_opt/faster_rcnn_test.pt')
	parser.add_argument('--model', dest='model', help='The model file', 
						default='../libraries/py-faster-rcnn/data/faster_rcnn_models/VGG16_faster_rcnn_final.caffemodel')
	parser.add_argument('--classes', dest='classes', help='The file containing the classes', 
						default='classes.txt')

	args = parser.parse_args()

	return args
    
def parseClasses(classFile):
	with open(classFile) as f:
		content = f.readlines()
	return ['__background__'] + map(lambda x: x[:-1], content)

def generateDetections (scores, boxes, classes, threshold):
	# Visualize detections for each class
	NMS_THRESH = 0.3	
	res = []

	for cls_ind, cls in enumerate(classes[1:]):
		cls_ind += 1 # because we skipped background
		cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
		cls_scores = scores[:, cls_ind]
		dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
		keep = nms(dets, NMS_THRESH)
		dets = dets[keep, :]

		inds = np.where(dets[:, -1] >= threshold)[0]

		for i in inds:
			bbox = dets[i, :4]
			score = dets[i, -1]
			
			msg = Detection()
			msg.header.stamp = rospy.Time.now()
			msg.x = bbox[0]
			msg.y = bbox[1]
			msg.width =  bbox[2] - bbox[0]
			msg.height = bbox[3] - bbox[1]
			msg.object_class = classes[cls_ind]
			msg.p = score
			res.append(msg)
	return res

'''def generateDetectionsThermal (scores, boxes, classes, threshold):
	# Visualize detections for each class
	NMS_THRESH = 0.3	
	res = []

	for cls_ind, cls in enumerate(classes[1:]):
		cls_ind += 1 # because we skipped background
		cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
		cls_scores = scores[:, cls_ind]
		dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
		keep = nms(dets, NMS_THRESH)
		dets = dets[keep, :]

		inds = np.where(dets[:, -1] >= threshold)[0]

		for i in inds:
			bbox = dets[i, :4]
			score = dets[i, -1]
			
			msg = Detection()
			msg.x = bbox[0]
			msg.y = bbox[1]
			msg.width =  bbox[2] - bbox[0]
			msg.height = bbox[3] - bbox[1]
			msg.object_class = classes[cls_ind]
			msg.p = score
			res.append(msg)
	return res'''
	
def getResultImageColor (detections, image):
	font = cv2.FONT_HERSHEY_SIMPLEX
	textSize = cv2.getTextSize("test", font, 1, 2)
	delta = (textSize[1] * .3, textSize[1] * 2.4)
		
	for det in detections:
		cv2.rectangle(image, (det.x, det.y), (det.x + det.width, det.y + det.height), (255, 0, 0), 3)
		text = "{}: p={:.2f}".format(det.object_class, det.p)
		cv2.putText(image, text, (int(det.x + delta[0]), int(det.y + delta[1])), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
	return image
	

'''def getResultImageThermal(detections,image):
	font = cv2.FONT_HERSHEY_SIMPLEX
	textSize = cv2.getTextSize("test", font, 1, 2)
	delta = (textSize[1] * .3, textSize[1] * 2.4)
		
	for det in detections:
		cv2.rectangle(image, (det.x, det.y), (det.x + det.width, det.y + det.height), (255, 0, 0), 3)
		text = "{}: p={:.2f}".format(det.object_class, det.p)
		cv2.putText(image, text, (int(det.x + delta[0]), int(det.y + delta[1])), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
	return image'''

if __name__ == '__main__':
	cfg.TEST.HAS_RPN = True  # Use RPN for proposals
	mutex = threading.Lock()
	args = parse_args()
	rospy.init_node('simpleDetect')
	bridge = CvBridge()
    # Colored images channels publisher
	pub_singleColor = rospy.Publisher('rcnn/resColor/single', Detection, queue_size = 10)
	pub_arrayColor = rospy.Publisher('rcnn/resColor/array', DetectionArray, queue_size = 2)
	pub_fullColor = rospy.Publisher('rcnn/resColor/full', DetectionFull, queue_size = 2)
	pub_resultedColor = rospy.Publisher('rcnn/resColor/image',Image,queue_size = 5)
	#sub_imageColor = rospy.Subscriber("camera/left/image_raw",Image, imageCallbackColor)

	# Thermal images channels publishers
	pub_singleThermal = rospy.Publisher("rcnn/resThermal/single",Detection,queue_size = 10)
	pub_arrayThermal = rospy.Publisher('rcnn/resThermal/array', DetectionArray, queue_size = 2)
	pub_fullThermal = rospy.Publisher('rcnn/resThermal/full', DetectionFull, queue_size = 2)
	pub_resultedThermal = rospy.Publisher('rcnn/resThermal/image',Image, queue_size = 5)
	sub_imageThermal = rospy.Subscriber("optris/image_raw", Image, imageCallbackThermal)
	#sub_imageThermal = rospy.Subscriber("camera/right/image_raw", Image, imageCallbackThermal)
	#sub_image = rospy.Subscriber("camera/left/image_raw", Image, imageCallback)
	prototxt = os.path.join(os.path.dirname(__file__), args.prototxt)
	caffemodel = os.path.join(os.path.dirname(__file__), args.model)
	classes = parseClasses(os.path.join(os.path.dirname(__file__), args.classes))

	if not os.path.isfile(caffemodel):
		rospy.logerr('%s not found.\nDid you run ./data/script/fetch_faster_rcnn_models.sh?', caffemodel)

	if args.cpu_mode:
		caffe.set_mode_cpu()
	else:
		caffe.set_mode_gpu()
		caffe.set_device(args.gpu_id)
		cfg.GPU_ID = args.gpu_id
	NET = caffe.Net(prototxt, caffemodel, caffe.TEST)

	rospy.loginfo('Loaded network %s', caffemodel)
	rospy.loginfo('Running detection with these classes: %s', str(classes))
	rospy.loginfo('Warmup started')
	'''im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
	timer = Timer()
	timer.tic()
	for i in xrange(2):
		_, _= im_detect(NET, im)
	timer.toc()
	rospy.loginfo('Warmup done in %f seconds. Starting node', timer.total_time)'''
	
	rate = rospy.Rate(100)
	
	#mingxing= CvBridge()
	while not rospy.is_shutdown():
		if (RUNNING):
			mutex.acquire()
			rospy.logwarn('Image2 succeed! the header is %s',IMAGE2.header.seq)
			cv_imageThermal = bridge.imgmsg_to_cv2(IMAGE2)
			mutex.release()
			(scoresThermal, boxesThermal) = detect(cv_imageThermal)
			detectionsThermal = generateDetections(scoresThermal, boxesThermal, classes, args.treshold)
			# Color publishers		
			#Thermal images info publisher
			if (pub_singleThermal.get_num_connections() > 0):
				for msg in detectionsThermal:
					pub_singleThermal.publish(msg)
			if (pub_arrayThermal.get_num_connections() > 0 or pub_fullThermal.get_num_connections() > 0):
				array = DetectionArray()
				array.size = len(detectionsThermal)
				array.data = detectionsThermal
				if (pub_fullThermal.get_num_connections() > 0):
					msg = DetectionFull()
					msg.detections = array
					msg.image =  bridge.cv2_to_imgmsg(getResultImageColor(detectionsThermal, cv_imageThermal),encoding="rgb8")
					pub_resultedThermal.publish(msg.image)
					pub_fullThermal.publish(msg)
				else :
					pub_arrayThermal.publish(array)
				
			RUNNING = False
		else:
			rate.sleep()
			
		
