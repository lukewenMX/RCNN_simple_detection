#!/usr/bin/env python
from __future__ import print_function
 

import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
 
class image_converter:
 
  def __init__(self):
    rospy.init_node('image_converter', anonymous=True)
    self.image_pub = rospy.Publisher("image_topic_2",Image,queue_size=2)
    self.image_pub1 = rospy.Publisher("image_topic_1",Image,queue_size=2)
    #self.image_pub = rospy.Publisher("image_topic_1",Image)
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("camera/left/image_raw",Image,self.callback)
    self.image_sub = rospy.Subscriber('optris/image_raw',Image,self.callback2)
    
  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)
 
    try:
      self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
    except CvBridgeError as e:
      print(e)

  def callback2(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)
 
    try:
      self.image_pub1.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
    except CvBridgeError as e:
      print(e)
  
 
def main(args):
  ic = image_converter()
  
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  
 
if __name__ == '__main__':
    main(sys.argv)