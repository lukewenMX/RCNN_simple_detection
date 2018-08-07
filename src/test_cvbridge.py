#! usr/bin/env python
import rospy 
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import thread
import sys

IMAGE1 = Image()
IMAGE2 = Image()

def callbackThermal(im):
    global IMAGE1
    IMAGE1 = im

def callbackColor(im):
    global IMAGE2
    IMAGE2 = im

if __name__ ="__main__":
    rospy.init_node('test')
    sub_color = rospy.Subscriber('camera/left/image_raw',Image,callbackColor)
    sub_thermal = rospy.Subscriber('optris/image_raw',Image,callbackThermal)

    pub_color = rospy.Publisher('color_image',Image,queue_size = 2)
    pub_thermal = rospy.Publisher('thermal_image',Image,queue_size = 2 )

    bridge = CvBridge()
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        color_image = bridge.imgmsg_to_cv2(IMAGE1)
        thermal_image = bridge.imgmsg_to_cv2(IMAGE2)

        color_image_pub = bridge.cv2_to_imgmsg(color_image)
        thermal_image_pub = bridge.cv2_to_imgmsg(thermal_image)
        pub_color.publish(color_image_pub)
        pub_thermal.publish(thermal_image_pub)

        rate.sleep()

