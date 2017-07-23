#!/usr/bin/env python
import sys
import io

import cv2
import numpy as np

import roslib
import rospy

from sensor_msgs.msg import CompressedImage

class image_feature:

    def __init__(self):
        self.image_pub = rospy.Publisher("/output/image_raw/compressed",
            CompressedImage, queue_size=5)
        image_np = cv2.imread('/home/pi/image.jpg')
        output = io.BytesIO()
        np.savez(output, x=image_np)
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        #msg.data = np.array(cv2.imencode('.jpg', image_np)[1]).tostring()
        msg.data = output.getvalue()
        # Publish new image
        while not rospy.is_shutdown():
            self.image_pub.publish(msg)

def main(args):
    '''Initializes and cleanup ros node'''
    rospy.init_node('image_feature', anonymous=True)
    ic = image_feature()
    # rospy.init_node('image_feature', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down ROS Image feature detector module"

if __name__ == '__main__':
    main(sys.argv)

