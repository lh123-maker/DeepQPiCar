#!/usr/bin/env python
import functools

import atexit
import rospy
import random

from geometry_msgs.msg import Twist

def delayed_callback(msg, event):
    pub.publish(msg)

rospy.init_node('distance_publisher', anonymous=True)
pub = rospy.Publisher("/cmd_vel", Twist, queue_size=5)
        
while not rospy.is_shutdown():

    msg = Twist()
            
    msg.linear.x = random.randint(-7, 7)
    msg.linear.y = random.randint(0, 7)
    msg.linear.z = random.randint(0, 7)

    msg.angular.x = random.randint(-7, 7)
    msg.angular.y = random.randint(0, 7)
    msg.angular.z = random.randint(0, 7)

    import time
    time.sleep(2)
    pub.publish(msg)
    

rospy.spin()
