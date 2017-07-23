#!/usr/bin/env python3
import os
import pickle
import math
import numpy as np

import roslib;roslib.load_manifest('xiaocar')
import rospy

from geometry_msgs.msg import Twist

from a_star import AStar

astar = AStar()

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

model = pickle.load(open(os.path.join(__location__, 'model.p'),'rb'))

def callback(msg):
    global model
    global astar

    rospy.loginfo("Received a /cmd_vel message!")
    rospy.loginfo("Linear Components: [%f, %f, %f]"%(msg.linear.x, msg.linear.y, msg.linear.z))
    rospy.loginfo("Angular Components: [%f, %f, %f]"%(msg.angular.x, msg.angular.y, msg.angular.z))

    # Do velocity processing here:
    # Use the kinematics of your robot to map linear and angular velocities into motor commands
    
    v = math.sqrt((float(msg.linear.x)**2 + float(msg.linear.y)**2 + float(msg.linear.z)**2))
    r = math.sqrt((float(msg.angular.x)**2 + float(msg.angular.y)**2+ float(msg.angular.z)**2))

    v = -v if float(msg.linear.x) < 0 else v
    r = -r if float(msg.angular.x) < 0 else r
    
    w = v/r

    y = np.array([v,w]).reshape(1, -1)
    x = model.predict(y)
    rospy.logwarn(' %s predicted %s' % (y,x))

    v_l = min(int(x[0][0]), 250)
    v_r = min(int(x[0][1]), 250)

    # Then set your wheel speeds (using wheel_left and wheel_right as examples)
    astar.motors(v_l, v_r)        

def listener():
    rospy.init_node('motor_controller')
    rospy.Subscriber("/cmd_vel", Twist, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()

