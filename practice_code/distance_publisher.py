#!/usr/bin/env python
import atexit
import rospy
from std_msgs.msg import String

import utils

class DistancePublisher:

    def __init__(self):
        self.distanc_pub = rospy.Publisher("/distance_cm", String, queue_size=15)
        
        rate = rospy.Rate(10) # 10hz

        while not rospy.is_shutdown():

            signal_level = utils.get_signal_level_from_router('wlan0')
            distance_cm = utils.get_distance_from_router(signal_level)

            message = "{'%s' : {'signal_level' : %s,'distance_cm': %s } }" % (rospy.get_time(),
                signal_level,
                distance_cm)

            self.distanc_pub.publish(message)
            rate.sleep()


def main():
    '''Initializes and cleanup ros node'''
    rospy.init_node('distance_publisher', anonymous=True)
    cm_pub = DistancePublisher()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down ROS Image feature detector module"

if __name__ == '__main__':
    main()
