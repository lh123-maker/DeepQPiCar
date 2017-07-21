#!/usr/bin/env python
import ast
import atexit
import pickle
import datetime

import rospy

from std_msgs.msg import String

global_data = dict()

def callback(data):
    global global_data
    d = ast.literal_eval(data.data)
    global_data.update(d)
 
def save_to_file():
    global global_data
    
    print 'saving data to pickle'

    with open('/home/sameh/distance_data/{}.p'.format(datetime.datetime.now()), 'wb') as _f:
        pickle.dump(global_data, _f, protocol=pickle.HIGHEST_PROTOCOL)


def listener():
    atexit.register(save_to_file)

    rospy.init_node('distance_listener', anonymous=True)

    rospy.Subscriber("/distance_cm", String, callback)

    rospy.spin()

if __name__ == '__main__':
    listener()

