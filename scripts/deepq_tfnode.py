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

    rospy.init_node('observation_listener', anonymous=True)

    rospy.Subscriber("/distance_cm", String, callback)

    rospy.spin()


class DeepQListener(object):
    """ """

    _tf = TensorFlowUtils()

    def __init__(self):
        """ """
        rospy.init_node('deepq_tfnode')

        rospy.Subscriber('/pi_car/observation',
            String,
            self._observation_callback)

        rospy.Subscriber('/pi_car/start_training', Bool, self._train_callback)

        self.publisher = rospy.Publisher('/pi_car/cmd_resume', Bool)

    def _observation_callback(self, msg):
        """ """
        data = base64.b64decode(msg.data)
        observation = pickle.loads(data)
        self._tf.observations.append(observation)

    def _train_callback(self, msg):
        """ """
        if msg.data:
            self._tf.train_cnn()

    def publisher_resume(self):
        """ """


if __name__ == '__main__':
    listener()

