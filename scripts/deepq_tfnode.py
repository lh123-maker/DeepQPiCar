#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Sam Mohamed

The DeepQPi class is a ROS node that's both a publisher and listener, which together
control the motion of the car in the environment and apply the policy during training.

Motion in a random action using an epsilon-greedy implementation of random direction.

Rewards are calculated based on the differential in the distance moved from one action
to the next.  Therefore continous motion is more highly rewarded.

The state is represented in four images collected from the front camera.  Actions are
one movement of forward, right, left or reverse.  State to action mappings are recorded
as observations with the reward and published using the ROS system.

After a number of published observations, the module publishes to start ConvNet training
on a mini batch of the total observations and stops motion and data collection.

The subscriber waits for messages that the training is complete and resumes motion and
and observation publishing.

Usage:
    usage::
        $ roslaunch deepqpicar training.launch
Attributes:
    HEADTER (tuple): First element in tuple holds the Numpy dtype object
        representing the header of the transaction file.
        Second element in tuple holds the big-endian length in bytes.
    CC_DB (tuple): First element in tuple holds the Numpy dtype object
        representing credit and debit card transactions.
        Second element in tuple holds the big-endian length in bytes.
    AUTO_PAY (tuple): First element in tuple holds the Numpy dtype object
        representing autopay transactions.
        Second element in tuple holds the big-endian length in bytes.
    RECORDS (deque): List of Record objects each representing a parsed record.

"""
import os
import time
import pickle
import glob
import base64

import rospy

from std_msgs.msg import String, Bool

from tensorflow_utils import TensorFlowUtils

class DeepQTFNode(object):
    """ """

    _tf = TensorFlowUtils()

    def __init__(self):
        """ """
        rospy.init_node('deepq_tfnode')

        rospy.Subscriber('/pi_car/observation',
            String,
            self._observation_callback)

        rospy.Subscriber('/pi_car/start_training', Bool, self._train_callback)

        self.publisher = rospy.Publisher('/pi_car/cmd_resume', Bool, queue_size=1)

        directory = 'pickles'
        tf_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        self.save_path =  "{}/{}".format(tf_dir, directory)

    def _observation_callback(self, msg):
        """ """
        data = base64.b64decode(msg.data)
        observation = pickle.loads(data)
        self._tf.observations.append(observation)

        if len(self._tf.observations) % 25 == 0:
            
            filename = str(round(time.time() * 1000))
            with open('{}/{}.p'.format(self.save_path, filename), 'wb') as f:
                pickle.dump(self._tf.observations, f)
                print("wrote {} to file".format(len(self._tf.observations)))

            files = glob.glob('{}/{}/*.p'.format(self.save_path, filename))
            for name in files:
                if not filename in name:
                    os.remove(name)


    def _train_callback(self, msg):
        """ """
        if msg.data:
            # print("beginning training.  have collected {} observations".format(len(self._tf.observations)))
            self._tf.train_cnn()
            self.publish_resume()

    def publish_resume(self):
        """ """
        msg = Bool()
        msg.data = True
        self.publisher.publish(msg)

    def run(self):
        # imitate a thread
        while True:
            time.sleep(1)

if __name__ == '__main__':
    node = DeepQTFNode()
    node.run()
