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
import base64
import pickle

import random
import rospy
import time

import picamera
import picamera.array
import numpy as np

from std_msgs.msg import String, Bool
import matplotlib.image as mpimg

import tf_utils

class DeepQPiCar(object):
    """ """
    LEARN_RATE = 1e-4
    ACTIONS_COUNT = 4  # number of valid actions. In this case forward, backward, right, left
    MEMORY_SIZE = 500000  # number of observations to remember
    STATE_FRAMES = 4  # number of frames to store in the state
    OBSERVATION_STEPS = 50000.  # time steps to observe before training
    MINI_BATCH_SIZE = 100  # size of mini batches
    STORE_SCORES_LEN = 200.
    SAVE_EVERY_X_STEPS = 1000
    EXPLORE_STEPS = 500000.  # frames over which to anneal epsilon
    FUTURE_REWARD_DISCOUNT = 0.99  # decay rate of past observations
    FINAL_RANDOM_ACTION_PROB = 0.05  # final chance of an action being random
    INITIAL_RANDOM_ACTION_PROB = 1.0  # starting chance of an action being random
    OBS_LAST_STATE_INDEX, OBS_ACTION_INDEX, OBS_REWARD_INDEX, OBS_CURRENT_STATE_INDEX = range(4)

    rate = rospy.Rate(10) # 10hz
    camera = picamera.PiCamera()
    output = picamera.array.PiRGBArray(camera)

    def __init__(self):
        """ """
        rospy.init_node('deepq_pi_car')

        self._last_state = None
        self._crashed = False
        self._distance_from_router = 0.

        self.camera.resolution = (640, 480)
        self.camera.color_effects = (128,128) # turn camera to black and white
        picamera.array.PiRGBArray(camera)

        self.trainer = rospy.Publisher('/pi_car/start_training', Bool)

        self.pub = rospy.Publisher('/pi_car/observation',
            String,
            queue_size=15,
            latch=True)

        rospy.Subscriber('/pi_car/cmd_resume', Bool, self.resume)

    def resume(self, msg):
        """ """
        self._wait = False
        self._run = True
        self.run()

    def _get_normalized_frame(self):
        """ """
        self.camera.capture(self.output, 'rgb')

        frame = np.array(self.output.array)
        frame = (frame-np.min(frame))/(np.max(frame)-np.min(frame))

        self.output.truncate(0)

        return frame

    def _calculate_reward(self):
        """ """
        current_distance = tf_utils.calc_distance_from_router()
        dx_distance = abs(self._distance_from_router - current_distance)
        self._distance_from_router = current_distance

        if self._crashed:
            self._crashed = False
            self._count_since_crash = 1
            return dx_distance

        return .5*self._count_since_crash*dx_distance + dx_distance

    def _set_state_and_action(self):
        """ """
        # first frame must be handled differently
        if self._last_state is None:
            self._last_action = np.zeros(self.ACTIONS_COUNT)
            self._last_action[0] = 1 # move forward
            current_state = np.stack(tuple(self._get_normalized_frame() for _ in range(self.STATE_FRAMES)), axis=2)

        else:
            self._last_action = np.zeros(self.ACTIONS_COUNT)
            self._last_action[random.randint(0,3)] = 1 # move forward
            current_state = np.append(self._last_state[:, :, 1:], self._get_normalized_frame(), axis=2)

    def _set_observation(self):
        """ """
        self._set_state_and_action()

        observation = (
            self._last_state,
            self._last_action,
            self._calculate_reward(),
            current_state)

        self._observations.append(observation)
        self._last_state = current_state

        if len(self._observations) > self.MEMORY_SIZE:
            self._observations.popleft()

        # only train if done observing
        if len(self._observations) > self.OBSERVATION_STEPS:
            self._run = False
            self._time += 1
            return None

        # update the old values
        self._last_state = current_state
        return observation

    def _publish(self, obs):
        """ """
        data = pickle.dumps(obs)
        msg.data = base64.b64encode(data)

        self.pub.publish(msg)
        rate.sleep()

    def _train(self):
        msg = Bool()
        msg.data = True

        if self._publish_train:
            self.trainer.publish(msg)

    def run(self):
        """ """
        while self._run:
            msg = String()
            obs = self._set_observation()

            if obs:
                self._publish()
            else:
                while self._wait:
                    self._train()
                    time.sleep(5)
                    self._publish_train = False


if __name__ == '__main__':
    main()


