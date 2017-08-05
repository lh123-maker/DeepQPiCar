#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Sam Mohamed

# This is heavily based off https://github.com/DanielSlater/PyGamePlayer/

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
import base64
import pickle
import glob

import random
import rospy
import time

from collections import deque

import cv2
import picamera
import picamera.array
import numpy as np

from std_msgs.msg import String, Bool
import matplotlib.image as mpimg

import utils

from tensorflow_utils import TensorFlowUtils
from kalman_filter import KalmanFilter
from a_star import AStar


def weighted_choice(s): return random.choice(
    sum(([v] * wt for v, wt in s), []))


class DeepQPiCar(object):
    """ """
    ACTIONS_COUNT = 4  # number of valid actions. In this case forward, backward, right, left

    MEMORY_SIZE = 5000  # number of observations to remember
    STATE_FRAMES = 4  # number of frames to store in the state
    OBSERVATION_STEPS = 75  # time steps to observe before training

    CHOICES = [(False, 25), (True, 75)]

    ACTIONS = [(65, 65), (75, 35), (35, 75), (-65, -65)]

    ACTION_CHOICES = [

        ((65, 65), 5),
        ((75, 35), 40),
        ((35, 75), 40),
        ((-65, -65), 15),
    ]

    calc_error_in_position = .01
    calc_error_in_velocity = .0001
    obs_error_in_position = .1
    obs_error_in_velocity = .001

    _tf = TensorFlowUtils()

    _kf = KalmanFilter(
        1,    # delta t is one second
        .8,  # approaximate distance of starting point
        .05,
        .001,
        calc_error_in_position,
        calc_error_in_velocity,
        obs_error_in_position,
        obs_error_in_velocity)

    def __init__(self):
        """ """
        rospy.init_node('deepq_carnode')

        self.rate = rospy.Rate(2)  # send 2 observations per second
        self.camera = picamera.PiCamera()
        self.camera.vflip = True
        self.camera.hflip = True
        self.output = picamera.array.PiRGBArray(self.camera)

        self.last_state = None
        self.crashed = False
        self.dead_counter = 0

        self.observations = deque()

        self.width = 320
        self.height = 240
        self.camera.resolution = (self.width, self.height)

        self.publish_train = False
        self.terminal_frame = False
        self.trainer = rospy.Publisher(
            '/pi_car/start_training', Bool, queue_size=1)

        self.pub = rospy.Publisher('/pi_car/observation',
                                   String,
                                   queue_size=15,
                                   latch=True)

        rospy.Subscriber('/pi_car/cmd_resume', Bool, self.resume)

        self._run = True
        self._wait = False
        self.driver = AStar()

        self.previous_controls = (65, 65)
        self.current_controls = (75, 35)
        self.distance_from_router = 1.0
        self.recovery_count = 0
        self.reward_is_stuck = -.005

    def resume(self, msg):
        """ """
        self._wait = False
        self._run = True

    def _get_normalized_frame(self):
        """ """
        self.camera.capture(self.output, 'rgb')

        frame = np.array(self.output.array)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray = gray[-90:, :]
        frame = (gray - np.min(gray)) / (np.max(gray) - np.min(gray))

        self.output.truncate(0)

        return frame, gray

    def _calculate_reward(self):
        """ """
        observed_distance = round(utils.get_distance_from_router(), 3)

        current_distance = self._kf.get_current_position(
            observed_distance,
            utils.get_velocity_from_motors((65, 65)),
            utils.get_acceleration_from_motors((65, 65), (75, 35))
        )

        current_distance = round(current_distance, 3)
        self.reward = round(current_distance - self.distance_from_router, 3)
        self.distance_from_router = float(current_distance)

        print('my reward', self.reward, 'observed distance',
              observed_distance, 'current distance', current_distance)

        if current_distance > 1.:
            print('.............', current_distance)
            self.reward_is_stuck = .0
        else:
            self.reward_is_stuck = -.005

        if self.reward < self.reward_is_stuck:
            self.dead_counter += 1
            if self.dead_counter >= 3:
                print('I am stuck!')
                self.terminal_frame = True
                self.dead_counter = 0
        else:
            self.terminal_frame = False
            self.dead_counter = 0

        if self.reward > 0.:
            self.reward *= 1.5

        else:
            self.reward *= -.1

        

    def _set_state(self):
        """ """
        # first frame must be handled differently
        if self.last_state is None:
            self.last_action = np.zeros(self.ACTIONS_COUNT)
            self.last_action[0] = 1  # move forward
            self.last_state = np.stack(tuple(self._get_normalized_frame()[
                                       0] for _ in range(self.STATE_FRAMES)), axis=2)

        frame, img = self._get_normalized_frame()
        frame = np.reshape(frame, (90, self.width, 1))

        current_state = np.append(self.last_state[:, :, 1:], frame, axis=2)
        return current_state, img

    def _set_action(self):
        if self.terminal_frame:
            print('attempting recovery')
            self._set_recovery_action()
            if self.recovery_count > 3:
                print('done with recovery')
                self.recovery_count = 0
                self.terminal_frame = False

            if self.recovery_count:
                return

        self.change = weighted_choice(self.CHOICES)
        if self.change:
            self.last_action = np.zeros(self.ACTIONS_COUNT)
            self.current_controls = weighted_choice(self.ACTION_CHOICES)
            index = self.ACTIONS.index(self.current_controls)
            self.last_action[index] = 1

    def _set_recovery_action(self):
        self.recovery_count += 1
        self.last_action = np.zeros(self.ACTIONS_COUNT)
        self.current_controls = (-65, -65)
        index = self.ACTIONS.index(self.current_controls)
        self.last_action[index] = 1

    def _publish_observation(self):
        """ """
        current_state, img = self._set_state()
        self._set_action()
        self._calculate_reward()
        

        observation = (
            self.last_state,
            self.last_action,
            self.reward,
            current_state,
            img,
            self.terminal_frame
        )

        self.observations.append(observation)
        self.last_state = current_state

        if len(self.observations) > self.MEMORY_SIZE:
            self.observations.popleft()

        self._publish(observation)

        if len(self.observations) > 0 and len(self.observations) % self.OBSERVATION_STEPS == 0:
            # stop the car while training
            self.move(0, 0)
            self.publish_train = True
            self._wait = True
            while self._wait:
                self._train()
                time.sleep(5)
                self.publish_train = False

    def _publish(self, obs):
        """ """
        msg = String()
        data = pickle.dumps(obs)
        msg.data = base64.b64encode(data)

        self.pub.publish(msg)
        self.rate.sleep()

    def _train(self):
        msg = Bool()
        msg.data = True

        if self.publish_train:
            self.trainer.publish(msg)

    def _move(self, action):
        self.move(*action)

    def move(self, left, right):
        try:
            self.driver.motors(left, right)
        except IOError as ioerror:
            print("io error")
        except Exception as e:
            print(e)

    def run(self):
        """ """
        while self._run:
            if not self._wait:
                self._publish_observation()
                self._move(self.current_controls)
                time.sleep(1.0)


if __name__ == '__main__':
    node = DeepQPiCar()
    node.run()
