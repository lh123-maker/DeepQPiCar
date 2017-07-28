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

from tensorflow_utils import TensorFlowUtils
from a_star import AStar

import random
weighted_choice = lambda s : random.choice(sum(([v]*wt for v,wt in s),[]))

class DeepQPiCar(object):
    """ """
    ACTIONS_COUNT = 10  # number of valid actions. In this case forward, backward, right, left
    NO_DX_COUNT = 4 # number of times there's no change in distance before assuming crash
    
    MEMORY_SIZE = 50000  # number of observations to remember
    STATE_FRAMES = 4  # number of frames to store in the state
    OBSERVATION_STEPS = 3  # time steps to observe before training
    NO_DX_MEASUREMENT = 0.05 # dx in distance that's not considered forward motion

    CHOICES = [(False, 80), (True, 20)]

    ACTION_CHOICES = [
        
        ((50, 50), 10),
        ((55, 55), 10),
        ((60, 60), 10),
        ((65, 65), 10),

        ((80, 40), 10),
        ((40, 80), 10),
        
        ((90, 45), 10),
        ((45, 90), 10),
        
        ((100,50), 10),
        ((50,100), 10),
        ]

    RECOVER_CHOICES = [
        ((-100,-100), 10),
        ((-50,-50), 10),
        ((-50,-75), 10),
        ((-75,-50), 10),
        ((-75,-100), 10),
        ((-100,-75), 10),
        ((-40,-60), 10),
        ((-60,-40), 10),
        ((-55,-65), 10),
        ((-65,-55), 10)
        ]

    _tf = TensorFlowUtils()

    def __init__(self):
        """ """
        rospy.init_node('deepq_carnode')

        self.move_args = (0, 0)
        self.rate = rospy.Rate(2) # send 2 observations per second
        self.camera = picamera.PiCamera()
        self.camera.vflip = True
        self.camera.hflip = True
        self.output = picamera.array.PiRGBArray(self.camera)

        self.last_state = None
        self.crashed = False
        self.nodx_counter = 0
        self.distance_from_router = self._tf.calc_distance_from_router()
        
        self.observations = deque()

        self.width = 320
        self.height = 240
        self.camera.resolution = (self.width, self.height)

        self.publish_train = False
        self.trainer = rospy.Publisher('/pi_car/start_training', Bool, queue_size=1)

        self.pub = rospy.Publisher('/pi_car/observation',
            String,
            queue_size=15,
            latch=True)

        rospy.Subscriber('/pi_car/cmd_resume', Bool, self.resume)

        self._run = True
        self._wait = False
        self.count_since_crash = 1
        self.driver = AStar()

    def resume(self, msg):
        """ """
        self._wait = False
        self._run = True

    def _get_normalized_frame(self):
        """ """
        self.camera.capture(self.output, 'rgb')

        frame = np.array(self.output.array)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray = gray[-90:,:]
        frame = (gray-np.min(gray))/(np.max(gray)-np.min(gray))

        self.output.truncate(0)

        return frame, gray

    def _set_dx_distance(self):
        """ """
        current_distance = self._tf.calc_distance_from_router()
        self.dx_distance = self.distance_from_router - current_distance
        self.distance_from_router = current_distance

    def _calculate_reward(self):
        """ """
        if self.crashed:
            self.crashed = False
            self.count_since_crash = 1
            self.reward = self.dx_distance
            return

        self.reward = .5*self.count_since_crash*self.dx_distance + self.dx_distance

    def _set_state_and_action(self):
        """ """
        # first frame must be handled differently
        if self.last_state is None:
            self.last_action = np.zeros(self.ACTIONS_COUNT)
            self.last_action[0] = 1 # move forward
            self.last_state = np.stack(tuple(self._get_normalized_frame()[0] for _ in range(self.STATE_FRAMES)), axis=2)

        else:
            self.change = weighted_choice(self.CHOICES)
            if self.change:
                self.last_action = np.zeros(self.ACTIONS_COUNT)
                self.move_args = weighted_choice(self.ACTION_CHOICES)
                index = self.ACTION_CHOICES.index((self.move_args, 10))
                self.last_action[index] = 1

        frame, img = self._get_normalized_frame()
        frame = np.reshape(frame, (90, self.width, 1))
        
        current_state = np.append(self.last_state[:, :, 1:], frame, axis=2)
        return current_state, img

    def _set_terminal_frame(self):
        """ """
        self.terminal_frame = False
        if self.dx_distance <= self.NO_DX_MEASUREMENT:
            self.nodx_counter += 1
            if self.nodx_counter == self.NO_DX_COUNT:
                self.nodx_counter = 0
                self.terminal_frame = True

    def _correct_observation(self, observation):
        """ """
        # only train if done observing
        if len(self.observations) % self.OBSERVATION_STEPS == 0:
            # self._run = False
            return None

        if self.terminal_frame:
            self._implement_recovery()
            self.last_state = None
            return observation

        return observation

    def _implement_recovery(self):
        """ """
        self.move(0,0)
        time.sleep(3)
        self.move(0,0)
        
        for i in range(2):
            args = weighted_choice(self.RECOVER_CHOICES)
            self.move(*args)
            time.sleep(1)
        self.move(0, 0)
        time.sleep(1.5)


    def _set_observation(self):
        """ """
        self._set_dx_distance()

        current_state, img = self._set_state_and_action()
        
        self._set_terminal_frame()
        
        self._calculate_reward()

        observation = (
            self.last_state,
            self.last_action,
            self.reward,
            current_state, 
            self.terminal_frame, img)

        self.observations.append(observation)
        self.last_state = current_state

        if len(self.observations) > self.MEMORY_SIZE:
            self.observations.popleft()

        return self._correct_observation(observation)

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
        self.move(*self.move_args)

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
                obs = self._set_observation()

                if obs:
                    self._move(np.argmax(self.last_action))
                    self._publish(obs)
                    time.sleep(.5)
                else:
                    self.move(0, 0)
                    self.publish_train = True
                    self._wait = True
                    while self._wait:
                        self._train()
                        time.sleep(5)
                        self.publish_train = False

if __name__ == '__main__':
    node = DeepQPiCar()
    node.run()