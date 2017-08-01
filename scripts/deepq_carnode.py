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
    ACTIONS_COUNT = 20  # number of valid actions. In this case forward, backward, right, left
    
    MEMORY_SIZE = 5000  # number of observations to remember
    STATE_FRAMES = 4  # number of frames to store in the state
    OBSERVATION_STEPS = 100  # time steps to observe before training

    CHOICES = [(False, 50), (True, 50)]

    # ACTION_CHOICES = [
        
    #     ((65, 65), 8),
    #     ((65, 65), 8),
    #     ((65, 65), 8),
    #     ((65, 65), 8),
    #     ((-65, -65), 2),
        

    #     ((75, 65), 8),
    #     ((75, 50), 8),
    #     ((75, 35), 8),
    #     ((75, 50), 2),
    #     ((-75, -35), 2),

    #     ((65, 75), 8),
    #     ((50, 75), 8),
    #     ((35, 75), 8),
    #     ((65, 75), 2),
    #     ((-50, -75), 2),

    #     ((-65, -65), 2),
    #     ((-65, -65), 2),
    #     ((-65, -65), 2),
    #     ((-75, -65), 2),    
    #     ((-35, -75), 2),

    #     ]

    ACTION_CHOICES = [
        
        ((65, 65), 8),
        ((65, 65), 8),
        ((65, 65), 8),
        ((65, 65), 8),
        ((65, 65), 2),
        

        ((75, 65), 8),
        ((75, 50), 8),
        ((75, 35), 8),
        ((75, 50), 2),
        ((75, 35), 2),

        ((65, 75), 8),
        ((50, 75), 8),
        ((35, 75), 8),
        ((65, 75), 2),
        ((50, 75), 2),

        ((65, 65), 2),
        ((65, 65), 2),
        ((65, 65), 2),
        ((75, 65), 2),    
        ((35, 75), 2),

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
        self.dead_counter = 0
        to_average = []
        for i in range(3):
            to_average.append(self._tf.calc_distance_from_router())

        self.distance_from_router = sum(to_average)/3.
        print('my distance', self.distance_from_router)
        
        self.observations = deque()

        self.width = 320
        self.height = 240
        self.camera.resolution = (self.width, self.height)

        self.publish_train = False
        self.terminal_frame = False
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

    def _calculate_reward(self):
        """ """
        to_average = []
        for i in range(3):
            to_average.append(self._tf.calc_distance_from_router())

        current_distance = sum(to_average)/3.
        self.reward = current_distance - self.distance_from_router
        self.distance_from_router = current_distance
        print('my reward', self.reward, 'and my distance', current_distance)
        if self.reward == 0.:
            self.dead_counter += 1
            if self.dead_counter >= 2:
                print('I am stuck!')
                self.terminal_frame = True
        else:
            self.terminal_frame = False
            self.dead_counter = 0

        if self.reward > .1:
            self.reward *= 1.5

        elif self.reward < -.1:
            self.reward *= -.75

        

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
                try:
                    index = self.ACTION_CHOICES.index((self.move_args, 8))
                except ValueError as e:
                    index = self.ACTION_CHOICES.index((self.move_args, 2))

                self.last_action[index] = 1

        frame, img = self._get_normalized_frame()
        frame = np.reshape(frame, (90, self.width, 1))
        
        current_state = np.append(self.last_state[:, :, 1:], frame, axis=2)
        return current_state, img

    def _publish_observation(self):
        """ """
        self._calculate_reward()

        current_state, img = self._set_state_and_action()
        observation = (
            self.last_state,
            self.last_action,
            self.reward,
            current_state, 
            img,
            self.terminal_frame)

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
                self._publish_observation()
                self._move(np.argmax(self.last_action))
                time.sleep(1.5)


if __name__ == '__main__':
    node = DeepQPiCar()
    node.run()