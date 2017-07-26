#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Sam Mohamed

# This is heavily based off https://github.com/DanielSlater/PyGamePlayer/

"""
import os
import math
import random

from collections import deque

import numpy as np
import tensorflow as tf

import utils

class TensorFlowUtils(object):

    
    img_width = 640
    img_height = 150
    state_frames = 4
    actions_count = 4

    observations = deque()

    mini_batch_size = 100

    LEARN_RATE = 1e-4
    SAVE_EVERY_X_STEPS = 100
    FUTURE_REWARD_DISCOUNT = 0.1 # decay rate of past observations
    OBS_LAST_STATE_INDEX, OBS_ACTION_INDEX, OBS_REWARD_INDEX, OBS_CURRENT_STATE_INDEX, OBS_CRASH_INDEX = range(5)

    def __init__(self):
        """ """
        self.time = 0
        self.X = tf.placeholder(tf.float32)
        self.Y_pred, self.poly_session = self._create_poly_graph()

        self._create_convolutional_network()

    def calc_distance_from_router(self):
        """ """
        signal_level = utils.get_signal_level_from_router('wlan0')
        return self.Y_pred.eval(feed_dict={ self.X : signal_level}, session=self.poly_session)[0]

    def _create_poly_graph(self):
        """ """
        model = 'models/25k-poly-model-1e-3/25k-poly-model-1e-3'

        tf_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

        Y_pred = tf.Variable(tf.random_normal([1]), name='bias')

        for pow_i in range(1, 3):
            W = tf.Variable(tf.random_normal([1]), name='weight_%d' % pow_i)
            Y_pred = tf.add(tf.multiply(tf.pow(self.X, pow_i), W), Y_pred)

        saver = tf.train.Saver()

        sess = tf.Session()

        saver.restore(sess, "{}/{}".format(tf_dir, model))

        return Y_pred, sess


    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def _create_convolutional_network(self):
        """ """

        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        # convolution_weights_1 = tf.Variable(tf.truncated_normal([8, 8, self.state_frames, 32], stddev=0.01))
        # convolution_bias_1 = tf.Variable(tf.constant(0.01, shape=[32]))
        
        

        input_layer = tf.placeholder("float", [None, self.img_height, self.img_width, self.state_frames])


        convolution_weights_1  = self.weight_variable([12, 12, self.state_frames, 32])
        convolution_bias_1     = self.bias_variable([32])

        convolution_weights_2  = self.weight_variable([12, 12, 32, 64])
        convolution_bias_2     = self.bias_variable([64])

        convolution_weights_3  = self.weight_variable([9, 9, 64, 64])
        convolution_bias_3     = self.bias_variable([64])

        convolution_weights_4  = self.weight_variable([9, 9, 64, 64])
        convolution_bias_4     = self.bias_variable([64])

        convolution_weights_5  = self.weight_variable([7, 7, 64, 64])
        convolution_bias_5     = self.bias_variable([64])

        convolution_weights_6  = self.weight_variable([7, 7, 64, 64])
        convolution_bias_6     = self.bias_variable([64])

        convolution_weights_7  = self.weight_variable([5, 5, 64, 64])
        convolution_bias_7     = self.bias_variable([64])

        feed_forward_weights_1 = self.weight_variable([640, 640])
        feed_forward_bias_1    = self.bias_variable([640])

        feed_forward_weights_2 = self.weight_variable([640, self.actions_count])
        feed_forward_bias_2    = self.bias_variable([self.actions_count])

        hidden_convolutional_layer_1 = tf.nn.relu(self.conv2d(input_layer, convolution_weights_1) + convolution_bias_1)
        hidden_max_pooling_layer_1   = self.max_pool_2x2(hidden_convolutional_layer_1)

        hidden_convolutional_layer_2 = tf.nn.relu(self.conv2d(hidden_max_pooling_layer_1, convolution_weights_2) + convolution_bias_2)
        hidden_max_pooling_layer_2   = self.max_pool_2x2(hidden_convolutional_layer_2)

        hidden_convolutional_layer_3 = tf.nn.relu(self.conv2d(hidden_max_pooling_layer_2, convolution_weights_3) + convolution_bias_3)
        hidden_max_pooling_layer_3   = self.max_pool_2x2(hidden_convolutional_layer_3)

        hidden_convolutional_layer_4 = tf.nn.relu(self.conv2d(hidden_max_pooling_layer_3, convolution_weights_4) + convolution_bias_4)
        hidden_max_pooling_layer_4   = self.max_pool_2x2(hidden_convolutional_layer_4)

        hidden_convolutional_layer_5 = tf.nn.relu(self.conv2d(hidden_max_pooling_layer_4, convolution_weights_5) + convolution_bias_5)
        hidden_max_pooling_layer_5   = self.max_pool_2x2(hidden_convolutional_layer_5)

        hidden_convolutional_layer_6 = tf.nn.relu(self.conv2d(hidden_max_pooling_layer_5, convolution_weights_6) + convolution_bias_6)
        hidden_max_pooling_layer_6   = self.max_pool_2x2(hidden_convolutional_layer_6)

        hidden_convolutional_layer_7 = tf.nn.relu(self.conv2d(hidden_max_pooling_layer_6, convolution_weights_7) + convolution_bias_7)
        hidden_max_pooling_layer_7   = self.max_pool_2x2(hidden_convolutional_layer_7)

        hidden_convolutional_layer_3_flat = tf.reshape(hidden_max_pooling_layer_7, [-1, 640])

        final_hidden_activations = tf.nn.relu(
            tf.matmul(hidden_convolutional_layer_3_flat, feed_forward_weights_1) + feed_forward_bias_1)

        output_layer = tf.matmul(final_hidden_activations, feed_forward_weights_2) + feed_forward_bias_2

        self.input_layer = input_layer
        self.output_layer = output_layer

        self.action = tf.placeholder("float", [None, self.actions_count])
        self.target = tf.placeholder("float", [None])

        readout_action = tf.reduce_sum(tf.multiply(self.output_layer, self.action), reduction_indices=1)

        self.cost = tf.reduce_mean(tf.square(self.target - readout_action))
        self.train_operation = tf.train.AdamOptimizer(self.LEARN_RATE).minimize(self.cost, global_step=self.global_step)

        model = 'models/convnet/3layers-1e-4'
        tf_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        self.save_path =  "{}/{}".format(tf_dir, model)

        self.cnn_session = tf.Session()
        self.cnn_saver = tf.train.Saver()

        if os.path.exists('{}/checkpoint'.format(self.save_path)):
            chkpoint_state = tf.train.get_checkpoint_state(self.save_path)
            self.cnn_saver.restore(self.cnn_session, chkpoint_state.model_checkpoint_path)
        else:
            self.cnn_session.run(tf.global_variables_initializer())


    def train_cnn(self):
        """ """
        if os.path.exists('{}/checkpoint'.format(self.save_path)):
            chkpoint_state = tf.train.get_checkpoint_state(self.save_path)
            self.cnn_saver.restore(self.cnn_session, chkpoint_state.model_checkpoint_path)

        count = len(self.observations)
        epochs = int((count * math.log(count))/count)

        print("number of observations is {}.  Training for {} epochs".format(count, epochs))
        for i in range(epochs):
            # sample a mini_batch to train on
            mini_batch = random.sample(self.observations, self.mini_batch_size-1)
            # get the batch variables
            previous_states = [d[self.OBS_LAST_STATE_INDEX] for d in mini_batch]
            actions = [d[self.OBS_ACTION_INDEX] for d in mini_batch]
            rewards = [d[self.OBS_REWARD_INDEX] for d in mini_batch]
            current_states = [d[self.OBS_CURRENT_STATE_INDEX] for d in mini_batch]
            agents_expected_reward = []
            # this gives us the agents expected reward for each action we might
            agents_reward_per_action = self.cnn_session.run(self.output_layer, feed_dict={self.input_layer: current_states})
            for i in range(len(mini_batch)):
                self.time += 1
                if mini_batch[i][self.OBS_CRASH_INDEX]:
                    # this was a crash frame so there is no future reward...
                    agents_expected_reward.append(rewards[i])
                else:
                    agents_expected_reward.append(
                        rewards[i] + self.FUTURE_REWARD_DISCOUNT * np.max(agents_reward_per_action[i]))

            # learn that these actions in these states lead to this reward
            self.cnn_session.run(self.train_operation, feed_dict={
                self.input_layer: previous_states,
                self.action: actions,
                self.target: agents_expected_reward})

        # save checkpoints for later
        global_step = tf.train.global_step(self.cnn_session, self.global_step)
        self.cnn_saver.save(self.cnn_session, self.save_path+'/my_model', global_step=global_step)
        print('one more training finished.  global_step: %s' % global_step)