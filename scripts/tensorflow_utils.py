#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Sam Mohamed

# This is heavily based off https://github.com/DanielSlater/PyGamePlayer/

"""
import os
from collections import deque

import tensorflow as tf

import utils

class TensorFlowUtils(object):

    
    img_width = 120
    img_height = 120
    state_frames = 4
    actions_count = 4

    observations = deque()

    mini_batch_size = 100

    FUTURE_REWARD_DISCOUNT = 0.1 # decay rate of past observations
    OBSlast_state_INDEX, OBS_ACTION_INDEX, OBS_REWARD_INDEX, OBS_CURRENT_STATE_INDEX, OBS_CRASH_INDEX = range(5)

    def __init__(self):
        """ """
        self.X = tf.placeholder(tf.float32)
        self.Y_pred, self.poly_session = self._create_poly_graph()

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

    def _create_convolutional_network(self):
        """ """
        convolution_weights_1 = tf.Variable(tf.truncated_normal([8, 8, self.state_frames, 32], stddev=0.01))

        convolution_bias_1 = tf.Variable(tf.constant(0.01, shape=[32]))
        convolution_weights_2 = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev=0.01))
        convolution_bias_2 = tf.Variable(tf.constant(0.01, shape=[64]))

        convolution_weights_3 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.01))
        convolution_bias_3 = tf.Variable(tf.constant(0.01, shape=[64]))

        feed_forward_weights_1 = tf.Variable(tf.truncated_normal([256, 256], stddev=0.01))
        feed_forward_bias_1 = tf.Variable(tf.constant(0.01, shape=[256]))

        feed_forward_weights_2 = tf.Variable(tf.truncated_normal([256, self.actions_count], stddev=0.01))
        feed_forward_bias_2 = tf.Variable(tf.constant(0.01, shape=[self.actions_count]))

        self.input_layer = tf.placeholder("float", [None, self.img_width, self.img_height, self.state_frames])

        hidden_convolutional_layer_1 = tf.nn.relu(
            tf.nn.conv2d(input_layer, convolution_weights_1, strides=[1, 4, 4, 1], padding="SAME") + convolution_bias_1)

        hidden_max_pooling_layer_1 = tf.nn.max_pool(hidden_convolutional_layer_1, ksize=[1, 2, 2, 1],
                                        strides=[1, 2, 2, 1], padding="SAME")

        hidden_convolutional_layer_2 = tf.nn.relu(
            tf.nn.conv2d(hidden_max_pooling_layer_1, convolution_weights_2, strides=[1, 2, 2, 1],
                    padding="SAME") + convolution_bias_2)

        hidden_max_pooling_layer_2 = tf.nn.max_pool(hidden_convolutional_layer_2, ksize=[1, 2, 2, 1],
                                                strides=[1, 2, 2, 1], padding="SAME")

        hidden_convolutional_layer_3 = tf.nn.relu(
            tf.nn.conv2d(hidden_max_pooling_layer_2, convolution_weights_3,
                    strides=[1, 1, 1, 1], padding="SAME") + convolution_bias_3)

        hidden_max_pooling_layer_3 = tf.nn.max_pool(hidden_convolutional_layer_3, ksize=[1, 2, 2, 1],
                                                strides=[1, 2, 2, 1], padding="SAME")

        hidden_convolutional_layer_3_flat = tf.reshape(hidden_max_pooling_layer_3, [-1, 256])

        final_hidden_activations = tf.nn.relu(
            tf.matmul(hidden_convolutional_layer_3_flat, feed_forward_weights_1) + feed_forward_bias_1)

        self.output_layer = tf.matmul(final_hidden_activations, feed_forward_weights_2) + feed_forward_bias_2

    def train_cnn(self):
        """ """
        # sample a mini_batch to train on
        mini_batch = random.sample(self.observations, self.mini_batch_size)
        # get the batch variables
        previous_states = [d[self.OBS_LAST_STATE_INDEX] for d in mini_batch]
        actions = [d[self.OBS_ACTION_INDEX] for d in mini_batch]
        rewards = [d[self.OBS_REWARD_INDEX] for d in mini_batch]
        current_states = [d[self.OBS_CURRENT_STATE_INDEX] for d in mini_batch]
        agents_expected_reward = []
        # this gives us the agents expected reward for each action we might
        agents_reward_per_action = self._session.run(self.output_layer, feed_dict={self.input_layer: current_states})
        for i in range(len(mini_batch)):
            if mini_batch[i][self.OBS_CRASH_INDEX]:
                # this was a crash frame so there is no future reward...
                agents_expected_reward.append(rewards[i])
            else:
                agents_expected_reward.append(
                    rewards[i] + self.FUTURE_REWARD_DISCOUNT * np.max(agents_reward_per_action[i]))

        # learn that these actions in these states lead to this reward
        self._session.run(self._train_operation, feed_dict={
            self._input_layer: previous_states,
            self._action: actions,
            self._target: agents_expected_reward})

        # save checkpoints for later
        if self._time % self.SAVE_EVERY_X_STEPS == 0:
            self._saver.save(self._session, self._checkpoint_path + '/network', global_step=self._time)