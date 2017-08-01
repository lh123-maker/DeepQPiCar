#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Sam Mohamed

# This is heavily based off https://github.com/DanielSlater/PyGamePlayer/

"""
import os
import math
import random
import glob
import pickle

from collections import deque

import numpy as np
import tensorflow as tf

import utils

class TensorFlowUtils(object):

    
    img_width = 320
    img_height = 90
    state_frames = 4
    actions_count = 20

    mini_batch_size = 100

    LEARN_RATE = 1e-4
    FUTURE_REWARD_DISCOUNT = 0.1 # decay rate of past observations
    OBS_LAST_STATE_INDEX, OBS_ACTION_INDEX, OBS_REWARD_INDEX, OBS_CURRENT_STATE_INDEX, OBS_CRASH_INDEX = range(5)

    def __init__(self):
        """ """
        self.time = 0
        self.X = tf.placeholder(tf.float32)
        self.Y_pred, self.poly_session = self._create_poly_graph()

        directory = 'pickles'
        tf_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        save_path =  "{}/{}".format(tf_dir, directory)
        
        files = glob.glob('{}/*.p'.format(save_path))
        if len(files) > 0:
            self.observations = deque(pickle.load(open(files[0], 'rb')))
        else:
            self.observations = deque()

    def calc_distance_from_router(self):
        """ """
        signal_level = utils.get_signal_level_from_router('wlan0')
        return self.Y_pred.eval(feed_dict={ self.X : signal_level}, session=self.poly_session)[0]

    def _create_poly_graph(self):
        """ """
        model = 'models/poly_model/poly-model-1e-3'

        tf_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

        Y_pred = tf.Variable(tf.random_normal([1]), name='bias')

        for pow_i in range(1, 3):
            W = tf.Variable(tf.random_normal([1]), name='weight_%d' % pow_i)
            Y_pred = tf.add(tf.multiply(tf.pow(self.X, pow_i), W), Y_pred)

        saver = tf.train.Saver()

        sess = tf.Session()

        saver.restore(sess, "{}/{}".format(tf_dir, model))

        return Y_pred, sess

    def variable_summaries(self, var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def conv2d(self, x, W_dims, B_dims, layer_name, strides=[1, 1, 1, 1]):
        with tf.name_scope(layer_name):
            with tf.name_scope('weights'):
                weights = self.weight_variable(W_dims)
                self.variable_summaries(weights)  
            with tf.name_scope('biases'):
                biases = self.bias_variable(B_dims)
                self.variable_summaries(biases)
            with tf.name_scope('convolution'):
                layer = tf.nn.conv2d(x, weights, strides=strides, padding='SAME')
            with tf.name_scope('activation'):
                preactivate = layer + biases
                tf.summary.histogram('pre_activations', preactivate)

                activations = tf.nn.relu(preactivate, name='activation')
                tf.summary.histogram('activations', activations)
            with tf.name_scope('pooling'):
                layer = self.max_pool_2x2(activations)
                tf.summary.histogram('pooling', layer)
            return layer

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def fully_connected_layer(self, x, W_dims, B_dims, layer_name):
        with tf.name_scope(layer_name):
            with tf.name_scope('weights'):
                weights = self.weight_variable(W_dims)
                self.variable_summaries(weights)
            with tf.name_scope('biases'):
                biases = self.bias_variable(B_dims)
                self.variable_summaries(biases)
            with tf.name_scope('Wx_plus_b'):
                preactivate = tf.matmul(x, weights) + biases
                tf.summary.histogram('pre_activations', preactivate)
            if layer_name != 'output_layer':
                activations = tf.nn.relu(preactivate, name='activation')
                tf.summary.histogram('activations', activations)
                return activations
            return preactivate


    def _create_convolutional_network(self):
        """ """

        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        self.action = tf.placeholder("float", [None, self.actions_count], name='actions')
        self.target = tf.placeholder("float", [None], name='target_action')

        self.input_layer = tf.placeholder("float", [None, self.img_height, self.img_width, self.state_frames], name='state')

        conv1_layer = self.conv2d(self.input_layer, [9, 9, self.state_frames, 32], [32], 'conv1_layer')
        conv2_layer = self.conv2d(conv1_layer, [7, 7, 32, 64], [64], 'conv2_layer')
        conv3_layer = self.conv2d(conv2_layer, [5, 5, 64, 64], [64], 'conv3_layer')
        conv4_layer = self.conv2d(conv3_layer, [3, 3, 64, 64], [64], 'conv4_layer')

        shape = conv4_layer.get_shape().as_list()
        print('output shape of convoutions', shape)

        layer4_flat = tf.reshape(conv4_layer, [-1, shape[1]*shape[2]*64])
        hidden1 = self.fully_connected_layer(layer4_flat, [shape[1]*shape[2]*64, 1024], [1024], 'hidden_layer')

        with tf.name_scope('dropout'):
            self.keep_prob = tf.placeholder(tf.float32)
            tf.summary.scalar('dropout_keep_probability', self.keep_prob)
            dropped = tf.nn.dropout(hidden1, self.keep_prob)

        self.output_layer = self.fully_connected_layer(dropped, [1024, self.actions_count], [self.actions_count], 'output_layer')

        with tf.name_scope('action_evaluation'):
            readout_action = tf.reduce_sum(tf.multiply(self.output_layer, self.action), reduction_indices=1)
            tf.summary.histogram('action_evaluation', readout_action)

        with tf.name_scope('cost'):
            self.cost = tf.reduce_mean(tf.square(self.target - readout_action))
            tf.summary.histogram('convolution_cost', self.cost)

        with tf.name_scope('train'):
            self.train_operation = tf.train.AdamOptimizer(self.LEARN_RATE).minimize(self.cost, global_step=self.global_step)

        model = 'models/convnet/3layers-1e-4'
        tf_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        self.save_path =  "{}/{}".format(tf_dir, model)

        self.cnn_session = tf.Session()
        self.cnn_saver = tf.train.Saver()

        self.merged = tf.summary.merge_all()
        self._writer = tf.summary.FileWriter(self.save_path+ '/summaries')

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
        epochs = max(int((count * math.log(count))/count), 1)

        stuck_index = 0
        print("number of observations is {}.  Training for {} epochs".format(count, epochs))
        for i in range(epochs):
            global_step = tf.train.global_step(self.cnn_session, self.global_step)
            # sample a mini_batch to train on
            mini_batch = random.sample(self.observations, self.mini_batch_size-1)
            # get the batch variables
            previous_states = [d[self.OBS_LAST_STATE_INDEX] for d in mini_batch]
            actions = [d[self.OBS_ACTION_INDEX] for d in mini_batch]
            rewards = [d[self.OBS_REWARD_INDEX] for d in mini_batch]
            current_states = [d[self.OBS_CURRENT_STATE_INDEX] for d in mini_batch]
            agents_expected_reward = []

            # this gives us the agents expected reward for each action we might
            agents_reward_per_action = self.cnn_session.run(self.output_layer,
                feed_dict={self.input_layer: current_states,
                self.keep_prob: 0.5})
            
            for i in range(len(mini_batch)):
                if mini_batch[i][self.OBS_CRASH_INDEX]:
                    # this was a crash frame so there is no future reward...
                    stuck_index += 1
                    agents_expected_reward.append(rewards[i])
                else:
                    agents_expected_reward.append(
                        rewards[i] + self.FUTURE_REWARD_DISCOUNT * np.max(agents_reward_per_action[i]))

            # learn that these actions in these states lead to this reward
            summary, _ = self.cnn_session.run([self.merged, self.train_operation], feed_dict={
                self.input_layer: previous_states,
                self.action: actions,
                self.target: agents_expected_reward,
                self.keep_prob: 0.5})

            self._writer.add_summary(summary, global_step)

            print('I was stuck {} times'.format(stuck_index))
        # save checkpoints for later
        global_step = tf.train.global_step(self.cnn_session, self.global_step)
        self.cnn_saver.save(self.cnn_session, self.save_path+'/my_model', global_step=global_step)
        print('one more training finished.  global_step: %s' % global_step)

