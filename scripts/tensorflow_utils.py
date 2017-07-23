import os
from collections import deque

import tensorflow as tf

import utils

Y_pred = None
input_layer, output_layer = None, None


def train_cnn():
    pass

def _create_ploy_graph():
    """ """
    model = '10k-poly-model/ploy_distance_10k_epoch-10000'

    tf_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

    X = tf.placeholder(tf.float32)

    Y_pred = tf.Variable(tf.random_normal([1]), name='bias')

    for pow_i in range(1, 3):
        W = tf.Variable(tf.random_normal([1]), name='weight_%d' % pow_i)
        Y_pred = tf.add(tf.multiply(tf.pow(X, pow_i), W), Y_pred)

    saver = tf.train.Saver()

    sess = tf.Session()

    saver.restore(sess, "{}/{}".format(tf_dir, model))

    return Y_pred

def calc_distance_from_router():
    """ """
    global Y_pred

    signal_level = utils.get_signal_level_from_router('wlan0')
    return Y_pred.eval(feed_dict={X:-53}, session=sess)[0]


Y_pred = _create_ploy_graph()
Observations = deque()

class TensorFlowUtils(object):

    def __init__(self):
        pass

    def create_convolutional_network(state_frames, actions_count):
        # network weights
        convolution_weights_1 = tf.Variable(tf.truncated_normal([8, 8, state_frames, 32], stddev=0.01))

        convolution_bias_1 = tf.Variable(tf.constant(0.01, shape=[32]))
        convolution_weights_2 = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev=0.01))
        convolution_bias_2 = tf.Variable(tf.constant(0.01, shape=[64]))

        convolution_weights_3 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.01))
        convolution_bias_3 = tf.Variable(tf.constant(0.01, shape=[64]))

        feed_forward_weights_1 = tf.Variable(tf.truncated_normal([256, 256], stddev=0.01))
        feed_forward_bias_1 = tf.Variable(tf.constant(0.01, shape=[256]))

        feed_forward_weights_2 = tf.Variable(tf.truncated_normal([256, actions_count], stddev=0.01))
        feed_forward_bias_2 = tf.Variable(tf.constant(0.01, shape=[actions_count]))

        input_layer = tf.placeholder("float", [None, width, height, state_frames])

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

        output_layer = tf.matmul(final_hidden_activations, feed_forward_weights_2) + feed_forward_bias_2

        return input_layer, output_layer

