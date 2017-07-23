import os

import tensorflow as tf

import utils


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


Y_pred = _create_ploy_graph()

def calc_distance_from_router():
	""" """
	global Y_pred

	signal_level = utils.get_signal_level_from_router('wlan0')
	return Y_pred.eval(feed_dict={X:-53}, session=sess)[0]