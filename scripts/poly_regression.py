import os
import sys
import glob
import pickle
import argparse

import numpy as np
import tensorflow as tf

FLAGS = None

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def load_data():
    files = glob.glob(
        '/home/sameh/distance_data/*.p')
    data = dict()

    for f in files:
        data.update(pickle.load(open(f, 'rb')))

    _X = []
    _Y = []

    for time, _data in data.items():
        _X.append(_data['signal_level'])
        _Y.append(_data['distance_cm'])

    return _X, _Y


def main(_):

    learning_rate = 1e-3
    print(learning_rate)
    X, Y = load_data()

    X_train = list(chunks(X, len(X)//2))
    X_train_1, X_train_2 = X_train[0], X_train[1]

    Y_train = list(chunks(Y, len(Y)//2))
    Y_train_1, Y_train_2 = Y_train[0], Y_train[1]

    n_samples = len(X_train_1)

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ['instance-1.c.xenon-region-175015.internal:2223'], "worker": ['instance-1.c.xenon-region-175015.internal:2222',
        'instance-2.c.xenon-region-175015.internal:2222']})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)

    



    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d/gpu:0" % FLAGS.task_index,
                cluster=cluster)):

            if FLAGS.my_name == 'instance-1':
                train_X = X_train_1
                train_Y = Y_train_1
            else:
                train_X = X_train_2
                train_Y = Y_train_2

            # define graph same as before
            global_step = tf.contrib.framework.get_or_create_global_step()

            X = tf.placeholder(tf.float32)
            Y = tf.placeholder(tf.float32)

            Y_pred = tf.Variable(tf.random_normal([1]), name='bias')
            for pow_i in range(1, 3):
                W = tf.Variable(tf.random_normal([1]), name='weight_%d' % pow_i)
                Y_pred = tf.add(tf.multiply(tf.pow(X, pow_i), W), Y_pred)

            cost = tf.reduce_sum(tf.pow(Y_pred - Y, 2)) / (n_samples - 1)
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost,
                                                               global_step=global_step)

    
        # The StopAtStepHook handles stopping after running given steps.
        hooks = [tf.train.StopAtStepHook(last_step=100000)]

        with tf.train.MonitoredTrainingSession(master=server.target,
                                                   is_chief=(
                                                       FLAGS.task_index == 0),
                                                   checkpoint_dir="/home/sameh/poly_100k-1e-3",
                                                   hooks=hooks,
                                                   log_step_count_steps=100) as mon_sess:

            for (x, y) in zip(train_X, train_Y):
                mon_sess.run(optimizer, feed_dict={X: x, Y: y})

                while not mon_sess.should_stop():
                    training_cost = mon_sess.run(
                        cost, feed_dict={X: train_X, Y: train_Y})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    # Flags for defining the tf.train.ClusterSpec
    parser.add_argument(
        "--ps_hosts",
        type=str,
        default="",
        help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
        "--worker_hosts",
        type=str,
        default="",
        help="Comma-separated list of hostname:port pairs"
    )
    parser.add_argument(
        "--job_name",
        type=str,
        default="",
        help="One of 'ps', 'worker'"
    )
    # Flags for defining the tf.train.Server
    parser.add_argument(
        "--task_index",
        type=int,
        default=0,
        help="Index of task within the job"
    )
    # Flags for defining the tf.train.Server
    parser.add_argument(
        "--my_name",
        type=str,
        default="",
        help="Internal of the VM"
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

"""
python3 poly_regression.py \
     --ps_hosts=instance-1:2223 \
     --worker_hosts=instance-1:2222,instance-2:2222 \
     --job_name=ps --task_index=0  --my_name=instance-1 &> ps.log &

python3 poly_regression.py \
     --ps_hosts=instance-1:2223 \
     --worker_hosts=instance-1:2222,instance-2:2222 \
     --job_name=worker --task_index=0 --my_name=instance-1 &> worker.log &

python3 poly_regression.py \
     --ps_hosts=instance-1:2223 \
     --worker_hosts=instance-1:2222,instance-2:2222 \
     --job_name=worker --task_index=1 --my_name=instance-2 &> worker.log & 

"""