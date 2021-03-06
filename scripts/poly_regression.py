import os
import sys
import glob
import pickle
import argparse

import numpy as np
import tensorflow as tf

FLAGS = None
BATCH_SIZE = 64
EPOCHS = 100
_index_in_epoch = 0


def shuffle(x, y):
    perm = np.arange(len(x))
    np.random.shuffle(perm)
    x = x[perm]
    y = y[perm]

    return (x, y)


def next_batch(data, labels, batch_size):
    """Return the next `batch_size` examples from this data set."""
    global _index_in_epoch
    start = _index_in_epoch
    _index_in_epoch += batch_size
    _num_examples = len(data)

    if _index_in_epoch > _num_examples:
        # Shuffle the data
        perm = np.arange(_num_examples)
        np.random.shuffle(perm)
        data = data[perm]
        labels = labels[perm]
        # Start next epoch
        start = 0
        _index_in_epoch = batch_size
        assert batch_size <= _num_examples

    end = _index_in_epoch
    return data[start:end], labels[start:end]


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

    X_train = list(chunks(_X, len(_X) // 2))
    X_train_1, X_train_2 = X_train[0], X_train[1]

    Y_train = list(chunks(_Y, len(_Y) // 2))
    Y_train_1, Y_train_2 = Y_train[0], Y_train[1]

    n_samples = len(X_train_1)

    if FLAGS.my_name == 'instance-1':
        train_X = X_train_1
        train_Y = Y_train_1
    else:
        train_X = X_train_2
        train_Y = Y_train_2

    return np.array(train_Y), np.array(train_Y)


def build_model(n_samples):
    learning_rate = 1e-3
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

    return X, Y, cost, optimizer, global_step


def train(server, X, Y, X_train, Y_train, optimizer, cost, global_step):
    logging_hook = tf.train.LoggingTensorHook(
        tensors={'step': global_step,
                 'loss': cost},
        every_n_iter=100)

    # The StopAtStepHook handles stopping after running given steps.
    hooks = [tf.train.StopAtStepHook(last_step=100000), logging_hook]

    with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=(
                                               FLAGS.task_index == 0),
                                           checkpoint_dir="/home/sameh/poly_100k-1e-3",
                                           hooks=hooks) as mon_sess:

        steps = len(X_train) // BATCH_SIZE
        train_num_examples = steps * BATCH_SIZE
        while not mon_sess.should_stop():
            for j in range(steps):
                # train for batch_size
                batch_x, batch_y = shuffle(
                    *next_batch(X_train, Y_train, BATCH_SIZE))
                mon_sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y})
                training_cost = mon_sess.run(
                    cost, feed_dict={X: batch_x, Y: batch_y})

            print(training_cost)


def main(_):

    X_train, Y_train = load_data()

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

            X, Y, cost, optimizer, global_step = build_model(
                (len(X_train) * 2))

        train(server, X, Y, X_train, Y_train, optimizer, cost, global_step)


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
