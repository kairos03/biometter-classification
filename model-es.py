# CopyRight kairos03 2017. All Right Reserved.

import tensorflow as tf
import time
from data import input_data
import numpy as np

learning_rate = 1e-4
total_epoch = 100
batch_size = 10
dropout_keep_prob = 0.9

name = 'lr_{}_epoch_{}_batch_{}_{}'.format(learning_rate, total_epoch, batch_size, time.time())
log_root = './log/' + name + '/'
model_root = './model/' + name + '/'

train_data, test_data = input_data.read_train_and_test_data()


def next_batch(batch, is_train=True, one_hot=True):
    if is_train:
        xs = train_data['image'][:(batch + 1) * batch_size]
        ys = train_data['is_contacted'][:(batch + 1) * batch_size]

    else:
        xs = train_data['image'][:]
        ys = train_data['is_contacted'][:]

    xs = np.transpose(xs, (0, 2, 1))
    xs = np.reshape(xs, (-1, 640, 512, 2))
    ys = np.array(ys)

    if one_hot:
        ys = np.array(ys).reshape(-1).astype(int)
        ys = np.eye(2)[ys]

    return xs, ys


def train():
    with tf.name_scope('input'):
        X = tf.placeholder(tf.float32, [None, 640, 512, 2])
        Y = tf.placeholder(tf.float32, [None, 2])
        keep_prob = tf.placeholder(tf.float32)

    #
    # with tf.name_scope('input_image'):
    #     tf.summary.image('input', x, 10)

    def var_weight(shape, stddev=0.1):
        var = tf.truncated_normal(shape, stddev=stddev)
        var_summary('W', var)
        return tf.Variable(var)

    def var_bias(shape):
        var = tf.constant(0.1, shape=shape)
        var_summary('B', var)
        return tf.Variable(var)

    def var_summary(name, var):
        with tf.name_scope(name):
            mean = tf.reduce_mean(var)
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('mean', mean)
            tf.summary.histogram('histogram', var)

    # with tf.name_scope('conv0'):
    #     filter0 = var_weight([1, 1, 2, 2])
    #     conv0 = tf.nn.conv2d(x, filter0, strides=[1, 1, 1, 1], padding='SAME')
    #     conv0 = tf.nn.relu(conv0)
    #     tf.summary.histogram('act', conv0)

    with tf.name_scope('conv1'):
        filter1 = var_weight([10, 10, 2, 16])
        conv1 = tf.nn.conv2d(X, filter1, strides=[1, 10, 8, 1], padding='SAME')
        conv1 = tf.nn.relu(conv1)
        tf.summary.histogram('act', conv1)

    with tf.name_scope('max_pool1'):
        pool1 = tf.nn.max_pool(conv1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    with tf.name_scope('conv2'):
        filter2 = var_weight([3, 3, 16, 32])
        conv2 = tf.nn.conv2d(pool1, filter2, strides=[1, 2, 2, 1], padding='SAME')
        conv2 = tf.nn.relu(conv2)
        tf.summary.histogram('act', conv2)

    with tf.name_scope('max_pool2'):
        pool2 = tf.nn.max_pool(conv2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    with tf.name_scope('conv3'):
        filter3 = var_weight([3, 3, 32, 64])
        conv3 = tf.nn.conv2d(pool2, filter3, strides=[1, 2, 2, 1], padding='SAME')
        conv3 = tf.nn.relu(conv3)
        tf.summary.histogram('act', conv3)

    # with tf.name_scope('max_pool3'):
    #     pool2 = tf.nn.max_pool(conv3, [1, 2, 2, 1], [1, 1, 1, 1], padding='SAME')

    with tf.name_scope('reshape'):
        reshaped = tf.reshape(conv3, [-1, 4 * 4 * 64])

    with tf.name_scope('fc1'):
        weight1 = var_weight([4 * 4 * 64, 512])
        bias1 = var_bias([512])
        fc1 = tf.matmul(reshaped, weight1) + bias1
        fc1 = tf.nn.relu(fc1)
        tf.summary.histogram('act', fc1)

    with tf.name_scope('dropout'):
        droped = tf.nn.dropout(fc1, keep_prob)

    with tf.name_scope('fc2'):
        weight2 = var_weight([512, 2])
        # bias2 = var_bias([2])
        model = tf.matmul(droped, weight2)  # + bias2

    #
    with tf.name_scope('matrix'):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1)), tf.float32))

        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('xent', cost)

    train_log = tf.summary.FileWriter(log_root+'train'.format(learning_rate, time.time()))
    test_log = tf.summary.FileWriter(log_root+'test'.format(learning_rate, time.time()))
    merged = tf.summary.merge_all()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        train_log.add_graph(sess.graph)

        total_batch = len(train_data['image']) // batch_size

        print('Train Start')
        for epoch in range(total_epoch):
            total_loss = 0
            summary = None

            for batch in range(total_batch):
                xs, ys = next_batch(batch)

                summary, loss, _ = sess.run([merged, cost, optimizer],
                                            feed_dict={
                                                X: xs,
                                                Y: ys,
                                                keep_prob: dropout_keep_prob})
                total_loss += loss

            train_log.add_summary(summary, epoch)
<<<<<<< HEAD
            print('epoch: {}, loss: {:.4}'.format(epoch, total_loss / total_batch))
=======
>>>>>>> 966595ce68ea0e4fb04e4ea6d47a2d3155a53c0e

            print('epoch: {:05}, loss: {:.5}'.format(epoch, total_loss / total_batch))

            if epoch % 5 == 4:
                xs, ys = next_batch(None, False)

                summary, acc = sess.run([merged, accuracy],
                                        feed_dict={
                                            X: xs,
                                            Y: ys,
                                            keep_prob: 1})

                test_log.add_summary(summary, epoch)
                print('accuracy: {:.4}'.format(acc))
<<<<<<< HEAD
=======

        saver.save(sess, model_root+'acc_{:.4}.ckpt'.format(acc))
        print('Train Finish')

        # test
        print('Test Start')
        # test data prepocess
        data = np.concatenate((train_data, test_data))
        xs = data['image'][:]
        ys = data['is_contacted'][:]
        xs = np.transpose(xs, (0, 2, 1))
        xs = np.reshape(xs, (-1, 640, 512, 2))
        ys = np.array(ys)

        acc = sess.run([accuracy],
                       feed_dict={
                           X: xs,
                           Y: ys,
                           keep_prob: 1
                       })

        print('TEST ACCURACY: {}'.format(acc))
        print('Test Finish')
>>>>>>> 966595ce68ea0e4fb04e4ea6d47a2d3155a53c0e

    train_log.close()
<<<<<<< HEAD
    saver.save(sess, model_root+'/acc_{:.4}'.format(acc))
=======
    test_log.close()
>>>>>>> 966595ce68ea0e4fb04e4ea6d47a2d3155a53c0e


train()
