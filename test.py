import tensorflow as tf
import numpy as np
import pandas as pd

from data import input_data

train_data, test_data = input_data.read_train_and_test_data()

model_path = './model/lr_0.001_epoch_30_batch_10_1511353388.6777036/'


def test():
    tf.reset_default_graph()
    saver = tf.train.import_meta_graph(model_path + 'acc_1.0.ckpt.meta')

    with tf.Session() as sess:
        # test
        print(tf.train.latest_checkpoint(model_path))
        saver.restore(sess, tf.train.latest_checkpoint(model_path))

        print(sess.run('x:0'))

        print('Test Start')
        # test data prepocess
        xs = np.concatenate((train_data['image'][:], test_data['image'][:]))
        ys = np.concatenate((train_data['is_contacted'][:], test_data['is_contacted'][:]))
        xs = np.transpose(xs, (0, 2, 1))
        xs = np.reshape(xs, (-1, 640, 512, 2))
        ys = np.array(ys)

        X = sess.run('X:0')
        Y = sess.run('Y:0')
        keep_prob = sess.run('keep_prob:0')

        accuracy = sess.run('accuracy:1')

        acc = sess.run([accuracy],
                       {
                           X: xs,
                           Y: ys,
                           keep_prob: 1.0
                       })

        print('TEST ACCURACY: {}'.format(acc))
        print('Test Finish')

test()