import tensorflow as tf
import numpy as np
import pandas as pd

from data import input_data

train_data, test_data = input_data.read_train_and_test_data()


def test():

    saver = tf.train.Saver()

    with tf.Session() as sess:
        # test

        model = saver.restore('./model/lr_0.0001_epoch_100_batch_10_1511342809.8350415/acc_0.9981.ckpt')

        print('Test Start')
        # test data prepocess
        xs = np.concatenate((train_data['image'][:], test_data['image'][:]))
        ys = np.concatenate((train_data['is_contacted'][:], test_data['is_contacted'][:]))
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

test()