# Copyright (C) KIST-Europe, Germany 2017
# Copyright (C) Moonhyeok Song 2017
# name : Read file from local storage
# due : 2017. 11. 22(ver.1)
# -*- coding: utf-8 -*-

import tensorflow as tf
import data.input_data as readdata
import time
import numpy as np

# kernel depths
k_1 = 2
k_2 = 4
k_3 = 32
f_node = 512

max_epoch = 50
batch_size = 30

# checkpoint : tf.summary 클래스 input으로 object를 넣어야 하는 이유?
class Summary(object):
    """ Design Tensorboard about CNN
          Args:

          Returns:
            next hidden layer data
    """
    @staticmethod
    def _summary(name, var):
        _mean = tf.reduce_mean(var)
        _variance = tf.reduce_mean(tf.square(var - _mean))
        tf.summary.scalar(name + '_mean', _mean)
        tf.summary.scalar(name + '_variance', _variance)
        tf.summary.histogram(name, var)
class ConvLayer(Summary):
    """ Define a Convolution layer model.
      Args:
        image : input image array
        ch_in : input image channel size(black&white = 1 -> depth : 2(front, Right image))
        ch_out : output map size (number of kernels)
        row_size, col_size : row and column size of kernel
        stride : kernel stride
        activation : activation function

      Returns:
        next hidden layer data
    """
    def __init__(self, image, ch_in, ch_out, col_size, row_size, stride, activation='none'):
        self.img = image
        # strides =  [batch, out_height, out_width, filter_height * filter_width * in_channels]
        # but first, end element is fixed '1'
        self.strd = stride
        self.act = activation.lower()

        # first step : 640, 512, 2, none(custom possible)
        W_shape = [col_size, row_size, ch_in, ch_out]

        # checkpoint : tf.truncated_normal?
        self.w = tf.Variable(tf.truncated_normal(W_shape, stddev=0.1), trainable=True, name='W')
        self._summary('W', self.w)

        if self.act != 'bn':
            self.B = tf.Variable(tf.constant(0.1, tf.float32, [ch_out]), trainable=True, name='B')
            self._summary('B',self.B)

    def out(self):
        # basic output
        WX = tf.nn.conv2d(self.img, self.w, strides=[1, self.strd, self.strd, 1], padding='SAME')

        # detect activation function
        if self.act == 'relu':
            return tf.nn.relu(WX + self.B)
        elif self.act == 'bn':
            return WX
        elif self.act == 'none':
            return WX + self.B
        else:
            raise ValueError('error : none activation function')
class FC_Layer(Summary):
    """ Define a Convolution layer model.
               Args:

               Returns:
        """
    def __init__(self, _input, n_in, n_out, activation='none'):
        self.input = _input
        self.n_in = n_in
        self.n_out = n_out
        self.act = activation.lower()

        # calculate weight, bis
        self.W = tf.Variable(tf.truncated_normal([n_in, n_out], stddev=0.1), trainable=True, name='W')
        self._summary('W', self.W)
        self.B = tf.Variable(tf.constant(0.0, tf.float32, [n_out]), trainable=True, name='B')
        self._summary('B', self.B)

    def out(self):
        if self.act == 'relu':
            return tf.nn.relu(tf.matmul(self.input, self.W) + self.B)
        elif self.act == 'none':
            return tf.matmul(self.input, self.W) + self.B
        else:
            raise ValueError('ERROR: unsupported activation option')

class batch_norm(Summary):
    """ Define a Convolution layer model.
           Args:

           Returns:
    """
    def __init__(self, input_tensor, out_size, _train, activation='none'):
        self.input = input_tensor
        self.act = activation.lower()

        self.beta = tf.Variable(tf.constant(0.0, shape=[out_size]), trainable=True, name='beta')
        self._summary('beta', self.beta)
        self.gamma = tf.Variable(tf.constant(1.0, shape=[out_size]), trainable=True, name='gamma')
        self._summary('gamma', self.gamma)

        batch_mean, batch_var = tf.nn.moments(input_tensor, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)
        ema_apply_op = ema.apply([batch_mean, batch_var])

        if _train:
            with tf.control_dependencies([ema_apply_op]):
                self.mean, self.var = tf.identity(batch_mean), tf.identity(batch_var)
        else:
            self.mean, self.var = ema.average(batch_mean), ema.average(batch_var)

    def out(self):
        norm = tf.nn.batch_normalization(self.input, self.mean, self.var, self.beta, self.gamma, 1e-5)
        if self.act == 'relu':
            return tf.nn.relu(norm)
        elif self.act == 'none':
            return norm
        else:
            raise ValueError('ERROR: unsupported activation option')

### Create Convolution Neural Networks
def cnn_model(image_array, result, p_keep = None):
    """ Define a Convolution layer model.
          Args:
            image_array : input image data(form : 1Darray)

          Returns:

    """

    train_step = False if p_keep is None else True

    # Todo : ksize have to be modified in MaxPool1,2
    with tf.variable_scope('Conv1'):
        # 2 : front,right image data (black&white)
        # .out() -> activate ReLU function
        # kernel size define reference : http://bit.ly/2jFdBjP
        step1 = ConvLayer(image_array, k_1, k_2, 50, 50, 4, activation='ReLU').out()

    with tf.name_scope('MaxPool1'):
        step1 = tf.nn.max_pool(step1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    with tf.variable_scope('Conv2'):
        step2 = ConvLayer(step1, k_2, k_3, 10, 10, 4, activation='ReLU').out()
    with tf.name_scope('MaxPool2'):
        step2 = tf.nn.max_pool(step2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    with tf.variable_scope('Conv3'):
        step3 = ConvLayer(step2, k_3, 64, 3, 3, 2, activation='ReLU').out()
        step4 = tf.reshape(step3, [-1, 5 * 4 * 64])

    with tf.variable_scope('output'):
        step5 = FC_Layer(step4, 5*4*64, f_node, activation='ReLU').out()
        if train_step: step5 = tf.nn.dropout(step5, p_keep)
        step6 = FC_Layer(step5, f_node, 50, activation='ReLU').out()
        if train_step: step6 = tf.nn.dropout(step6, p_keep)
        Ylogits = FC_Layer(step6, p_keep, 2).out()
    Y = tf.nn.softmax(Ylogits, name='Y')


    # checkpoint : there need to re-customize
    with tf.variable_scope('cross_entropy'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=result)
        cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.variable_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(result, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # incorrects = tf.squeeze(tf.where(tf.logical_not(correct_prediction)), [1])

    return Y, cross_entropy, accuracy

# checkpoint
def one_hot(arr, depth):
    arr = np.array(arr).reshape(-1).astype(int)
    hot = np.eye(depth)[arr]
    return hot

def next_batch(batch, is_train=True):
    if is_train:
        xs = train['image'][:(batch+1)*batch_size]
        ys = train['is_contacted'][:(batch+1)*batch_size]
    else:
        xs = train['image'][:]
        ys = train['is_contacted'][:]

    xs = np.transpose(xs, (0, 2, 1))
    xs = np.reshape(xs, (-1, 640, 512, 2))
    ys = np.array(ys)
    ys = one_hot(ys, 2)
    return xs, ys

def phase_train(MAX_EPOCH, BATCH_SIZE, SAVE_MODEL_PATH, LOG_PATH):
    start = time.time()

    # shape of train, test : [total_num, kernel_num, col_size, row_size]
    n_train = len(train['image'])
    #  n_test = test[0]

    with tf.Graph().as_default() as train_g:
        # input : 640*512 pix image, black&white
        # output : 1 = separate, 0 = contact
        X_ = tf.placeholder(tf.float32, [None, 640, 512, 2], name='X_')
        Y_ = tf.placeholder(tf.float32, [None, 2], name='Y_')
        with tf.variable_scope('configure'):
            # dropout keep probability
            p_keep = tf.placeholder(tf.float32, name='p_keep')
            # learning rate
            global_step = tf.Variable(0, name='global_step', trainable=False)
            # decayed LR. this code is recommended to lower the LR as the training step
            # whenever it reach a certain step, tf will edit LR
            LR = tf.train.exponential_decay(0.001, global_step, int(MAX_EPOCH / 5), 0.5, staircase=True, name='LR')

        # load inference model
        Y, cross_entropy, accuracy = cnn_model(X_, Y_, p_keep)
        train_op = tf.train.AdamOptimizer(LR).minimize(cross_entropy, global_step=global_step)

        with tf.variable_scope('Metrics'):
            tf.summary.scalar('accuracy', accuracy)
            tf.summary.scalar('cross_entropy', cross_entropy)
            tf.summary.scalar('learning_rate', LR)

        # set tensorboard summary and saver
        merged = tf.summary.merge_all()
        saver = tf.train.Saver(max_to_keep=100)

        # training session
        print('----- training start -----')
        with tf.Session() as sess:
            sum_writer = tf.summary.FileWriter(LOG_PATH + "LR3e-1/50_50_4/10_10_4/3_3_2", sess.graph)
            tf.global_variables_initializer().run() # .run() is possible! because of with tf.Session()
            step = 1
            while step <= MAX_EPOCH:
                for batch in range(int(n_train / BATCH_SIZE) + 1):
                    # checkpoint
                    s = batch * BATCH_SIZE
                    e = (batch + 1) * BATCH_SIZE if (batch + 1) * BATCH_SIZE < n_train else n_train
                    batch_xs, batch_ys = next_batch(batch)
                    if e <= s: break
                    _, summary, acc, ent = sess.run([train_op, merged, accuracy, cross_entropy],
                                                    {X_: batch_xs, Y_: batch_ys, p_keep: 0.75})
                    sum_writer.add_summary(summary, step)
                    print('[%6.2f] step:%3d, size:%3d, lr:%f, accuracy:%f, cross entropy:%f'
                          % (time.time() - start, step, e - s, LR.eval(), acc, ent))
                    if (MAX_EPOCH - step) < 10 or step % 100 == 0:
                        saver.save(sess, SAVE_MODEL_PATH, global_step=step)
                    step += 1
                    if step > MAX_EPOCH: break
        print('-----  training end  -----')

# test session
def phase_test(BATCH_SIZE, SAVE_MODEL_PATH, LOG_PATH):
    start = time.time()

    # shape of train, test : [total_num, kernel_num, col_size, row_size]
    n_test = len(test['image'])
    # input : 640*512 pix image, black&white
    # output : 1 = separate, 0 = contact
    X_ = tf.placeholder(tf.float32, [None, 640, 512, 2], name='X_')
    Y_ = tf.placeholder(tf.float32, [None, 2], name='Y_')

    with tf.Graph().as_default() as test_g:
        # load inference model
        Y, cross_entropy, accuracy = cnn_model(X_, Y_)

        with tf.variable_scope('Metrics'):
            tf.summary.scalar('accuracy', accuracy)
            tf.summary.scalar('cross_entropy', cross_entropy)

        # set tensorboard summary and saver
        merged = tf.summary.merge_all()
        saver = tf.train.Saver()

        print('----- test start -----')
        with tf.Session() as sess:
            sum_writer = tf.summary.FileWriter(LOG_PATH + "LR3e-1/50_50_4/10_10_4/3_3_2", sess.graph)
            tf.global_variables_initializer().run() # .run() is possible! because of with tf.Session()

            saver.restore(sess, SAVE_MODEL_PATH)
            avg_acc = 0


            for step in range(int(n_test / BATCH_SIZE) + 1):
                # checkpoint
                s = step * BATCH_SIZE
                e = (step + 1) * BATCH_SIZE if (step + 1) * BATCH_SIZE < n_test else n_test
                # batch_xs, batch_ys = next_batch(n_test)
                if e <= s: break
                summary, acc, ent = sess.run([merged, accuracy, cross_entropy],
                                                {X_: n_test['image'][s:e], Y_:  n_test['is_contacted'][s:e]})
                sum_writer.add_summary(summary, step)
                avg_acc += acc*(e-s)
                print('[%6.2f] step:%3d, size:%3d, accuracy:%f, cross entropy:%f'
                      % (time.time() - start, step+1, e - s, acc, ent))
        print('-----  test end  -----')
        print('[%6.2f] total average accuracy : %f' % (time.time()-start, avg_acc/n_test))

# do train with batch size 60 and maximum step 50
train, test = readdata.read_train_and_test_data()

SAVE='/home/mike2ox/biometter-classification/train/model/ckpt'
LOG='/home/mike2ox/biometter-classification/train/log/'
phase_train(max_epoch, batch_size, SAVE, LOG)
MODEL='/home/mike2ox/biometter-classification/test/model/ckpt'
LOG='/home/mike2ox/biometter-classification/test/log/'
phase_test(100, MODEL, LOG)

