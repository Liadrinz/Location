# import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf
import numpy as np
import settings
import argparse

from Net import ResLSTMNet, ConvNet
from load import DataLoader

from time import time

sequences = tf.placeholder(tf.float32, shape=[None, None, settings.n_inputs], name='sequences')
labels = tf.placeholder(tf.int32, shape=[None, settings.n_classes], name='labels')
n_frames = tf.placeholder(tf.int32, shape=[None, ], name='n_frames')
training = tf.placeholder(tf.bool, name='training')

global_step = tf.Variable(0, trainable=False)

net = ResLSTMNet(sequences, labels, n_frames, training)

total_cost = net.cost
train_op = tf.train.RMSPropOptimizer(settings.lr, momentum=0.1).minimize(total_cost, global_step=global_step)

logit_argmax = tf.argmax(net.logits, axis=-1)
label_argmax = tf.argmax(labels, axis=-1)
train_acc = tf.reduce_mean(tf.cast(tf.equal(logit_argmax, label_argmax), tf.float32))
test_acc = tf.reduce_mean(tf.cast(tf.equal(logit_argmax, label_argmax), tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=5)

cost_sum = tf.summary.scalar('train/cost', total_cost, collections=['train'])
train_acc_sum = tf.summary.scalar('train/train_acc', train_acc, collections=['train'])
test_acc_sum = tf.summary.scalar('test/test_acc', test_acc, collections=['test'])


with tf.Session() as sess:
    train_sum_op = tf.summary.merge_all('train')
    test_sum_op = tf.summary.merge_all('test')
    sw = tf.summary.FileWriter('summary/', sess.graph)

    sess.run(init)

    latest_ckpt = tf.train.latest_checkpoint('ckpt/')
    if latest_ckpt:
        saver.restore(sess, latest_ckpt)
    
    loader = DataLoader()
    idx = 1
    # prev_test_a = 0
    while True:
        # if time() >= 1565712754 + 17 * 3600:
        #     print('Done!')
        #     exit(0)
        xs, ys, fs = loader.next_batch(settings.batch_size)
        if idx % 10 == 0:
            # train
            _, g_step, cost, train_sum, g_step, train_a = sess.run([train_op, global_step, net.cost, train_sum_op, global_step, train_acc], feed_dict={
                sequences: xs,
                labels: ys,
                n_frames: fs,
                training: True
            })
            
            # test
            txs, tys, tfs = loader.next_batch(settings.batch_size, mode='test')
            test_sum, test_a = sess.run([test_sum_op, test_acc], feed_dict={
                sequences: txs,
                labels: tys,
                n_frames: tfs,
                training: False
            })

            # if test_a > prev_test_a or train_a - test_a < 0.05:
            prev_test_a = test_a
            saver.save(sess, 'ckpt/', global_step=g_step)
            print('Step: ', g_step)
            print('Cost: ', cost)
            print('Train accuracy: ', train_a)
            print('Test accuracy: ', test_a)
            
            sw.add_summary(train_sum, g_step)
            sw.add_summary(test_sum, g_step)
            # else:
            #     prev_test_a *= 0.96
            #     idx -= 10
            #     latest_ckpt = tf.train.latest_checkpoint('ckpt/')
            #     if latest_ckpt:
            #         saver.restore(sess, latest_ckpt)
        else:
            sess.run(train_op, feed_dict={
                sequences: xs,
                labels: ys,
                n_frames: fs,
                training: True
            })
        
        idx += 1
