import tensorflow as tf
import numpy as np
import settings
import argparse

from Net import ResLSTMNet
from load import DataLoader

sequences = tf.placeholder(tf.float32, shape=[None, None, settings.n_inputs], name='sequences')
labels = tf.placeholder(tf.int32, shape=[None, settings.n_classes], name='labels')
n_frames = tf.placeholder(tf.int32, shape=[None, ], name='n_frames')
training = tf.placeholder(tf.bool, name='training')

# global_step = tf.Variable(0, trainable=False)

net = ResLSTMNet(sequences, labels, n_frames, training)
# total_cost = 0.4 * net.last_frame_cost + 0.6 * net.cost
# train_op = tf.train.AdamOptimizer(settings.lr).minimize(total_cost, global_step=global_step)

logit_argmax = tf.argmax(net.logits, axis=-1)
label_argmax = tf.argmax(labels, axis=-1)
# train_acc = tf.reduce_mean(tf.cast(tf.equal(logit_argmax, label_argmax), tf.float32))
# test_acc = tf.reduce_mean(tf.cast(tf.equal(logit_argmax, label_argmax), tf.float32))
# train_acc_lst = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(net.last_frame_logits, axis=-1), tf.argmax(labels, axis=-1)), tf.float32))
# test_acc_lst = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(net.last_frame_logits, axis=-1), tf.argmax(labels, axis=-1)), tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=None)

# cost_sum = tf.summary.scalar('train/cost', total_cost, collections=['train'])
# train_acc_sum = tf.summary.scalar('train/train_acc', train_acc, collections=['train'])
# test_acc_sum = tf.summary.scalar('test/test_acc', test_acc, collections=['test'])
# train_acc_lst_sum = tf.summary.scalar('train/train_acc_lst', train_acc, collections=['train'])
# test_acc_lst_sum = tf.summary.scalar('test/test_acc_lst', test_acc, collections=['test'])

with tf.Session() as sess:
    sess.run(init)

    latest_ckpt = tf.train.latest_checkpoint('ckpt/')
    if latest_ckpt:
        saver.restore(sess, latest_ckpt)
    
    loader = DataLoader()

    for frame in [256, 512]:
        correct = 0
        samples = 15
        total = samples * settings.batch_size * 2
        for i in range(samples):
            xs, ys, fs = loader.next_batch(settings.batch_size)
            xt, yt, ft = loader.next_batch(settings.batch_size, mode='test', max_frame=frame)
            logs, labs = sess.run([logit_argmax, label_argmax], feed_dict={
                sequences: xs,
                labels: ys,
                n_frames: fs,
                training: False
            })
            for log, lab in zip(logs, labs):
                correct += int(log == lab)
            logs, labs = sess.run([logit_argmax, label_argmax], feed_dict={
                sequences: xt,
                labels: yt,
                n_frames: ft,
                training: False
            })
            for log, lab in zip(logs, labs):
                correct += int(log == lab)
        print('frame length: ', frame, 'accuracy: ', correct / total)
