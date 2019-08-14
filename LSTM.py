import numpy as np
import tensorflow as tf
import settings
import math

def get_lstm_unit(sequences, n_frames, training, name=''):

    # sqrt_shape = int(math.sqrt(settings.n_hidden))
    # offset = tf.expand_dims(sequences, [3])  # (batch, time, n_input, 1)
    # offset = tf.layers.conv2d(offset, sqrt_shape, [1, settings.n_inputs], padding='VALID', kernel_regularizer=tf.contrib.layers.l2_regularizer(settings.rr), activation=tf.nn.relu)  # (batch, time, 1, n_hidden)
    # offset = tf.layers.max_pooling2d(offset, [settings.n_frames, 1], [1, 1])  # (batch, 1, 1, n_hidden)
    # offset = tf.reshape(offset, [-1, sqrt_shape])
    
    # offset = tf.reshape(offset, [-1, sqrt_shape, sqrt_shape, 1])  # (batch, 16, 16, 1)
    # offset = tf.layers.conv2d(offset, sqrt_shape, [16, 16], kernel_regularizer=tf.contrib.layers.l2_regularizer(settings.rr), activation=tf.nn.relu)  # (batch, 1, 1, 16)
    # offset = tf.reduce_mean(tf.squeeze(offset, [1, 2]), axis=-1) * settings.clipping

    # slist = []
    # for i in range(settings.batch_size):
    #     begin = tf.cast(offset[i], tf.int32)
    #     seq = tf.expand_dims(sequences[i][begin : begin+settings.clipping], axis=0)
    #     slist.append(seq)
    # sequences = tf.concat(slist, axis=0)

    cell = tf.nn.rnn_cell.LSTMCell(num_units=settings.n_hidden, name=name)
    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=settings.kp1)
    init_state = lstm_cell.zero_state(settings.batch_size, tf.float32)

    outputs, _ = tf.nn.dynamic_rnn(
        lstm_cell,
        sequences,
        initial_state=init_state,
        time_major=False,
        sequence_length=n_frames)
    
    # (batch, time, features)
    return outputs


class _SingleLSTMCell:

    def __init__(self, sequences, n_frames, training, name=''):

        hidden_forward = tf.layers.dense(sequences, settings.n_hidden, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(settings.rr))
        forward = get_lstm_unit(hidden_forward, n_frames, training, name= name + 'forward')

        # sequences = tf.reshape(sequences, [-1, settings.n_frames, settings.n_inputs])
        hidden_backward = tf.layers.dense(tf.reverse(sequences, [1]), settings.n_hidden, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(settings.rr))
        backward = get_lstm_unit(hidden_backward, n_frames, training, name= name + 'backward')

        self.outputs = tf.concat((forward, backward), axis=1)  # (batch, 2 * time, features)
        # self.outputs = forward  # (batch , time, features)


class ResidualLSTMCell:

    def __init__(self, sequences, n_frames, training):
        hidden_LSTM_layer = get_lstm_unit(sequences, n_frames, training, name='lstm_0')  # (batch, features)
        for i in range(settings.num_layers - 1):
            hidden_LSTM_layer = hidden_LSTM_layer + get_lstm_unit(sequences, n_frames, training, name='lstm_%d' % (i + 1))
        self.outputs = hidden_LSTM_layer  # (batch, 2 * time, features)
