import settings
import tensorflow as tf
from LSTM import get_lstm_unit, ResidualLSTMCell

class Net:
    
    def __init__(self):
        self.logits = None
        self.cost = None


# class RecursiveNet(Net):

#     def __init__(self, sequences, labels, n_frames, training):
#         super().__init__()
#         # sequences: (batch, time, n_input)
#         # labels: (batch, n_classes)
#         # n_frames: (batch, )
#         hidden_in = tf.layers.dense(sequences, settings.n_hidden)
#         hidden_in_2 = tf.layers.dense(hidden_in, settings.n_hidden)
#         hidden_layer = tf.layers.dropout(hidden_in_2, rate=0.1, training=training)

#         gru_unit = GRUUnit(hidden_layer, n_frames, training)

#         # (frame, batch, n_classes)
#         self.logit_seqs = tf.layers.dense(gru_unit.outputs, settings.n_classes)
#         # (frame, batch, n_classes)
#         self.softmax_seqs = tf.nn.softmax(self.logit_seqs)
#         # (batch, n_classes)
#         self.logits = tf.reduce_mean(self.logit_seqs, axis=0)
#         # (batch, )
#         self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=labels)

#         self.cost = tf.reduce_mean(self.cross_entropy) + self.regularized


class ResLSTMNet(Net):

    def __init__(self, sequences, labels, n_frames, training):
        super().__init__()
        # sequences: (batch, time, n_input)
        # labels: (batch, n_classes)
        # n_frames: (batch, )

        # sequences = tf.layers.dense(sequences, settings.n_hidden, kernel_regularizer=tf.contrib.layers.l2_regularizer(settings.rr))
        # sequences = tf.layers.dropout(sequences, rate=(1 - settings.kp2))

        sequences = tf.expand_dims(sequences, [3])  # (batch, time, n_input, 1)
        sequences = tf.layers.conv2d(sequences, settings.n_hidden, [1, settings.n_inputs], padding='VALID', kernel_regularizer=tf.contrib.layers.l2_regularizer(settings.rr), activation=tf.nn.relu)  # (batch, time, 1, n_hidden)
        sequences = tf.squeeze(sequences, [2])

        res_lstm_output = ResidualLSTMCell(sequences, n_frames, training).outputs

        res_lstm_output = tf.expand_dims(res_lstm_output, [3])  # (batch, time, n_hidden, 1)
        logits = tf.layers.conv2d(res_lstm_output, settings.n_classes, [1, settings.n_hidden], padding='VALID', kernel_regularizer=tf.contrib.layers.l2_regularizer(settings.rr))  # (batch, time, 1, n_classes)
        logits = tf.squeeze(logits, [2])

        # logits = tf.layers.dense(res_lstm_output, settings.n_classes, kernel_regularizer=tf.contrib.layers.l2_regularizer(settings.rr))
        logits = tf.layers.dropout(logits, rate=(1 - settings.kp2))
        self.logits = tf.reduce_mean(logits, axis=1)  # (batch, n_classes)
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=labels)
        self.cost = tf.reduce_mean(self.cross_entropy)


class ConvNet(Net):

    def __init__(self, sequences, labels, n_frames, training):
        super().__init__()
        sequences = tf.expand_dims(sequences, -1)  # (batch, time, n_input, 1)
        sequences = tf.layers.conv2d(sequences, 128, [1, 1], padding='SAME') # (batch, time, n_input, 128)
        sequences = tf.layers.conv2d(sequences, 1024, [1, 1], padding='SAME') # (batch, time, n_input, 1024)
        sequences = tf.reshape(sequences, [-1, settings.n_frames, settings.n_inputs * 1024]) # (batch, time, n_input * 1024)
        logits = tf.layers.dense(sequences, 256, kernel_regularizer=tf.contrib.layers.l2_regularizer(settings.rr)) # (batch, time, 256)
        logits = tf.layers.dense(logits, settings.n_classes, kernel_regularizer=tf.contrib.layers.l2_regularizer(settings.rr)) # (batch, time, n_classes)
        self.logits = tf.reduce_mean(logits, axis=1)  # (batch, n_classes)
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=labels)
        self.cost = tf.reduce_mean(self.cross_entropy)
