# coding:utf-8

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell 
## define lstm model and reture related features


# return n outputs of the n lstm cells
def biLSTM(x, hidden_size):

	# define the forward and backward lstm cells
	lstm_fw_cell = rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
	lstm_bw_cell = rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
	outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)
	output = tf.concat(outputs, 2)

	return output