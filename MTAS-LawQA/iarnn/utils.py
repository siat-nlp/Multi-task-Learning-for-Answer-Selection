import tensorflow as tf

def feature2cos_sim(feat_q, feat_a):
    norm_q = tf.sqrt(tf.reduce_sum(tf.multiply(feat_q, feat_q), 1))
    norm_a = tf.sqrt(tf.reduce_sum(tf.multiply(feat_a, feat_a), 1))
    mul_q_a = tf.reduce_sum(tf.multiply(feat_q, feat_a), 1)
    cos_sim_q_a = tf.div(mul_q_a, tf.multiply(norm_q, norm_a))
    return cos_sim_q_a

def max_pooling_3dim(lstm_out):
    height, width = int(lstm_out.get_shape()[1]), int(lstm_out.get_shape()[2])       # (step, length of inp    ut for one step)

    # do max-pooling to change the (sequence_lenght) tensor to 1-lenght tensor
    lstm_out = tf.expand_dims(lstm_out, -1)
    output = tf.nn.max_pool(
        lstm_out,
        ksize=[1, 1, width, 1],
        strides=[1, 1, 1, 1],
        padding='VALID')

    output = tf.reshape(output, [-1, height])

    return output


# return 1 output of lstm cells after pooling, lstm_out(batch, step, rnn_size * 2)
def max_pooling(lstm_out):
    height, width = int(lstm_out.get_shape()[1]), int(lstm_out.get_shape()[2])  # (step, length of input for one step)

    # do max-pooling to change the (sequence_lenght) tensor to 1-lenght tensor
    lstm_out = tf.expand_dims(lstm_out, -1)
    output = tf.nn.max_pool(
        lstm_out,
        ksize=[1, height, 1, 1],
        strides=[1, 1, 1, 1],
        padding='VALID')

    output = tf.reshape(output, [-1, width])

    return output


def avg_pooling(lstm_out):
    height, width = int(lstm_out.get_shape()[1]), int(lstm_out.get_shape()[2])  # (step, length of input for one step)

    # do max-pooling to change the (sequence_lenght) tensor to 1-lenght tensor
    lstm_out = tf.expand_dims(lstm_out, -1)
    output = tf.nn.avg_pool(
        lstm_out,
        ksize=[1, height, 1, 1],
        strides=[1, 1, 1, 1],
        padding='VALID')

    output = tf.reshape(output, [-1, width])

    return output


def cal_loss_and_acc(ori_cand, ori_neg, m):
    # the target function
    zero = tf.fill(tf.shape(ori_cand), 0.0)
    margin = tf.fill(tf.shape(ori_cand), m)
    with tf.name_scope("loss"):
        losses = tf.maximum(zero, tf.subtract(margin, tf.subtract(ori_cand, ori_neg)))
        loss = tf.reduce_sum(losses)
        # cal accurancy
    with tf.name_scope("acc"):
        correct = tf.equal(zero, losses)
        acc = tf.reduce_mean(tf.cast(correct, "float"), name="acc")
    return loss, acc


def get_feature(input_q, input_a, att_W):
    h_q, w = int(input_q.get_shape()[1]), int(input_q.get_shape()[2])
    h_a = int(input_a.get_shape()[1])

    output_q = max_pooling(input_q)

    reshape_q = tf.expand_dims(output_q, 1)
    reshape_q = tf.tile(reshape_q, [1, h_a, 1])
    reshape_q = tf.reshape(reshape_q, [-1, w])
    reshape_a = tf.reshape(input_a, [-1, w])

    M = tf.tanh(tf.add(tf.matmul(reshape_q, att_W['Wqm']), tf.matmul(reshape_a, att_W['Wam'])))
    M = tf.matmul(M, att_W['Wms'])

    S = tf.reshape(M, [-1, h_a])
    S = tf.nn.softmax(S)

    S_diag = tf.matrix_diag(S)
    attention_a = tf.matmul(S_diag, input_a)
    attention_a = tf.reshape(attention_a, [-1, h_a, w])

    output_a = max_pooling(attention_a)

    return tf.tanh(output_q), tf.tanh(output_a)


# cal final output with attention weight and lstm out (batch_size, step, rnn_size)
def cal_attention(input_q, input_a, batch_size, att_W):
    input_q = tf.transpose(input_q, [0, 2, 1])
    input_a = tf.transpose(input_a, [0, 2, 1])
    q_len = int(input_q.get_shape()[2])
    a_len = int(input_a.get_shape()[2])
    rnn_size = int(input_a.get_shape()[1])
    G = tf.tanh(tf.matmul(tf.matmul(input_q, tf.tile(tf.expand_dims(att_W['U'], 0), [batch_size, 1, 1]), True), input_a))
    delta_q = tf.nn.softmax(max_pooling_3dim(G))
    delta_a = tf.nn.softmax(max_pooling(G))

    final_ori_q = tf.reshape(tf.matmul(input_q, tf.reshape(delta_q, [-1, q_len, 1])), [-1, rnn_size])
    final_cand_a = tf.reshape(tf.matmul(input_a, tf.reshape(delta_q, [-1, a_len, 1])), [-1, rnn_size])

    #ori_q_output = tf.batch_matmul(input_q, tf.matrix_diag(delta_q))
    #cand_a_output = tf.batch_matmul(input_a, tf.matrix_diag(delta_a))
    ##final_ori_q = tf.tanh(max_pooling_3dim(ori_q_output))
    #final_ori_q = max_pooling_3dim(ori_q_output)
    #final_cand_a = max_pooling_3dim(cand_a_output)
    return tf.tanh(final_ori_q), tf.tanh(final_cand_a)

