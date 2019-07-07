import tensorflow as tf

#----------------------------- cal attention -------------------------------
def feature2cos_sim(feat_q, feat_a):
    norm_q = tf.sqrt(tf.reduce_sum(tf.multiply(feat_q, feat_q), 1))
    norm_a = tf.sqrt(tf.reduce_sum(tf.multiply(feat_a, feat_a), 1))
    mul_q_a = tf.reduce_sum(tf.multiply(feat_q, feat_a), 1)
    cos_sim_q_a = tf.div(mul_q_a, tf.multiply(norm_q, norm_a))
    return cos_sim_q_a

# return 1 output of lstm cells after pooling, lstm_out(batch, step, rnn_size * 2)
def max_pooling(lstm_out):
    height, width = int(lstm_out.get_shape()[1]), int(lstm_out.get_shape()[2])       # (step, length of input for one step)

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
    height, width = int(lstm_out.get_shape()[1]), int(lstm_out.get_shape()[2])       # (step, length of input for one step)
    
    # do max-pooling to change the (sequence_lenght) tensor to 1-lenght tensor
    lstm_out = tf.expand_dims(lstm_out, -1)
    output = tf.nn.avg_pool(
        lstm_out,
        ksize=[1, height, 1, 1],
        strides=[1, 1, 1, 1],
        padding='VALID')
    
    output = tf.reshape(output, [-1, width])
    
    return output

def cal_loss_and_acc(ori_cand, ori_neg,m):
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

def multihead(input_q, input_a, att_W):
    N = att_W['Wqm'].get_shape()[2]

    Output_q, Output_a = get_feature(input_q, input_a, att_W,0)
    for i in range(N):
        if i != 0:
            output_q, output_a = get_feature(input_q, input_a, att_W,i)
            Output_q = tf.concat([Output_q,output_q,],1)
            Output_a = tf.concat([Output_a,output_a,],1)
    return Output_q, Output_a

def get_feature(input_q, input_a, att_W,index):
    h_q, w = int(input_q.get_shape()[1]), int(input_q.get_shape()[2])
    h_a = int(input_a.get_shape()[1])

    output_q = max_pooling(input_q)

    reshape_q = tf.expand_dims(output_q, 1)
    reshape_q = tf.tile(reshape_q, [1, h_a, 1])
    reshape_q = tf.reshape(reshape_q, [-1, w])
    reshape_a = tf.reshape(input_a, [-1, w])

    M = tf.tanh(tf.add(tf.matmul(reshape_q, tf.squeeze(att_W['Wqm'][:,:,index])), tf.matmul(reshape_a, tf.squeeze(att_W['Wam'][:,:,index]))))
    M = tf.matmul(M, tf.expand_dims(att_W['Wms'][:,index],-1))

    S = tf.reshape(M, [-1, h_a])
    S = tf.nn.softmax(S)

    S_diag = tf.matrix_diag(S)
    attention_a = tf.matmul(S_diag, input_a)
    attention_a = tf.reshape(attention_a, [-1, h_a, w])

    output_a = max_pooling(attention_a)

    return tf.tanh(output_q), tf.tanh(output_a)

