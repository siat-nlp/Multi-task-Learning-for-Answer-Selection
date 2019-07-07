# coding:utf-8
import tensorflow as tf
from bilstm import biLSTM
from utils import feature2cos_sim, max_pooling, cal_loss_and_acc, get_feature, cal_attention

CAT_NUMBER = 10

class LSTM_QA(object):
    def __init__(self, batch_size, num_unroll_steps, embeddings, embedding_size, rnn_size, num_rnn_layers, max_grad_norm, attention_matrix_size, loss_ratio,l2_reg_lambda=0.0, adjust_weight=False,label_weight=[],is_training=True,m=0.1):
        """
        LSTM-BASED DEEP LEARNING MODELS FOR NON-FACTOID ANSWER SELECTION
        """
        # define input variable
        self.batch_size = batch_size
        self.embeddings = embeddings
        self.embedding_size = embedding_size
        self.adjust_weight = adjust_weight
        self.label_weight = label_weight
        self.rnn_size = rnn_size
        self.num_rnn_layers = num_rnn_layers
        self.num_unroll_steps = num_unroll_steps
        self.max_grad_norm = max_grad_norm
        self.l2_reg_lambda = l2_reg_lambda
        self.is_training = is_training

        self.keep_prob = tf.placeholder(tf.float32, name="keep_drop")
        
        self.lr = tf.Variable(0.0,trainable=False)
        self.new_lr = tf.placeholder(tf.float32, shape=[],name="new_learning_rate")
        self._lr_update = tf.assign(self.lr, self.new_lr)

        self.ori_input_quests = tf.placeholder(tf.int32, shape=[None, self.num_unroll_steps])
        self.cand_input_quests = tf.placeholder(tf.int32, shape=[None, self.num_unroll_steps])
        self.neg_input_quests = tf.placeholder(tf.int32, shape=[None, self.num_unroll_steps])
        self.test_input_q = tf.placeholder(tf.int32, shape=[None, self.num_unroll_steps])
        self.test_input_a = tf.placeholder(tf.int32, shape=[None, self.num_unroll_steps])
        self.cat_ids = tf.placeholder(tf.int32, [None, CAT_NUMBER], name='cat_ids')

        #embedding layer
        with tf.device("/cpu:0"),tf.name_scope("embedding_layer"):
            W = tf.Variable(tf.to_float(self.embeddings), trainable=True, name="W")
            ori_quests =tf.nn.embedding_lookup(W, self.ori_input_quests)
            cand_quests =tf.nn.embedding_lookup(W, self.cand_input_quests)
            neg_quests =tf.nn.embedding_lookup(W, self.neg_input_quests)
            test_q =tf.nn.embedding_lookup(W, self.test_input_q)
            test_a =tf.nn.embedding_lookup(W, self.test_input_a)

        #build LSTM network
        U = tf.Variable(tf.truncated_normal([2 * self.rnn_size, self.embedding_size], stddev=0.1), name="U")
        with tf.variable_scope("LSTM_scope", reuse=None):
            ori_q = biLSTM(ori_quests, self.rnn_size)
            ori_q_feat = tf.nn.tanh(max_pooling(ori_q))
        with tf.variable_scope("LSTM_scope", reuse=True):
            cand_att_weight = tf.sigmoid(tf.matmul(cand_quests, tf.reshape(tf.expand_dims(tf.matmul(ori_q_feat, U), 1), [-1, self.embedding_size, 1])))
            neg_att_weight = tf.sigmoid(tf.matmul(neg_quests, tf.reshape(tf.expand_dims(tf.matmul(ori_q_feat, U), 1), [-1, self.embedding_size, 1])))
            cand_a = biLSTM(tf.multiply(cand_quests, tf.tile(cand_att_weight, [1, 1, self.embedding_size])), self.rnn_size)
            neg_a = biLSTM(tf.multiply(neg_quests, tf.tile(neg_att_weight, [1, 1, self.embedding_size])), self.rnn_size)
            cand_q_feat = tf.nn.tanh(max_pooling(cand_a))
            neg_q_feat = tf.nn.tanh(max_pooling(neg_a))
            test_q_out = biLSTM(test_q, self.rnn_size)
            test_q_out = tf.nn.tanh(max_pooling(test_q_out))
            test_att_weight = tf.sigmoid(tf.matmul(test_a, tf.reshape(tf.expand_dims(tf.matmul(test_q_out, U), 1), [-1, self.embedding_size, 1])))
            test_a_out = biLSTM(tf.multiply(test_a, tf.tile(test_att_weight, [1, 1, self.embedding_size])), self.rnn_size)
            test_a_out = tf.nn.tanh(max_pooling(test_a_out))

        # multitasking
        with tf.name_scope("multitasking"):
            feature_size = int(ori_q_feat.get_shape()[1])

            fc1 = tf.layers.dense(ori_q_feat, feature_size * 2, activation=tf.nn.relu, name='fc1')
            fc2 = tf.layers.dense(fc1, feature_size, activation=tf.nn.relu, name='fc2')
            logits = tf.layers.dense(fc2, CAT_NUMBER, activation=tf.nn.sigmoid)

            # feature_size = int(ori_q_feat.get_shape()[1])

            # w = tf.get_variable(name='weights', shape=(feature_size, CAT_NUMBER, initializer=tf.random_normal_initializer())
            # b = tf.get_variable(name='bias', shape=(1, CAT_NUMBER), initializer=tf.zeros_initializer())

            # positive_qa = tf.concat([out_ori,out_cand],1,name="embedding_for_multitask")

            # logits = tf.matmul(ori_q_feat, w) + b

            entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.cat_ids, name='loss')
            loss_multitask = tf.reduce_mean(entropy)

        # acc
        self.ori_cand_score = feature2cos_sim(ori_q_feat, cand_q_feat)
        self.ori_neg_score = feature2cos_sim(ori_q_feat, neg_q_feat)
        loss_origin, self.acc = cal_loss_and_acc(self.ori_cand_score, self.ori_neg_score, m)

        self.loss = loss_origin * (1 - loss_ratio) + loss_multitask * loss_ratio

        self.test_q_a = feature2cos_sim(test_q_out, test_a_out)

        # multitasking_acc
        with tf.name_scope("multi_acc"):
            self.preds = tf.nn.softmax(logits)
            self.correct_preds = tf.equal(tf.argmax(self.preds, 1), tf.argmax(self.cat_ids, 1))
            self.multi_acc = tf.reduce_sum(tf.cast(self.correct_preds, tf.float32))

        def assign_new_lr(self, session, lr_value):
            session.run(self._lr_update, feed_dict={self.new_lr: lr_value})