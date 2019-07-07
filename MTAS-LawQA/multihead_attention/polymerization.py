# coding:utf-8
import tensorflow as tf
from bilstm import biLSTM
from utils import feature2cos_sim, max_pooling, cal_loss_and_acc, multihead

CAT_NUMBER = 10
MULTI_NUMBER = 4

def fc(feat, feature_size):
    #with tf.variable_scope("fc",reuse=tf.AUTO_REUSE):
    #with tf.variable_scope("fc"):
    fc1 = tf.layers.dense(feat, feature_size / 2, activation=tf.nn.relu)
    fc2 = tf.layers.dense(fc1, feature_size, activation=tf.nn.relu)
    logits = tf.layers.dense(fc2, CAT_NUMBER, activation=tf.nn.sigmoid)
    return logits

class LSTM_QA(object):

    def __init__(self, batch_size, num_unroll_steps, embeddings, embedding_size, rnn_size, num_rnn_layers, max_grad_norm,attention_matrix_size,loss_ratio, l2_reg_lambda=0.0, adjust_weight=False,label_weight=[],is_training=True,m=0.1):
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


        self.test_input_q = tf.placeholder(tf.int32, shape=[None, self.num_unroll_steps], name='test_q')
        self.test_input_a = tf.placeholder(tf.int32, shape=[None, self.num_unroll_steps], name='test_a')
        self.q_cats = tf.placeholder(tf.int32, [None, CAT_NUMBER], name='q_cats')
        self.a_cats = tf.placeholder(tf.int32, [None, CAT_NUMBER], name='a_cats')

        #embedding layer
        with tf.device("/cpu:0"),tf.name_scope("embedding_layer"):
            W = tf.Variable(tf.to_float(self.embeddings), trainable=True, name="W")
            ori_quests =tf.nn.embedding_lookup(W, self.ori_input_quests)
            cand_quests =tf.nn.embedding_lookup(W, self.cand_input_quests)
            neg_quests =tf.nn.embedding_lookup(W, self.neg_input_quests)

            test_q =tf.nn.embedding_lookup(W, self.test_input_q)
            test_a =tf.nn.embedding_lookup(W, self.test_input_a)

        # run lstm without attention
        # with tf.variable_scope("LSTM_scope") as scope:
        #     ori_q = biLSTM(ori_quests, self.rnn_size)
        #     ori_q_feat = tf.nn.tanh(max_pooling(ori_q))
        #
        #     scope.reuse_variables()
        #
        #     cand_a = biLSTM(cand_quests, self.rnn_size)
        #     neg_a = biLSTM(neg_quests, self.rnn_size)
        #     cand_q_feat = tf.nn.tanh(max_pooling(cand_a))
        #     neg_q_feat = tf.nn.tanh(max_pooling(neg_a))
        #
        #     test_q_out = biLSTM(test_q, self.rnn_size)
        #     test_q_out = tf.nn.tanh(max_pooling(test_q_out))
        #     test_a_out = biLSTM(test_a, self.rnn_size)
        #     test_a_out = tf.nn.tanh(max_pooling(test_a_out))

        #build LSTM network
        with tf.variable_scope("LSTM_scope") as scope:
            ori_q = biLSTM(ori_quests, self.rnn_size)
            #ori_q_feat = tf.nn.tanh(max_pooling(ori_q))

            scope.reuse_variables()

            cand_a = biLSTM(cand_quests, self.rnn_size)
            neg_a = biLSTM(neg_quests, self.rnn_size)
            #cand_q_feat = tf.nn.tanh(max_pooling(cand_a))
            #neg_q_feat = tf.nn.tanh(max_pooling(neg_a))

            test_q_out = biLSTM(test_q, self.rnn_size)
            #test_q_out = tf.nn.tanh(max_pooling(test_q_out))
            test_a_out = biLSTM(test_a, self.rnn_size)
            #test_a_out = tf.nn.tanh(max_pooling(test_a_out))

        with tf.name_scope("att_weight"):
            # attention params
            att_W = {
            	'Wam': tf.Variable(tf.truncated_normal([2 * self.rnn_size, attention_matrix_size,MULTI_NUMBER], stddev=0.1)),
            	'Wqm': tf.Variable(tf.truncated_normal([2 * self.rnn_size, attention_matrix_size,MULTI_NUMBER], stddev=0.1)),
            	'Wms': tf.Variable(tf.truncated_normal([attention_matrix_size, MULTI_NUMBER], stddev=0.1))
            }
            ori_nq_feat, cand_q_feat = multihead(ori_q, cand_a, att_W)
            ori_q_feat, neg_q_feat = multihead(ori_q, neg_a, att_W)
            test_q_out, test_a_out = multihead(test_q_out, test_a_out, att_W)

        # multitasking
        with tf.variable_scope("multitasking") as scope:

            feature_size = int(ori_q_feat.get_shape()[1])

            logits_q = fc(ori_q_feat,feature_size)
            # scope.reuse_variables()
            logits_a = fc(cand_q_feat, feature_size)
            logits_ng_a = fc(neg_q_feat, feature_size)

            #feature_size = int(ori_q_feat.get_shape()[1])

            #w = tf.get_variable(name='weights', shape=(feature_size, CAT_NUMBER, initializer=tf.random_normal_initializer())
            #b = tf.get_variable(name='bias', shape=(1, CAT_NUMBER), initializer=tf.zeros_initializer())

            # positive_qa = tf.concat([out_ori,out_cand],1,name="embedding_for_multitask")

            #logits = tf.matmul(ori_q_feat, w) + b

            entropy_q = tf.nn.softmax_cross_entropy_with_logits(logits=logits_q, labels=self.q_cats, name='loss1')
            entropy_a= tf.nn.softmax_cross_entropy_with_logits(logits=logits_a, labels=self.q_cats, name='loss2')
            entropy_ng_a = tf.nn.softmax_cross_entropy_with_logits(logits=logits_ng_a, labels=self.a_cats, name='loss3')
            loss_multitask_q = tf.reduce_mean(entropy_q)
            loss_multitask_a = tf.reduce_mean(entropy_a)
            loss_multitask_ng_a = tf.reduce_mean(entropy_ng_a)

            loss_multitask = loss_multitask_q+loss_multitask_a+loss_multitask_ng_a

            loss_multitask = loss_multitask_q
        # acc
        self.ori_cand_score = feature2cos_sim(ori_q_feat, cand_q_feat)
        self.ori_neg_score = feature2cos_sim(ori_q_feat, neg_q_feat)
        loss_origin, self.acc = cal_loss_and_acc(self.ori_cand_score, self.ori_neg_score,m)

        self.loss = loss_origin * (1 - loss_ratio) + loss_multitask * loss_ratio

        self.test_q_a = feature2cos_sim(test_q_out, test_a_out)

        #evaluate multitasking_acc
        with tf.name_scope("multi_acc"):
            self.preds_q = tf.nn.softmax(logits_q)
            self.correct_preds_q = tf.equal(tf.argmax(self.preds_q, 1), tf.argmax(self.q_cats, 1))
            self.multi_acc_q = tf.reduce_sum(tf.cast(self.correct_preds_q, tf.float32))

            self.preds_a = tf.nn.softmax(logits_a)
            self.correct_preds_a = tf.equal(tf.argmax(self.preds_a, 1), tf.argmax(self.q_cats, 1))
            self.multi_acc_a = tf.reduce_sum(tf.cast(self.correct_preds_a, tf.float32))

            self.preds_ng_a = tf.nn.softmax(logits_ng_a)
            self.correct_preds_ng_a = tf.equal(tf.argmax(self.preds_ng_a, 1), tf.argmax(self.a_cats, 1))
            self.multi_acc_ng_a = tf.reduce_sum(tf.cast(self.correct_preds_ng_a, tf.float32))

            self.multi_acc = (self.multi_acc_q + self.multi_acc_a + self.multi_acc_ng_a)/3

            self.multi_acc = self.multi_acc_q 
    def assign_new_lr(self,session,lr_value):
        session.run(self._lr_update,feed_dict={self.new_lr:lr_value})
