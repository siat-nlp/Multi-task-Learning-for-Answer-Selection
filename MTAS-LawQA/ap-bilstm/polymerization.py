
import tensorflow as tf
from bilstm import biLSTM
from utils import cal_attention, max_pooling, ortho_weight, uniform_weight, feature2cos_sim, cal_loss_and_acc

CAT_NUMBER = 10

class LSTM(object):
    def __init__(self, batch_size, quest_len, answer_len, embeddings, embedding_size, rnn_size, num_rnn_layers, max_grad_norm,loss_ratio, l2_reg_lambda=0.0, adjust_weight=False,label_weight=[],is_training=True,m=0.1):
        # define input variable
        self.batch_size = batch_size
        self.embeddings = embeddings
        self.embedding_size = embedding_size
        self.adjust_weight = adjust_weight
        self.label_weight = label_weight
        self.rnn_size = rnn_size
        self.num_rnn_layers = num_rnn_layers
        self.quest_len = quest_len 
        self.answer_len = answer_len 
        self.max_grad_norm = max_grad_norm
        self.l2_reg_lambda = l2_reg_lambda
        self.is_training = is_training

        self.keep_prob = tf.placeholder(tf.float32, name="keep_drop")
        
        self.lr = tf.Variable(0.0,trainable=False)
        self.new_lr = tf.placeholder(tf.float32, shape=[],name="new_learning_rate")
        self._lr_update = tf.assign(self.lr, self.new_lr)

        self.ori_input_quests = tf.placeholder(tf.int32, shape=[None, self.quest_len], name="ori_quest")
        self.cand_input_quests = tf.placeholder(tf.int32, shape=[None, self.answer_len], name="cand_quest")
        self.neg_input_quests = tf.placeholder(tf.int32, shape=[None, self.answer_len], name="neg_quest")
        self.test_input_q = tf.placeholder(tf.int32, shape=[None, self.quest_len], name="test_input_q")
        self.test_input_a = tf.placeholder(tf.int32, shape=[None, self.answer_len], name="test_input_a")
        self.cat_ids = tf.placeholder(tf.int32, [None, CAT_NUMBER], name='cat_ids')

        #embedding layer
        with tf.device("/cpu:0"),tf.name_scope("embedding_layer"):
            W = tf.Variable(tf.to_float(self.embeddings), trainable=True, name="W")
            ori_quests =tf.nn.embedding_lookup(W, self.ori_input_quests)
            cand_quests =tf.nn.embedding_lookup(W, self.cand_input_quests)
            neg_quests =tf.nn.embedding_lookup(W, self.neg_input_quests)
            test_quest =tf.nn.embedding_lookup(W, self.test_input_q)
            test_answer =tf.nn.embedding_lookup(W, self.test_input_a)

        #ori_quests = tf.nn.dropout(ori_quests, self.keep_prob)
        #cand_quests = tf.nn.dropout(cand_quests, self.keep_prob)
        #neg_quests = tf.nn.dropout(neg_quests, self.keep_prob)


        #build LSTM network
        with tf.variable_scope("LSTM_scope", reuse=None):
            ori_q = biLSTM(ori_quests, self.rnn_size)
        with tf.variable_scope("LSTM_scope", reuse=True):
            cand_a = biLSTM(cand_quests, self.rnn_size)
            neg_a = biLSTM(neg_quests, self.rnn_size)
            test_q = biLSTM(test_quest, self.rnn_size)
            test_a = biLSTM(test_answer, self.rnn_size)

        #----------------------------- cal attention -------------------------------
        with tf.variable_scope("attention", reuse=None) as scope:
            U = tf.get_variable("U", [2 * self.rnn_size, 2 * rnn_size], initializer=tf.truncated_normal_initializer(stddev=0.1))
            G = tf.matmul(tf.matmul(ori_q, tf.tile(tf.expand_dims(U, 0), [batch_size, 1, 1])), cand_a, adjoint_b=True)
            delta_q = tf.nn.softmax(tf.reduce_max(G, 2))
            delta_a = tf.nn.softmax(tf.reduce_max(G, 1))
            neg_G = tf.matmul(tf.matmul(ori_q, tf.tile(tf.expand_dims(U, 0), [batch_size, 1, 1])), neg_a, adjoint_b=True)
            delta_neg_q = tf.nn.softmax(tf.reduce_max(neg_G, 2))
            delta_neg_a = tf.nn.softmax(tf.reduce_max(neg_G, 1))
        with tf.variable_scope("attention", reuse=True) as scope:
            test_G = tf.matmul(tf.matmul(test_q, tf.tile(tf.expand_dims(U, 0), [batch_size, 1, 1])), test_a, adjoint_b=True)
            delta_test_q = tf.nn.softmax(tf.reduce_max(test_G, 2))
            delta_test_a = tf.nn.softmax(tf.reduce_max(test_G, 1))

        #-------------------------- recalculate lstm output -------------------------
        #ori_q_feat = tf.squeeze(tf.matmul(ori_q, tf.reshape(delta_q, [-1, self.quest_len, 1]), adjoint_a=True))
        #cand_q_feat = tf.squeeze(tf.matmul(cand_a, tf.reshape(delta_a, [-1, self.answer_len, 1]), adjoint_a=True))
        #neg_ori_q_feat = tf.squeeze(tf.matmul(ori_q, tf.reshape(delta_neg_q, [-1, self.quest_len, 1]), adjoint_a=True))
        #neg_q_feat = tf.squeeze(tf.matmul(neg_a, tf.reshape(delta_neg_a, [-1, self.answer_len, 1]), adjoint_a=True))
        #test_q_out = tf.squeeze(tf.matmul(test_q, tf.reshape(delta_test_q, [-1, self.quest_len, 1]), adjoint_a=True))
        #test_a_out = tf.squeeze(tf.matmul(test_a, tf.reshape(delta_test_a, [-1, self.answer_len, 1]), adjoint_a=True))
        ori_q_feat = max_pooling(tf.multiply(ori_q, tf.reshape(delta_q, [-1, self.quest_len, 1])))
        cand_q_feat = max_pooling(tf.multiply(cand_a, tf.reshape(delta_a, [-1, self.answer_len, 1])))
        neg_ori_q_feat = max_pooling(tf.multiply(ori_q, tf.reshape(delta_neg_q, [-1, self.quest_len, 1])))
        neg_q_feat = max_pooling(tf.multiply(neg_a, tf.reshape(delta_neg_a, [-1, self.answer_len, 1])))
        test_q_out = max_pooling(tf.multiply(test_q, tf.reshape(delta_test_q, [-1, self.quest_len, 1])))
        test_a_out = max_pooling(tf.multiply(test_a, tf.reshape(delta_test_a, [-1, self.answer_len, 1])))

        #-------------------------- recalculate lstm output end ---------------------
        # dropout
        #self.out_ori = tf.nn.dropout(self.out_ori, self.keep_prob)
        #self.out_cand = tf.nn.dropout(self.out_cand, self.keep_prob)
        #self.out_neg = tf.nn.dropout(self.out_neg, self.keep_prob)

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
