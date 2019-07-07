# coding=utf-8

import logging
import datetime
import time
import tensorflow as tf
import operator

from data_helper import load_train_data, load_test_data, load_embedding, batch_iter
from polymerization import LSTM_QA
import numpy as np
import sys


CAT_NUMBER = 10
ratio = 0.0

#------------------------- define parameter -----------------------------
tf.flags.DEFINE_string("train_file", "../lawQA/t_r_a_i_n.txt", "train corpus file")
tf.flags.DEFINE_string("test_file", "../lawQA/t_e_s_t_LONG.txt", "test corpus file")

#tf.flags.DEFINE_string("test_SHORT", "../lawQA/t_e_s_t.txt", "test corpus file")
tf.flags.DEFINE_string("train_LONG", "../lawQA/t_r_a_i_n_LONG.txt", "test corpus file")

tf.flags.DEFINE_string("embedding_file", "../lawQA/word2vec_150.txt", "embedding file")


tf.flags.DEFINE_integer("embedding_size", 150, "embedding size")
tf.flags.DEFINE_float("dropout", 1, "the proportion of dropout")
tf.flags.DEFINE_float("lr", 0.1, "the proportion of dropout")
tf.flags.DEFINE_integer("batch_size", 1000, "batch size of each batch")
tf.flags.DEFINE_integer("epoches", 300, "epoches")
tf.flags.DEFINE_integer("rnn_size", 300, "embedding size")
tf.flags.DEFINE_integer("num_rnn_layers", 1, "embedding size")
tf.flags.DEFINE_integer("evaluate_every", 250, "run evaluation")
tf.flags.DEFINE_integer("num_unroll_steps", 100, "length")
tf.flags.DEFINE_integer("max_grad_norm", 5, "embedding size")
tf.flags.DEFINE_integer("attention_matrix_size", 100, "matrix size")

tf.flags.DEFINE_float("margin", 0.2, "the proportion of dropout")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_float("gpu_options", 0.7, "use memory rate")

FLAGS = tf.flags.FLAGS
#----------------------------- define parameter end ----------------------------------

suffix = ""

if len(sys.argv) > 1:
    for arg in sys.argv[1:]:
        k = arg.split("=")[0]
        v = arg.split("=")[1]
        if  k == "lr":
            FLAGS.lr = float(v)
            suffix = suffix + "_" + arg
        elif k == "batch_size":
            FLAGS.batch_size = int(v)
            suffix = suffix + "_" + arg
        elif k == "rnn_size":
            FLAGS.rnn_size = int(v)
            suffix = suffix + "_" + arg
        elif k == "max_grad_norm":
            FLAGS.max_grad_norm = int(v)
            suffix = suffix + "_" + arg
        elif k == "attention_matrix_size":
            FLAGS.attention_matrix_size = int(v)
            suffix = suffix + "_" + arg
        elif  k == "margin":
            FLAGS.lr = float(v)
            suffix = suffix + "_" + arg
#----------------------------- define a logger -------------------------------
logger = logging.getLogger("execute")
logger.setLevel(logging.INFO)

fh = logging.FileHandler("./run.log"+suffix, mode="w")
fh.setLevel(logging.INFO)

fmt = "%(asctime)-15s %(levelname)s %(filename)s %(lineno)d %(process)d %(message)s"
datefmt = "%a %d %b %Y %H:%M:%S"
formatter = logging.Formatter(fmt, datefmt)

fh.setFormatter(formatter)
logger.addHandler(fh)

#----------------------------- define a logger end ----------------------------------

#------------------------------------load data -------------------------------
embedding, word2idx, idx2word = load_embedding(FLAGS.embedding_file, FLAGS.embedding_size)

ori_quests, cand_quests,neg_quests,cat_ids = load_train_data(FLAGS.train_file, word2idx, FLAGS.num_unroll_steps)

test_ori_quests, test_cand_quests, labels, results , test_cat_ids = load_test_data(FLAGS.test_file, word2idx, FLAGS.num_unroll_steps)


for_test_ori_quests, for_test_cand_quests, for_labels, for_results , for_test_cat_ids = load_test_data(FLAGS.train_LONG, word2idx, FLAGS.num_unroll_steps)
#test_like_train_ori_quests, test_like_train_cand_quests,test_like_train_neg_quests,test_like_train_cat_ids = load_train_data(FLAGS.test_TRAIN, word2idx, FLAGS.num_unroll_steps)

#----------------------------------- load data end ----------------------

def onehot_encoder(cat_ids_batch):
    return np.eye(CAT_NUMBER)[cat_ids_batch]

#----------------------------------- execute train model ---------------------------------
def run_step(sess, ori_batch, cand_batch, neg_batch,cat_ids_batch, lstm, dropout, is_optimizer=True,print_log=True):
    start_time = time.time()
    cat_ids_batch_onehot = onehot_encoder(cat_ids_batch)

    if is_optimizer:
        feed_dict = {
            lstm.ori_input_quests : ori_batch,
            lstm.cand_input_quests : cand_batch,
            lstm.neg_input_quests : neg_batch,
            lstm.keep_prob : dropout,
            lstm.cat_ids: cat_ids_batch_onehot
        }
        _, step, loss, acc, ori_cand_score, ori_neg_score, multi_acc = sess.run(
            [train_op, global_step, lstm.loss, lstm.acc, lstm.ori_cand_score, lstm.ori_neg_score, lstm.multi_acc], feed_dict)

        right, wrong, score = [0.0] * 3
        for i in range(0, len(ori_batch)):
            if ori_cand_score[i] > 0.55:
                right += 1.0
            else:
                wrong += 1.0
            score += ori_cand_score[i] - ori_neg_score[i]
        time_elapsed = time.time() - start_time
        if print_log:
            logger.info("step %s, loss %s, acc %s, wrong %s, score %s,multi_acc %s, %6.7f secs/batch" % (
            step, loss, acc, wrong, score, multi_acc / FLAGS.batch_size, time_elapsed))

    else:
        feed_dict = {
            lstm.test_input_q: ori_batch,
            lstm.test_input_a: cand_batch,
            lstm.keep_prob: dropout,
            lstm.cat_ids: cat_ids_batch_onehot
        }
        step, ori_cand_score = sess.run(
            [global_step,  lstm.test_q_a], feed_dict)
        loss = 0.0


    return loss, ori_cand_score

#---------------------------------- execute train model end --------------------------------------
def cal_acc(labels, results, total_ori_cand):
    if len(labels) == len(results) == len(total_ori_cand):
        retdict = {}
        for label, result, ori_cand in zip(labels, results, total_ori_cand):
            if result not in retdict:
                retdict[result] = []
            retdict[result].append((ori_cand, label))

        correct = 0
        for key, value in retdict.items():
            value.sort(key=operator.itemgetter(0), reverse=True)
            score, flag = value[0]
            if flag == 1:
                correct += 1
        acc = 1. * correct / len(retdict)

        MAP, MRR = 0, 0
        for key, value in retdict.items():
            p, AP = 0, 0
            MRR_check = False

            value.sort(key=operator.itemgetter(0), reverse=True)

            for idx, (score, flag) in enumerate(value):
                if flag == 1:
                    if not MRR_check:
                        MRR += 1 / (idx + 1)
                        MRR_check = True

                    p += 1
                    AP += p / (idx + 1)
            AP /= p
            MAP += AP

        num_questions = len(retdict)
        MAP /= num_questions
        MRR /= num_questions
        return acc, MAP, MRR

    else:
        logger.info("data error")
        logger.info(len(labels))
        logger.info(len(results))
        logger.info(len(total_ori_cand))
        return 0


#---------------------------------- execute valid model ------------------------------------------
def valid_model(sess, lstm, valid_ori_quests, valid_cand_quests, labels, results,test_cat_ids):
    total_loss, idx = 0, 0
    total_ori_cand = []
    #total_right, total_wrong, step = 0, 0, 0, 0
    for ori_valid, cand_valid, neg_valid,cat_ids_test in batch_iter(valid_ori_quests, valid_cand_quests, test_cat_ids,FLAGS.batch_size, 1,isvalid=True):
        loss, ori_cand = run_step(sess, ori_valid, cand_valid, cand_valid,cat_ids_test, lstm, FLAGS.dropout, False,False)
        total_loss += loss
        total_ori_cand.extend(ori_cand)
        #total_right += right
        #total_wrong += wrong
        idx += 1

    acc, MAP, MRR = cal_acc(labels, results, total_ori_cand)
    logger.info("evaluation acc:%s"%(acc))
    logger.info("evaluation MAP:%s"%(MAP))
    logger.info("evaluation MRR:%s"%(MRR))

    return acc, MAP, MRR
#---------------------------------- execute valid model end --------------------------------------


# def valid_model_train_format(sess, lstm, valid_ori_quests, valid_cand_quests, valid_neg_quests, test_cat_ids):
#
#     for ori_valid, cand_valid, neg_valid,cat_ids_test in batch_iter(valid_ori_quests, valid_cand_quests, test_cat_ids,FLAGS.batch_size, 1,neg_quests=valid_neg_quests):
#         run_step(sess, ori_valid, cand_valid, neg_valid,cat_ids_test, lstm, FLAGS.dropout, False)

#----------------------------------- begin to train -----------------------------------
with tf.Graph().as_default():
    with tf.device("/gpu:0"):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_options)
        session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement, log_device_placement=FLAGS.log_device_placement, gpu_options=gpu_options)
        with tf.Session(config=session_conf).as_default() as sess:
            lstm = LSTM_QA(FLAGS.batch_size, FLAGS.num_unroll_steps, embedding, FLAGS.embedding_size, FLAGS.rnn_size, FLAGS.num_rnn_layers, FLAGS.max_grad_norm, FLAGS.attention_matrix_size,ratio,m=FLAGS.margin)
            global_step = tf.Variable(0, name="globle_step",trainable=False)
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(lstm.loss, tvars),
                                          FLAGS.max_grad_norm)

            #optimizer = tf.train.GradientDescentOptimizer(lstm.lr)
            optimizer = tf.train.GradientDescentOptimizer(FLAGS.lr)
            optimizer.apply_gradients(zip(grads, tvars))
            train_op=optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)

            sess.run(tf.global_variables_initializer())

            acc_test_max, acc_train_max = 0.0, 0.0
            test_map, test_mrr, train_map, train_mrr = 0.0, 0.0, 0.0, 0.0

            last_map,last_mrr=0.0,0.0

            for ori_train, cand_train, neg_train, cat_ids_train in batch_iter(ori_quests, cand_quests, cat_ids,
                                                                              FLAGS.batch_size, FLAGS.epoches,
                                                                              neg_quests=neg_quests):
                run_step(sess, ori_train, cand_train, neg_train, cat_ids_train, lstm, FLAGS.dropout)
                cur_step = tf.train.global_step(sess, global_step)

                if cur_step % FLAGS.evaluate_every == 0 and cur_step != 0:

                    logger.info("start to evaluation model")
                    # valid_model(sess, lstm, valid_ori_quests, valid_cand_quests)
                    acc, MAP, MRR = valid_model(sess, lstm, test_ori_quests, test_cand_quests, labels, results,
                                                test_cat_ids)
                    if acc_test_max < acc:
                        acc_test_max = acc
                        test_map = MAP
                        test_mrr = MRR
                    logger.info("start to evaluation model for TRAIN")
                    acc, MAP, MRR = valid_model(sess, lstm, for_test_ori_quests, for_test_cand_quests, for_labels,
                                                for_results, for_test_cat_ids)
                    if acc_train_max < acc:
                        acc_train_max = acc
                        train_map = MAP
                        train_mrr = MRR

                    if last_map == MAP and last_mrr == MRR:
                        break
                    last_map, last_mrr = MAP, MRR

                    # logger.info("start to evaluation model in train format")
                    # valid_model_train_format(sess, lstm, test_like_train_ori_quests, test_like_train_cand_quests,test_like_train_neg_quests,test_like_train_cat_ids )
                    # logger.info("end of evaluation model in train format")

            logger.info("start to evaluation model")
            # valid_model(sess, lstm, valid_ori_quests, valid_cand_quests)
            acc, MAP, MRR = valid_model(sess, lstm, test_ori_quests, test_cand_quests, labels, results, test_cat_ids)
            if acc_test_max < acc:
                acc_test_max = acc
                test_map = MAP
                test_mrr = MRR
            logger.info("start to evaluation model for TRAIN")
            acc, MAP, MRR = valid_model(sess, lstm, for_test_ori_quests, for_test_cand_quests, for_labels, for_results,
                                        for_test_cat_ids)
            if acc_train_max < acc:
                acc_train_max = acc
                train_map = MAP
                train_mrr = MRR
            logger.info("max acc for TEST: {}, MAP: {}, MRR: {}".format(acc_test_max, test_map, test_mrr))
            logger.info("max acc for TRAIN: {},  MAP: {}, MRR: {}".format(acc_train_max, train_map, train_mrr))

