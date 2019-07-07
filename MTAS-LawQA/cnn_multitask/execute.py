# coding=utf-8

import logging
import datetime
import time
import operator
import tensorflow as tf
import numpy as np


from data_helper import load_train_data, load_test_data, load_embedding, create_valid, batch_iter
from cnn import CNN


CAT_NUMBER = 10
ratio = 0.0

#------------------------- define parameter -----------------------------
tf.flags.DEFINE_string("train_file", "../lawQA/t_r_a_i_n.txt", "train corpus file")
tf.flags.DEFINE_string("test_file", "../lawQA/t_e_s_t_LONG.txt", "test corpus file")

#tf.flags.DEFINE_string("test_SHORT", "../lawQA/t_e_s_t.txt", "test corpus file")
tf.flags.DEFINE_string("train_LONG", "../lawQA/t_r_a_i_n_LONG.txt", "test corpus file")

tf.flags.DEFINE_string("embedding_file", "../lawQA/word2vec_150.txt", "embedding file")
tf.flags.DEFINE_string("filter_sizes", "1,2,3,5", "filter size of cnn")
tf.flags.DEFINE_integer("embedding_size", 150, "embedding size")
tf.flags.DEFINE_integer("sequence_len", 100, "max length")
tf.flags.DEFINE_integer("num_filters", 600, "the number of filter in every layer")
tf.flags.DEFINE_float("dropout", 1, "the proportion of dropout")
tf.flags.DEFINE_integer("batch_size", 1000, "batch size of each batch")
tf.flags.DEFINE_integer("epoches", 200, "epoches")
tf.flags.DEFINE_integer("evaluate_every", 500, "run evaluation")
tf.flags.DEFINE_float("lr", 0.001, "run evaluation")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_float("gpu_options", 0.7, "use memory rate")

FLAGS = tf.flags.FLAGS
#----------------------------- define parameter end ----------------------------------

#----------------------------- define a logger -------------------------------
logger = logging.getLogger("execute")
logger.setLevel(logging.INFO)

fh = logging.FileHandler("./run.log_"+str(ratio))
fh.setLevel(logging.INFO)

fmt = "%(asctime)-15s %(levelname)s %(filename)s %(lineno)d %(process)d %(message)s"
datefmt = "%a %d %b %Y %H:%M:%S"
formatter = logging.Formatter(fmt, datefmt)

fh.setFormatter(formatter)
logger.addHandler(fh)

#----------------------------- define a logger end ----------------------------------

#------------------------------------load data -------------------------------
embedding, word2idx, idx2word = load_embedding(FLAGS.embedding_file, FLAGS.embedding_size)

ori_quests, cand_quests,neg_quests,cat_ids = load_train_data(FLAGS.train_file, word2idx, FLAGS.sequence_len)

#train_quests, valid_quests = create_valid(zip(ori_quests, cand_quests))

test_ori_quests, test_cand_quests, labels, results , test_cat_ids = load_test_data(FLAGS.test_file, word2idx, FLAGS.sequence_len)


# for_test_ori_quests, for_test_cand_quests, for_labels, for_results , for_test_cat_ids = load_test_data(FLAGS.train_for_test, word2idx, FLAGS.sequence_len)
# test_like_train_ori_quests, test_like_train_cand_quests,test_like_train_neg_quests,test_like_train_cat_ids = load_train_data(FLAGS.test_file_like_train, word2idx, FLAGS.sequence_len)

for_test_ori_quests, for_test_cand_quests, for_labels, for_results , for_test_cat_ids = load_test_data(FLAGS.train_LONG, word2idx, FLAGS.sequence_len)
#test_like_train_ori_quests, test_like_train_cand_quests,test_like_train_neg_quests,test_like_train_cat_ids = load_train_data(FLAGS.test_SHORT, word2idx, FLAGS.sequence_len)



#----------------------------------- load data end ----------------------

#----------------------------------- build model --------------------------------------
filter_sizes = [int(filter_size.strip()) for filter_size in FLAGS.filter_sizes.strip().split(",")]
#----------------------------------- build model end ----------------------------------

def onehot_encoder(cat_ids_batch):
    return np.eye(CAT_NUMBER)[cat_ids_batch]

#----------------------------------- execute train model ---------------------------------
def run_step(sess, ori_batch, cand_batch, neg_batch,cat_ids_batch, cnn, dropout, is_optimizer=True,print_log=True):
    start_time = time.time()
    cat_ids_batch_onehot = onehot_encoder(cat_ids_batch)

    feed_dict = {
        cnn.org_quest:ori_batch,
        cnn.cand_quest:cand_batch, 
        cnn.neg_quest:neg_batch,
        cnn.keep_dropout:dropout,
        cnn.cat_ids:cat_ids_batch_onehot
    }

    if is_optimizer:
        _, step, loss, acc, ori_cand_score, ori_neg_score,multi_acc = sess.run([train_op, global_step, cnn.loss, cnn.acc, cnn.ori_cand_score, cnn.ori_neg_score,cnn.multi_acc], feed_dict)
    else:
        step, loss, acc, ori_cand_score, ori_neg_score,multi_acc = sess.run([global_step, cnn.loss, cnn.acc, cnn.ori_cand_score, cnn.ori_neg_score,cnn.multi_acc], feed_dict)


    right, wrong, score = [0.0] * 3
    for i in range(0 ,len(ori_batch)):
        if ori_cand_score[i] > 0.55:
            right += 1.0
        else:
            wrong += 1.0
        score += ori_cand_score[i] - ori_neg_score[i]
    time_elapsed = time.time() - start_time
    if print_log:
        logger.info("step %s, loss %s, acc %s, wrong %s, score %s,multi_acc %s, %6.7f secs/batch"%(step, loss, acc, wrong, score,multi_acc/FLAGS.batch_size, time_elapsed))
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
        acc = 1. * correct/len(retdict)

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
        return 0

#---------------------------------- execute valid model ------------------------------------------
def valid_model(sess, cnn, valid_ori_quests, valid_cand_quests, labels, results,test_cat_ids):
    total_loss, idx = 0, 0
    total_ori_cand = []
    #total_right, total_wrong, step = 0, 0, 0, 0
    for ori_valid, cand_valid, neg_valid,cat_ids_test in batch_iter(valid_ori_quests, valid_cand_quests, test_cat_ids,FLAGS.batch_size, 1,isvalid=True):
        loss, ori_cand = run_step(sess, ori_valid, cand_valid, cand_valid,cat_ids_test, cnn, FLAGS.dropout, False,False)
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
    #logger.info("%s, evaluation loss:%s, acc:%s"%(timestr, total_loss/step, total_right/(total_right + total_wrong)))
#---------------------------------- execute valid model end --------------------------------------


def valid_model_train_format(sess, cnn, valid_ori_quests, valid_cand_quests, valid_neg_quests, test_cat_ids):

    for ori_valid, cand_valid, neg_valid,cat_ids_test in batch_iter(valid_ori_quests, valid_cand_quests, test_cat_ids,FLAGS.batch_size, 1,neg_quests=valid_neg_quests):
        run_step(sess, ori_valid, cand_valid, neg_valid,cat_ids_test, cnn, FLAGS.dropout, False)


#----------------------------------- begin to train -----------------------------------
with tf.Graph().as_default():
    with tf.device("/gpu:0"):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_options)
        session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement, log_device_placement=FLAGS.log_device_placement, gpu_options=gpu_options)
        with tf.Session(config=session_conf).as_default() as sess:
            cnn = CNN(FLAGS.sequence_len, embedding, FLAGS.embedding_size, filter_sizes, FLAGS.num_filters,ratio)
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(FLAGS.lr)
            #optimizer = tf.train.GradientDescentOptimizer(1e-1)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            sess.run(tf.global_variables_initializer())
            #ori_quests, cand_quests = zip(*train_quests)
            #valid_ori_quests, valid_cand_quests = zip(*valid_quests)


            acc_test_max, acc_train_max =0.0, 0.0
            test_map, test_mrr, train_map, train_mrr = 0.0, 0.0, 0.0, 0.0

            for ori_train, cand_train, neg_train, cat_ids_train in batch_iter(ori_quests, cand_quests,cat_ids, FLAGS.batch_size, FLAGS.epoches,neg_quests=neg_quests):
                run_step(sess, ori_train, cand_train, neg_train, cat_ids_train,cnn, FLAGS.dropout)
                cur_step = tf.train.global_step(sess, global_step)
                
                if cur_step % FLAGS.evaluate_every == 0 and cur_step != 0:

                    logger.info("start to evaluation model")
                    #valid_model(sess, cnn, valid_ori_quests, valid_cand_quests)
                    acc, MAP, MRR=valid_model(sess, cnn, test_ori_quests, test_cand_quests, labels, results,test_cat_ids)
                    if acc_test_max < acc:
                        acc_test_max = acc
                        test_map = MAP
                        test_mrr = MRR
                    logger.info("start to evaluation model for TRAIN")
                    acc, MAP, MRR =valid_model(sess, cnn, for_test_ori_quests, for_test_cand_quests, for_labels, for_results, for_test_cat_ids)
                    if acc_train_max < acc:
                        acc_train_max = acc
                        train_map = MAP
                        train_mrr = MRR
                    # logger.info("start to evaluation model in train format")
                    # valid_model_train_format(sess, cnn, test_like_train_ori_quests, test_like_train_cand_quests,test_like_train_neg_quests,test_like_train_cat_ids )
                    # logger.info("end of evaluation model in train format")

            logger.info("start to evaluation model")
            # valid_model(sess, cnn, valid_ori_quests, valid_cand_quests)
            acc, MAP, MRR = valid_model(sess, cnn, test_ori_quests, test_cand_quests, labels, results, test_cat_ids)
            if acc_test_max < acc:
                acc_test_max = acc
                test_map = MAP
                test_mrr = MRR
            logger.info("start to evaluation model for TRAIN")
            acc, MAP, MRR = valid_model(sess, cnn, for_test_ori_quests, for_test_cand_quests, for_labels, for_results,
                                        for_test_cat_ids)
            if acc_train_max < acc:
                acc_train_max = acc
                train_map = MAP
                train_mrr = MRR
            logger.info("max acc for TEST: {}, MAP: {}, MRR: {}".format(acc_test_max,test_map,test_mrr))
            logger.info("max acc for TRAIN: {},  MAP: {}, MRR: {}".format(acc_train_max,train_map,train_mrr))

            # logger.info("start to evaluation model in train format")
            # valid_model_train_format(sess, cnn, test_like_train_ori_quests, test_like_train_cand_quests,test_like_train_neg_quests, test_like_train_cat_ids)
            # logger.info("evaluation finish")
            #---------------------------------- end train -----------------------------------
