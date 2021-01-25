#!/usr/bin/env python
# encoding: utf-8

import os
import random
import collections
import tensorflow as tf
import numpy as np
import pandas as pd
from data_helper import DataHelper
from classifier import train_LR
from indicator import get_loc_data, positive_first, effort_aware


class CrossProjectModel(object):
    """The detail of the cross project model"""

    def __init__(self, params):
        tf.reset_default_graph()

        # model parameter
        self.batch_size = params['batch_size']
        self.num_epoch = params['num_epoch']
        self.fea_dim = params['fea_dim']
        self.layer_num = params['layer_num']
        self.hidden_dim = params['hidden_dim']
        self.output_dim = params['output_dim']
        self.repeat_num = params['repeat_num']
        self.l2 = params['l2']
        self.inner_project_weight = 1
        self.cross_project_weight = 2
        self.margin = 0.5

        # activation function
        self.activation = tf.nn.relu
        self.initializer = tf.random_uniform_initializer(minval=-0.8, maxval=0.8)
        self.global_step = 0
        self.best_loss = np.inf

        # classifiers
        self.classifiers = [train_LR]

        self.save_path = 'weights'

        # placeholder
        self.anchor_inp_a = tf.placeholder(dtype=tf.float32, shape=[None, self.fea_dim], name="Anchor_A")
        self.pos_inp_a = tf.placeholder(dtype=tf.float32, shape=[None, self.fea_dim], name="Positive_A")
        self.neg_inp_a = tf.placeholder(dtype=tf.float32, shape=[None, self.fea_dim], name="Negative_A")

        self.anchor_inp_b = tf.placeholder(dtype=tf.float32, shape=[None, self.fea_dim], name="Anchor_B")
        self.pos_inp_b = tf.placeholder(dtype=tf.float32, shape=[None, self.fea_dim], name="Positive_B")
        self.neg_inp_b = tf.placeholder(dtype=tf.float32, shape=[None, self.fea_dim], name="Negative_B")

        self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
        self.learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')

        # build model
        self.anchor_vec_a = self.encoder(self.anchor_inp_a, reuse=False)
        self.pos_vec_a = self.encoder(self.pos_inp_a, reuse=True)
        self.neg_vec_a = self.encoder(self.neg_inp_a, reuse=True)

        self.anchor_vec_b = self.encoder(self.anchor_inp_b, reuse=True)
        self.pos_vec_b = self.encoder(self.pos_inp_b, reuse=True)
        self.neg_vec_b = self.encoder(self.neg_inp_b, reuse=True)

        # compute loss
        self.loss1 = self.compute_loss(self.anchor_vec_a, self.pos_vec_a, self.neg_vec_a)
        self.loss2 = self.compute_loss(self.anchor_vec_b, self.pos_vec_b, self.neg_vec_b)
        self.loss3 = self.compute_loss(self.anchor_vec_a, self.pos_vec_b, self.neg_vec_b)
        self.loss4 = self.compute_loss(self.anchor_vec_b, self.pos_vec_a, self.neg_vec_a)
        self.loss = (self.loss1 + self.loss2) * self.inner_project_weight + (
                self.loss3 + self.loss4) * self.cross_project_weight

        # l2 regularization
        self.loss += tf.add_n(tf.get_collection('l2_loss')) * self.l2

        # optimizer
        self.opt = self.optimize(self.loss)

        # save model
        self.saver = tf.train.Saver(max_to_keep=10)

    def encoder(self, inp, reuse=False):
        """Embedding the original features"""

        for i in range(self.layer_num):
            inp_dim = inp.get_shape()[-1]
            with tf.variable_scope("encoder_%s" % i, reuse=reuse):
                if i == self.layer_num - 1:
                    hidden_dim = self.output_dim
                else:
                    hidden_dim = self.hidden_dim
                W = tf.get_variable(name="W", shape=[inp_dim, hidden_dim], initializer=self.initializer)
                tf.add_to_collection("l2_loss", tf.nn.l2_loss(W))
                b = tf.get_variable(name="b", shape=[hidden_dim], initializer=tf.zeros_initializer)
                hidden = tf.matmul(inp, W) + b

                if i < self.layer_num - 1:
                    hidden = self.activation(hidden)
                    hidden = tf.nn.dropout(hidden, keep_prob=self.keep_prob)
                    inp = hidden

        return hidden

    def compute_loss(self, anchor, pos, neg):
        """Calculate the triplet loss using Euclidean distance"""

        D_an = tf.reduce_sum(tf.square(anchor - neg), axis=-1)
        D_ap = tf.reduce_sum(tf.square(anchor - pos), axis=-1)
        loss = tf.reduce_mean(tf.square(tf.maximum(0., D_ap - D_an + self.margin)), axis=0) * 0.5
        return loss

    def optimize(self, loss, var_list=None):
        """Adam optimizer"""

        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        grad_var = optimizer.compute_gradients(loss, var_list=var_list)
        cliped_grad, global_norm = tf.clip_by_global_norm([grad for grad, var in grad_var], clip_norm=1)
        grad_var = [(cg, v) for cg, (g, v) in zip(cliped_grad, grad_var)]
        opt = optimizer.apply_gradients(grad_var)
        return opt

    def get_feed_dict(self, batch_data):
        """Obtain the input data for training model"""

        feed_dict = {self.anchor_inp_a: batch_data[0],
                     self.pos_inp_a: batch_data[1],
                     self.neg_inp_a: batch_data[2],
                     self.anchor_inp_b: batch_data[3],
                     self.pos_inp_b: batch_data[4],
                     self.neg_inp_b: batch_data[5]}

        return feed_dict

    def train(self, sess, train_data_helper, keep_prob=0.75, lr=0.1):
        """Model training"""

        batch_size = self.batch_size
        num_epoch = self.num_epoch
        for i in range(num_epoch):
            data_gen = train_data_helper.triplet_batch_generator(batch_size=batch_size)
            total_loss = 0.
            for batch_datas in data_gen:
                self.global_step += 1
                feed_dict = self.get_feed_dict(batch_datas)
                feed_dict[self.keep_prob] = keep_prob
                feed_dict[self.learning_rate] = lr
                _, loss = sess.run([self.opt, self.loss], feed_dict=feed_dict)
                total_loss += loss
            total_loss = total_loss / len(data_gen)
            if self.global_step % 10 == 0:
                # validate
                if total_loss <= self.best_loss:
                    print("epoch:%s, global step:%s, loss: %s" % (i, self.global_step, total_loss))
                    self.best_loss = total_loss
                    self.save_weights(sess, global_step=self.global_step)

    def batch_generator(self, datas, shuffle=False):
        """Generator the batch data"""

        num = len(datas[0])
        batch_size = self.batch_size
        all_data = []
        if shuffle:
            ids = random.sample(list(num), num)
            datas = [data[ids] for data in datas]
        for i in range((num + batch_size - 1) // batch_size):
            s = i * batch_size
            e = (i + 1) * batch_size
            batch_datas = [data[s:e] for data in datas]
            all_data.append(batch_datas)

        return all_data

    def encode_data_source(self, sess, datas):
        """Encoder the source data"""

        test_gen = self.batch_generator([datas])
        vectors = []
        for batch_data, in test_gen:
            train_vecs = sess.run(self.anchor_vec_a, feed_dict={self.anchor_inp_a: batch_data, self.keep_prob: 1.0})
            vectors.extend(train_vecs)

        return np.array(vectors)

    def encode_data_target(self, sess, datas):
        """Encoder the target data"""

        test_gen = self.batch_generator([datas])
        vectors = []
        for batch_data, in test_gen:
            train_vecs = sess.run(self.anchor_vec_b, feed_dict={self.anchor_inp_b: batch_data, self.keep_prob: 1.0})
            vectors.extend(train_vecs)

        return np.array(vectors)

    def get_most_similar(self, test_vecs, labeled_vecs, topk=1):
        """计算测试数据与标注数据的相似度，并选择topk"""
        test_vecs = test_vecs / (np.sqrt(np.sum(np.square(test_vecs), axis=-1, keepdims=True) + 1e-6))
        labeled_vecs = labeled_vecs / (np.sqrt(np.sum(np.square(labeled_vecs), axis=-1, keepdims=True) + 1e-6))
        cos_sims = np.dot(test_vecs, labeled_vecs.T)
        sorted_args = np.argsort(cos_sims, axis=-1)
        most_sim_ids = sorted_args[:, -topk:][:, ::-1]
        return most_sim_ids

    def test(self, sess, train_data_helper):
        """Model test"""

        train_datas, train_labels = train_data_helper.data_source
        train_vecs = self.encode_data_source(sess, train_datas)

        test_datas, test_labels = train_data_helper.test_data, train_data_helper.test_labels
        test_vecs = self.encode_data_target(sess, test_datas)

        # save indicator values
        all_model = []
        all_indicators = []

        # multiple classifiers can be specified simultaneously here
        for classifier in self.classifiers:
            model, indicators = classifier(train_vecs, train_labels, test_vecs, test_labels)
            all_model.append(model)
            all_indicators.append(indicators)

        return all_model, all_indicators

    def save_weights(self, sess, global_step):
        """Save model weights"""

        self.saver.save(sess, save_path=os.path.join(self.save_path, self.__class__.__name__), global_step=global_step)

    def load_weights(self, sess):
        """Load model weights"""

        ckpt = tf.train.get_checkpoint_state(self.save_path)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('Fail to restore ..., ckpt: %s' % ckpt)


def test_one_data(train_path, test_path, params):
    """Test one data set"""

    # Save all indicators from all models
    all_model_indicators = collections.defaultdict(dict)

    for i in range(params['repeat_num']):
        train_dp = DataHelper(train_path, test_path)
        model = CrossProjectModel(params)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        model.train(sess, train_dp)

        # load model parameters
        model.load_weights(sess)

        # test model and obtain the indicator values
        all_model, all_indicators = model.test(sess, train_dp)

        target_test_data, target_test_labels = train_dp.target_test_data, train_dp.target_test_labels
        test_vecs = model.encode_data_target(sess, train_dp.test_data)

        for index in range(len(model.classifiers)):
            single_model = all_model[index]
            single_indicator = all_indicators[index]

            # obtain the classifier name
            classify = model.classifiers[index].__name__

            pred = single_model.predict(test_vecs)
            pred_class = [0 if i < 0.5 else 1 for i in pred]

            test_df = get_loc_data(target_test_data, target_test_labels, train_dp.columns)
            test_df['pred'] = pred_class

            XYLuYp_sorted = test_df.sort_values('loc')
            eff_aware = effort_aware(XYLuYp_sorted, positive_first(XYLuYp_sorted))

            # merge the indicators
            new_indicators = dict(single_indicator, **eff_aware)
            for key in new_indicators:
                if key in all_model_indicators[classify].keys():
                    all_model_indicators[classify][key].append(new_indicators[key])
                else:
                    all_model_indicators[classify][key] = [new_indicators[key]]

    return all_model_indicators


def test_all_data(paths, params, outpath):
    """Test all data set"""

    mean_results = collections.defaultdict(list)
    all_results = collections.defaultdict(list)

    columns = ["Precision", "Recall", "F_2", "MCC", "auc", "EA_Precision", "EA_Recall", "EA_F2", "P_opt"]

    for test_path in os.listdir(paths):
        test_name = os.path.split(test_path)[-1].replace('.xlsx', '')
        for train_path in os.listdir(paths):
            if train_path != test_path:
                train_name = os.path.split(train_path)[-1].replace('.xlsx', '')

                complete_train_path = os.path.join(paths, train_path)
                complete_test_path = os.path.join(paths, test_path)

                # obtain all indicators
                total_indicator_data = test_one_data(complete_train_path, complete_test_path, params)
                for classifier in list(total_indicator_data.keys()):
                    indicator_data = total_indicator_data[classifier]

                    # obtain the key and value from dict
                    dict_columns = list(indicator_data.keys())

                    new_columns = [dict_columns.index(col) for col in columns]
                    indicator_values = np.array(list(indicator_data.values()))[new_columns, :]

                    all_row = [train_name, test_name] + indicator_values.tolist()
                    all_results[classifier].append(all_row)

                    # calculate the mean value of each indicator
                    mean_value = indicator_values.transpose()
                    mean_value = mean_value.mean(axis=0)

                    mean_row = [train_name, test_name] + mean_value.tolist()
                    mean_results[classifier].append(mean_row)

    # save the mean values
    for classifier in list(mean_results.keys()):
        print("Now is print", classifier, "mean model")
        one_data = mean_results[classifier]
        one_data = pd.DataFrame(one_data, columns=["source data", "target data"] + columns)
        final_path = os.path.join(outpath, 'mean_' + classifier + '.xlsx')
        one_data.to_excel(final_path, index=False)

    # save all the values
    for classifier in list(all_results.keys()):
        print("Now is print", classifier, "all model")
        one_data = all_results[classifier]
        one_data = pd.DataFrame(one_data, columns=["source data", "target data"] + columns)
        final_path = os.path.join(outpath, 'all_' + classifier + '.xlsx')
        one_data.to_excel(final_path, index=False)


if __name__ == '__main__':
    params = {"batch_size": 32, 'l2': 0.1, "fea_dim": 14, "layer_num": 2, "hidden_dim": 16,
              "output_dim": 4, "num_epoch": 30, "repeat_num": 50}

    data_path = 'datas/dataset'

    out_path = './results'

    test_all_data(data_path, params, out_path)

