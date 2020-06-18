#!/usr/bin/env python
# encoding: utf-8

import random
import itertools
import numpy as np
from collections import defaultdict
from data_loader import read_data, data_normalization


class DataHelper(object):
    """Wrapper class for build data"""
    def __init__(self, source_path, target_path, pos_num=1, neg_num=1, test_split=0.1):
        self.pos_num = pos_num
        self.neg_num = neg_num

        self.test_split = test_split
        self.labels = [0, 1]

        train_data, train_labels, self.target_train_data, self.target_train_labels, \
        self.target_test_data, self.target_test_labels, self.columns = read_data(source_path, target_path,
                                                                                  self.test_split, re_sample=True)

        self.data_source = train_data, train_labels
        self.data_target = data_normalization(self.target_train_data), self.target_train_labels.astype(np.int32)

        self.test_data = data_normalization(self.target_test_data)
        self.test_labels = self.target_test_labels.astype(np.int32)

        self.source_pos_neg_ids = self.get_pos_neg_ids(self.data_source)
        self.target_pos_neg_ids = self.get_pos_neg_ids(self.data_target)

    def get_pos_neg_ids(self, datas):
        """
        Classify the labels into different groups which contains the corresponding index
        :param datas: (features, labels)
        :return: ids
        """

        pos_neg_ids = defaultdict(list)
        i = 0
        for data, label in zip(*datas):
            pos_neg_ids[label].append(i)
            i += 1

        return pos_neg_ids

    def get_triplets(self, data, pos_neg_ids):
        """Randomly select the positive and negative sample to build the triplet"""

        triplets = defaultdict(list)

        for idx in range(len(data[0])):
            label = data[1][idx]
            pos = random.sample(pos_neg_ids[label] * 2, self.pos_num)
            # itertools.chain：take multiple arrays into a container simultaneously
            all_negs = list(
                itertools.chain(*[pos_neg_ids[neg_label] for neg_label in self.labels if neg_label != label]))
            neg = random.sample(all_negs * 2, self.neg_num)
            # id: the id-th sample
            # pid: the id of the corresponding positive sample
            # nid: the id of the corresponding negative sample
            for pid in pos:
                for nid in neg:
                    triplets[label].append([idx, pid, nid])

        return triplets

    def triplet_batch_generator(self, batch_size=32):
        """Generate the triplets with the batch size"""

        # obtain the triplets from source project
        source_triplets = self.get_triplets(self.data_source, self.source_pos_neg_ids)
        # obtain the triplets from target project
        target_triplets = self.get_triplets(self.data_target, self.target_pos_neg_ids)

        data = []
        for label, source_triplet in source_triplets.items():
            for source in source_triplet:
                target = random.sample(target_triplets[label], 1)[0]
                data.append(source + target)

        # shuffle the data
        random.shuffle(data)

        num = len(data)
        final_data = []

        for i in range((num + batch_size - 1) // batch_size):
            batch_idxs = data[i * batch_size: (i + 1) * batch_size]
            batch_data = [[] for _ in range(6)]
            for idxs in batch_idxs:
                source = [self.data_source[0][i] for i in idxs[:3]]
                target = [self.data_target[0][i] for i in idxs[3:]]
                tmp = source + target
                for i in range(6):
                    batch_data[i].append(tmp[i])
            batch_data = [np.array(bd) for bd in batch_data]   # 6 x 32
            final_data.append(batch_data)

        return np.array(final_data)
