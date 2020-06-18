#!/usr/bin/env python
# encoding: utf-8

import os
import pickle
import random
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split


def data_normalization(datas):
    """Normalize the data"""

    datas = (datas - np.mean(datas, axis=0, keepdims=True)) / (np.std(datas, axis=0, keepdims=True) + 1e-6)
    return datas


def read_data(train_path, test_path, test_split=0.1, re_sample=True):
    """Load data"""

    train_name = os.path.split(train_path)[-1].split('.')[0]
    test_name = os.path.split(test_path)[-1].split('.')[0]

    # Dump the merged training set and test set into a new file
    dump_path = 'datas/temps/%s_%s.pkl' % (train_name, test_name)

    # If existing the file, load it directly
    if os.path.exists(dump_path) and re_sample is False:
        return pickle.load(open(dump_path, 'rb'))
    else:
        # load train data
        train = pd.read_excel(train_path, header=0)
        # eliminate nan
        train.fillna(value=0, inplace=True)
        columns = list(train.columns)

        # iloc(): read a specific column
        train_data = train.iloc[:, :-1].values
        train_data = data_normalization(train_data)
        train_labels = train.iloc[:, -1].values
        # shuffle the data and labels
        train_data, train_labels = shuffle_data(train_data, train_labels)

        # load test data
        test = pd.read_excel(test_path, header=0)
        test.fillna(value=0, inplace=True)
        test_data = test.iloc[:, :-1].values
        test_labels = test.iloc[:, -1].values

        # split data
        # (target_train_data, target_train_labels), (target_test_data, target_test_labels) = split_data(test_data, test_labels, split=test_split)

        (target_train_data, target_train_labels), (target_test_data, target_test_labels) = split_data(test,
                                                                                                        split=test_split)
        # dump the data into a file with dump_path
        pickle.dump([train_data, train_labels,
                     target_train_data, target_train_labels,
                     target_test_data, target_test_labels, columns], open(dump_path, 'wb'))

        return train_data, train_labels, target_train_data, target_train_labels, target_test_data, target_test_labels, columns


def shuffle_data(data, labels):
    """Shuffle the data"""

    num = len(data)
    ids = random.sample(list(range(num)), num)
    return data[ids], labels[ids]


def split_proportion(df: DataFrame, split, values=[1, 0], key='contains_bug'):

    positives, negatives = (df[df[key] == v] for v in values)
    (p_train, p_test), (n_train, n_test) = map(
        lambda dataset: train_test_split(dataset, test_size=split, shuffle=True, random_state=None),
        (positives, negatives))

    return p_train.append(n_train), p_test.append(n_test)


def split_data(df, split):
    """Split train set and test set"""

    train, test = split_proportion(df, 1 - split)

    train, test = train.loc[:].values, test.loc[:].values
    train_labels = train[:, [-1]]
    train_datas = train[:, : -1]
    test_labels = test[:, [-1]]
    test_datas = test[:, : -1]

    train_datas = np.array(train_datas)
    train_labels = np.array(train_labels).flatten()
    test_datas = np.array(test_datas)
    test_labels = np.array(test_labels).flatten()

    return (train_datas, train_labels), (test_datas, test_labels)
