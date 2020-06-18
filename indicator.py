#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import pandas as pd
from pandas import DataFrame, concat
from sklearn.metrics import auc


def get_metrics(y_true, y_pred):
    """Calculate indicator values"""

    # Obtain the four terms of binary classification
    TN, FP, FN, TP = np.fromiter((sum(
        bool(j >> 1) == bool(y_true[i]) and
        bool(j & 1) == bool(y_pred[i])
        for i in range(len(y_true)))
        for j in range(4)), float)

    # Accuracy = (TN + TP) / (TN + FP + FN + TP + 1e-8)
    Precision = TP / (TP + FP + 1e-8)
    Recall = TP / (TP + FN + 1e-8)
    # FPR = FP / (FP + TN + 1e-8)

    # F_measure = 2 * Recall * Precision / (Recall + Precision + 1e-8)
    # g_mean = np.sqrt((TN / (TN + FP + 1e-8)) * (TP / (TP + FN + 1e-8)))
    # Balance = 1 - np.sqrt((0 - FPR) ** 2 + (1 - Recall) ** 2) / np.sqrt(2)
    MCC = (TP * TN - FN * FP) / np.sqrt((TP + FN) * (TP + FP) * (FN + TN) * (FP + TN) + 1e-8)

    F_2 = 5 * Recall * Precision / (4 * Recall + Precision + 1e-8)
    # G_measure = 2 * Recall * (1 - FPR) / (Recall + (1 - FPR) + 1e-8)

    # vars(): return the dictionary object that contains the property and value of the property
    y_pred = vars()
    return {k: y_pred[k] for k in reversed(list(y_pred)) if k not in ['y_true', 'y_pred', 'TN', 'FP', 'FN', 'TP', 'FPR']}


def get_loc_data(target_datas, target_label, columns):
    """Add the attribute: loc and bug"""

    target_label = target_label.astype('int')
    target_label = np.reshape(target_label, newshape=(len(target_label), 1))
    target_datas = np.hstack((target_datas, target_label))
    df = pd.DataFrame(target_datas, columns=columns)
    df['bug'] = df.pop(df.columns[-1])
    df['loc'] = df['ld'] + df['la']
    return df


def positive_first(df: DataFrame) -> DataFrame:
    """Move the positive instances to the front of the dataset."""

    if sum(df.pred == df.bug) * 2 < len(df):
        df.pred = (df.pred == False)

    return concat([df[df.pred == True], df[df.pred == False]])


def effort_aware(df: DataFrame, EAPredict: DataFrame):
    """Calculate the effort-aware indicators"""

    EAOptimal = concat([df[df.bug == True], df[df.bug == False]])
    EAWorst = EAOptimal.iloc[::-1]

    M = len(df)
    N = sum(df.bug)
    m = threshold_index(EAPredict['loc'], 0.2)
    n = sum(EAPredict.bug.iloc[:m])
    for k, y in enumerate(EAPredict.bug):
        if y:
            break

    y = set(vars().keys())
    EA_Precision = n / m
    EA_Recall = n / N
    # EA_F1 = harmonic_mean(EA_Precision, EA_Recall)
    EA_F2 = 5 * EA_Precision * EA_Recall / np.array(4 * EA_Precision + EA_Recall + 1e-8)
    # PCI = m / M
    # IFA = k
    P_opt = norm_opt(EAPredict, EAOptimal, EAWorst)
    M = vars()

    return {k: M[k] for k in reversed(list(M)) if k not in y}


def threshold_index(loc, percent: float) -> int:
    """Returns the first subscript value within the LOC attribute value greater than sum(LOC)*percent"""

    threshold = sum(loc) * percent
    for i, x in enumerate(loc):
        threshold -= x
        if threshold < 0:
            return i + 1


def norm_opt(*args) -> float:
    """Calculate the Alberg-diagram-based effort-aware indicator"""

    predict, optimal, worst = map(alberg_auc, args)
    return 1 - (optimal - predict) / (optimal - worst)


def alberg_auc(df: DataFrame) -> float:
    """Calculate the area under curve in Alberg diagrams"""
    points = df[['loc', 'bug']].values.cumsum(axis=0)
    points = np.insert(points, 0, [0, 0], axis=0) / points[-1]
    return auc(*points.T)