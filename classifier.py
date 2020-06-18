#!/usr/bin/env python
# encoding: utf-8

from indicator import get_metrics
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression


def train_LR(train_datas, train_labels, test_datas, test_labels):
    """Logistic regression classifier"""

    model = LogisticRegression(random_state=0)
    print("=========================%s=====================" % model.__class__.__name__)
    model.fit(train_datas, train_labels)

    indicators = evaluate(model, test_datas, test_labels)
    return model, indicators


def evaluate(model, datas, labels, threadshold=0.5):
    """Model prediction"""

    preds = model.predict(datas)
    pred_class = [0 if i < threadshold else 1 for i in preds]

    # obtain indicator values
    all_indicators = get_metrics(labels, pred_class)

    # calculate auc
    auc = roc_auc_score(y_true=labels, y_score=preds)

    all_indicators['auc'] = auc

    return all_indicators


