# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 15:33:05 2022

@author: bonnyaigergo

https://scikit-learn.org/stable/modules/classes.html?highlight=metrics#module-sklearn.metrics.cluster
https://saskeli.github.io/data-analysis-with-python-summer-2019/clustering.html
"""

from sklearn import metrics
# acc = (TP+TN) / (TP+FP+FN+TN)

def benchmarking(true_labels, pred_labels):
    # TODO: Please note that the AMI definition used in the paper differs from that in the sklearn python package.
    # TODO: Please modify it accordingly.
    numeval = len(gtlabels)
    ari = metrics.adjusted_rand_score(gtlabels[:numeval], labels[:numeval])
    ami = metrics.adjusted_mutual_info_score(gtlabels[:numeval], labels[:numeval])
    nmi = metrics.normalized_mutual_info_score(gtlabels[:numeval], labels[:numeval])
    acc = clustering_accuracy(gtlabels[:numeval], labels[:numeval])

    return ari, ami, nmi, acc