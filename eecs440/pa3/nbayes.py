"""
The Naive Bayes Classifier
"""
from __future__ import division
from __future__ import print_function

import sys

import numpy as np
import scipy


def fancy_print(i):
    print('\b' * len(str(abs(i - 1))), end='')
    print(i, end='')
    sys.stdout.flush()


class NaiveBayes(object):

    NUM_BINS = 10

    def __init__(self, m=None, schema=None):
        self.schema = schema
        self.discretizations = {}

    def fit(self, X, y):
        self._enable_discretization(X)
        X = self.discretize(X).astype('int')
        pos_y, neg_y = np.where(y == 1)[0], np.where(y == -1)[0]
        pos_X, neg_X = X[pos_y], X[neg_y]

        pos_probs, neg_probs = [], []
        max_feat_vec_len = 0

        print('Training feature: 1', end='')
        for i, _ in enumerate(self.schema.feature_names):
            fancy_print(i + 1)
            feat_vec_len = max(pos_X[:, i]) + 1
            p = 1 / (feat_vec_len - 1)
            m = self.m
            max_feat_vec_len = max(feat_vec_len, max_feat_vec_len)
            pos_probs.append((np.bincount(pos_X[:, i]) + (m * p)) /
                             (len(pos_y) + m))
            neg_probs.append((np.bincount(neg_X[:, i]) + (m * p)) /
                             (len(neg_y) + m))
        print()

        # Normalize the arrays so we can safely put them in a 2darray
        self.pos_probs = np.array(
            [np.concatenate([p, np.zeros(max_feat_vec_len - len(p))])
             for p in pos_probs]
        )
        self.neg_probs = np.array(
            [np.concatenate([p, np.zeros(max_feat_vec_len - len(p))])
             for p in neg_probs]
        )

        self.y_prob = len(pos_y) / len(y)

    def predict(self, X):
        X = self.discretize(X).astype('int')
        pos, neg = [self.predict_p(X, p)
                    for p in [self.pos_probs, self.neg_probs]]
        pos_probs = pos * self.y_prob
        neg_probs = neg * (1 - self.y_prob)
        ones = np.ones(len(pos_probs))
        return np.where(pos_probs > neg_probs, ones, ones - 2)

    def predict_proba(self, X):
        X = self.discretize(X).astype('int')
        return self.predict_p(X, self.pos_probs) * self.y_prob

    def predict_p(self, X, probs):
        row_vec = np.arange(len(probs))
        return np.array([np.product(probs[row_vec, x]) for x in X])

    def _enable_discretization(self, X):
        print('Enabling discretization: 1', end='')
        for i, _ in enumerate(self.schema.feature_names):
            fancy_print(i + 1)
            if not self.schema.is_nominal(i):
                self.discretizations[i] = []
                lb, ub = np.min(X[:, i]), np.max(X[:, i])
                bin_size = (ub - lb) / self.NUM_BINS

                ub = lb  # array really hold the upper bound
                for j in range(self.NUM_BINS + 1):
                    self.discretizations[i].append(ub)
                    ub += bin_size
        print()

    def discretize(self, X):
        D = np.empty_like(X)
        print('Discretizing feature: 1', end='')
        for i, _ in enumerate(self.schema.feature_names):
            fancy_print(i + 1)
            X_C = X[:, i]
            if not self.schema.is_nominal(i):
                d = self.discretizations[i]
                for j, r in enumerate(zip([-np.inf] + d, d + [np.inf])):
                    lb, ub = r
                    matches = np.where((X_C >= lb) & (X_C < ub))[0]
                    D[:, i][matches] = j
            else:
                D[:, i] = X_C
        print()
        return D
