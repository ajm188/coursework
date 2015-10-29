"""
The Naive Bayes Classifier
"""
from __future__ import division
from __future__ import print_function

import sys

import numpy as np
import scipy

import stats
from folds import get_folds


class NaiveBayes(object):

    NUM_BINS = 10

    def __init__(self, m=None, schema=None):
        """
        Initialize a NaiveBayes classifier with m and schema.
        m is generally ignored, in favor of doing parameter tuning.
        """
        self.m = m
        self.schema = schema
        self.discretizations = {}

    def fit(self, X, y):
        """
        Fit a classifier to the training data X and y.
        Does internal cross-fold validation to tune the m parameter for doing
        m-estimates.
        """
        self._enable_discretization(X)
        X = self.discretize(X).astype('int')
        m = self.tune(X, y, [0, 0.001, 0.01, 0.1, 1, 10, 100])
        print('Selected {} for m'.format(m))
        self.m = m
        self._fit(X, y)

    def _fit(self, X, y):
        """
        Does the actual fitting. Assumes the data has already been transformed
        from continuous attributes into discrete ones.

        Performs m-estimate fitting.
        """
        pos_y, neg_y = np.where(y == 1)[0], np.where(y == -1)[0]
        pos_X, neg_X = X[pos_y], X[neg_y]

        pos_probs, neg_probs = [], []
        max_feat_vec_len = 0

        for i, _ in enumerate(self.schema.feature_names):
            feat_vec_len = max(pos_X[:, i]) + 1
            m = self.m
            p = 1 / (feat_vec_len - 1)
            max_feat_vec_len = max(feat_vec_len, max_feat_vec_len)
            pos_probs.append((np.bincount(pos_X[:, i]) + (m * p)) /
                             (len(pos_y) + m))
            neg_probs.append((np.bincount(neg_X[:, i]) + (m * p)) /
                             (len(neg_y) + m))

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

    def tune(self, X, y, m_range):
        """
        Performs parameter tuning on m. Uses internal cross-fold validation.
        Assumes the training data has already been transformed into discrete
        values, since this operation is expensive.

        Chooses the m which maximizes AUC.
        """
        folds = get_folds(X, y, 5)
        AUCs = []
        for m in m_range:
            sm = stats.StatisticsManager()
            for train_X, train_y, test_X, test_y in folds:
                m_classifier = NaiveBayes(m=m, schema=self.schema)
                m_classifier._fit(train_X, train_y)
                preds = m_classifier._predict(test_X)
                probs = m_classifier._predict_proba(test_X)
                sm.add_fold(test_y, preds, probs, 0)
            A = sm.get_statistic('auc', pooled=True)
            AUCs.append(A)
        return m_range[np.argmax(AUCs)]

    def predict(self, X):
        """
        Make predictions on X after converting X into discretely-valued
        attributes.
        """
        X = self.discretize(X).astype('int')
        return self._predict(X)

    def _predict(self, X):
        """
        The actual prediction code. Assumes that X is discretely-valued.
        Returns 1 if p(Y=1,x) > p(Y=-1,x) and -1 otherwise for each x in X.
        """
        pos, neg = [self.predict_p(X, p)
                    for p in [self.pos_probs, self.neg_probs]]
        pos_probs = pos * self.y_prob
        neg_probs = neg * (1 - self.y_prob)
        ones = np.ones(len(pos_probs))
        return np.where(pos_probs > neg_probs, ones, ones - 2)

    def predict_proba(self, X):
        """
        Return p(y=1,x) for each x in X.
        """
        X = self.discretize(X).astype('int')
        return self._predict_proba(X)

    def _predict_proba(self, X):
        """
        Returns p(y=1,x) for each x in X. Assumes X is discretely-valued.
        """
        return self.predict_p(X, self.pos_probs) * self.y_prob

    def predict_p(self, X, probs):
        """
        Returns p(y=y',x) for each p' in probs, x in X.
        """
        row_vec = np.arange(len(probs))
        return np.array([np.product(probs[row_vec, x]) for x in X])

    def _enable_discretization(self, X):
        """
        Sets up an instance variable that will be used later to transform
        continuous attributes into discretely-valued attributes.
        """
        for i, _ in enumerate(self.schema.feature_names):
            if not self.schema.is_nominal(i):
                self.discretizations[i] = []
                lb, ub = np.min(X[:, i]), np.max(X[:, i])
                bin_size = (ub - lb) / self.NUM_BINS

                ub = lb  # array really hold the upper bound
                for j in range(self.NUM_BINS + 1):
                    self.discretizations[i].append(ub)
                    ub += bin_size

    def discretize(self, X):
        """
        Transforms X from continuous attributes into discretely-valued
        attributes.
        """
        D = np.empty_like(X)
        for i, _ in enumerate(self.schema.feature_names):
            X_C = X[:, i]
            if not self.schema.is_nominal(i):
                d = self.discretizations[i]
                for j, r in enumerate(zip([-np.inf] + d, d + [np.inf])):
                    lb, ub = r
                    matches = np.where((X_C >= lb) & (X_C < ub))[0]
                    D[:, i][matches] = j
            else:
                D[:, i] = X_C
        return D
