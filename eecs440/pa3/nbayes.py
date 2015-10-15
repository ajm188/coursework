"""
The Naive Bayes Classifier
"""
from __future__ import division
from __future__ import print_function

import sys
from collections import defaultdict

import numpy as np
import scipy

from range_dict import RangeDict


def fancy_print(i):
    print('\b' * len(str(abs(i - 1))), end='')
    print(i, end='')
    sys.stdout.flush()


class NaiveBayes(object):

    NUM_BINS = 10

    def __init__(self, alpha=0, schema=None, m=0):
        """
        Constructs a Naive Bayes classifier

        @param m : Smoothing parameter (0 for no smoothing)
        """
        self._schema = schema
        self.m = m
        self.y_prob = None
        self.pos_probs = defaultdict(lambda *args: defaultdict(int))
        self.neg_probs = defaultdict(lambda *args: defaultdict(int))

    def fit(self, X, y):
        X_T = X.T

        pos_i, neg_i = np.where(y == 1)[0], np.where(y == - 1)[0]
        pos_x, neg_x = X[pos_i], X[neg_i]

        pos_x_cols, neg_x_cols = pos_x.T, neg_x.T

        print('Feature: 0', end='')
        for i in xrange(len(self._schema.feature_names)):
            fancy_print(i)
            pos_feature, neg_feature = pos_x_cols[i], neg_x_cols[i]

            if self._schema.is_nominal(i):
                # handle nominal attributes
                pos_probs, neg_probs = self.find_nominal_probs(
                    [pos_feature, neg_feature],
                    self._schema.nominal_values[i],
                    [pos_x, neg_x],
                )
            else:
                # discretize continuous attributes
                pos_probs, neg_probs = self.find_continuous_probs(
                    [pos_feature, neg_feature],
                    [np.min(X_T[i]), np.max(X_T[i])],
                    [pos_x, neg_x],
                )

            self.pos_probs[i] = pos_probs
            self.neg_probs[i] = neg_probs

        self.y_prob = len(pos_i) / len(y)
        print()

    def find_nominal_probs(self, feature, nominal_values, x):
        pos_feature, neg_feature = feature
        pos_x, neg_x = x

        pos_probs, neg_probs = defaultdict(int), defaultdict(int)
        for v in nominal_values:
            v = int(v)  # why do you do this to me, template code?
            pos_matches = np.where(pos_feature == v)[0]
            neg_matches = np.where(neg_feature == v)[0]

            laplace = self.m * len(nominal_values)
            pos_probs[v] = (len(pos_matches) + laplace) / (len(pos_x) + self.m)
            neg_probs[v] = (len(neg_matches) + laplace) / (len(neg_x) + self.m)

        return pos_probs, neg_probs

    def find_continuous_probs(self, feature, feature_range, x):
        pos_feature, neg_feature = feature
        pos_x, neg_x = x

        lb, ub = feature_range
        bin_size = (ub - lb) / self.NUM_BINS

        pos_probs, neg_probs = RangeDict(), RangeDict()
        prev_boundary = -np.inf
        for i in xrange(1, self.NUM_BINS + 1):
            boundary = lb + (bin_size * i)

            pos_matches = np.sum((prev_boundary < pos_feature) & (pos_feature <= boundary))
            neg_matches = np.sum((prev_boundary < neg_feature) & (neg_feature <= boundary))

            laplace = self.m + self.NUM_BINS
            pos_probs[(prev_boundary, boundary)] = \
                (pos_matches + laplace) / (len(pos_x) + self.m)
            neg_probs[(prev_boundary, boundary)] = \
                (neg_matches + self.NUM_BINS) / (len(neg_x) + self.m)

            prev_boundary = boundary

        return pos_probs, neg_probs

    def predict(self, X):
        predictions = []
        print('Predicting example: 0', end='')
        for i, x in enumerate(X):
            fancy_print(i)
            probs = self.compute_running_probs(
                x,
                [self.neg_probs, self.pos_probs],
            )
            predictions.append(np.argmax(probs) * 2 - 1)
        print()
        return np.array(predictions)

    def predict_proba(self, X):
        pred_probs = []
        print('Predicting prob for example: 0', end='')
        for i, x in enumerate(X):
            fancy_print(i)
            prob = self.compute_running_probs(x, [self.pos_probs])[0]
            pred_probs.append(prob / self.y_prob)
        print()
        return np.array(pred_probs)

    def compute_running_probs(self, x, probs):
        running_probs = np.ones(len(probs))
        for i in xrange(len(x)):
            running_probs = running_probs * [p[i].get(x[i], 0) for p in probs]
        return running_probs
