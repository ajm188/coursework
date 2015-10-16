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
        self.m = 0
        self.y_prob = None
        self.pos_probs = defaultdict(lambda *args: defaultdict(int))
        self.neg_probs = defaultdict(lambda *args: defaultdict(int))
        self.discrete_mappings = {}

    def fit(self, X, y):
        largest_feature_value = max(
            [max([int(v) for v in vs] + [0])
             for vs in self._schema.nominal_values]
        )
        largest_feature_value = max(largest_feature_value, self.NUM_BINS)
        X_T = X.T

        pos_i, neg_i = np.where(y == 1)[0], np.where(y == - 1)[0]
        pos_x, neg_x = X[pos_i], X[neg_i]

        pos_x_cols, neg_x_cols = pos_x.T, neg_x.T

        all_pos_probs, all_neg_probs = [], []
        print('Feature: 0', end='')
        for i, _ in enumerate(self._schema.feature_names):
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
                pos_probs, neg_probs, lb, bs = self.find_continuous_probs(
                    [pos_feature, neg_feature],
                    [np.min(X_T[i]), np.max(X_T[i])],
                    [pos_x, neg_x],
                )
                self.discrete_mappings[i] = (lb, bs)

            pos, neg = [], []
            for j in xrange(largest_feature_value + 1):
                for probs_dict, probs_list in [(pos_probs, pos),
                                               (neg_probs, neg)]:
                    probs_list.append(probs_dict[j])
            pos_probs = np.array(pos)
            neg_probs = np.array(neg)

            all_pos_probs.append(pos_probs)
            all_neg_probs.append(neg_probs)
        self.pos_probs = np.array(all_pos_probs)
        self.neg_probs = np.array(all_neg_probs)

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

        pos_probs, neg_probs = defaultdict(int), defaultdict(int)
        prev_boundary = lb
        for i in xrange(1, self.NUM_BINS + 1):
            boundary = lb + (bin_size * i)

            pos_matches = np.sum(
                (prev_boundary < pos_feature) & (pos_feature <= boundary),
            )
            neg_matches = np.sum(
                (prev_boundary < neg_feature) & (neg_feature <= boundary),
            )

            laplace = self.m * self.NUM_BINS
            pos_probs[i] = (pos_matches + laplace) / (len(pos_x) + self.m)
            neg_probs[i] = (neg_matches + laplace) / (len(neg_x) + self.m)

            prev_boundary = boundary

        return pos_probs, neg_probs, lb, bin_size

    def predict(self, X):
        rv = np.arange(len(X))
        for i, _ in enumerate(self._schema.feature_names):
            if not self._schema.is_nominal(i):
                lb, bin_size = self.discrete_mappings[i]
                X[rv, [i]] = ((X[rv, [i]] - lb) / bin_size) + 1
        X = X.astype('int')
        joint_probs = np.apply_along_axis(
            self.joint_probs,
            1,
            X,
            np.arange(len(self._schema.feature_names)),
            [self.neg_probs, self.pos_probs],
            [1 - self.y_prob, self.y_prob],
        )
        return np.apply_along_axis(
            lambda probs: np.argmax(probs) * 2 - 1,
            1,
            joint_probs,
        )

    def predict_proba(self, X):
        joint_probs = np.apply_along_axis(
            self.joint_probs,
            1,
            X,
            np.arange(len(self._schema.feature_names)),
            [self.pos_probs],
            [self.y_prob],
        )
        return np.apply_along_axis(
            lambda x: x[0],
            1,
            joint_probs,
        )

    def joint_probs(self, x, rv, probs, divisors):
        p = np.array([prob[rv, x] for prob in probs])
        return np.prod(p, axis=1) / divisors
