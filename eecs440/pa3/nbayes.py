"""
The Naive Bayes Classifier
"""
from __future__ import division
from __future__ import print_function

from collections import defaultdict

import numpy as np
import scipy

from range_dict import RangeDict


class NaiveBayes(object):

    NUM_BINS = 10

    def __init__(self, alpha=0, schema=None, m=None):
        """
        Constructs a Naive Bayes classifier

        @param m : Smoothing parameter (0 for no smoothing)
        """
        self._schema = schema
        self.y_prob = None
        self.pos_probs = defaultdict(lambda *args: defaultdict(int))
        self.neg_probs = defaultdict(lambda *args: defaultdict(int))

    def fit(self, X, y):
        X_T = X.T

        pos_i, neg_i = np.where(y == 1)[0], np.where(y == - 1)[0]
        pos_x, neg_x = X[pos_i], X[neg_i]

        pos_x_cols, neg_x_cols = pos_x.T, neg_x.T

        for i in xrange(len(self._schema.feature_names)):
            pos_feature, neg_feature = pos_x_cols[i], neg_x_cols[i]

            if self._schema.is_nominal(i):
                # handle nominal attributes
                pos_probs, neg_probs = defaultdict(int), defaultdict(int)
                for v in self._schema.nominal_values[i]:
                    v = int(v)  # why do you do this to me, template code?
                    pos_matches = np.where(pos_feature == v)[0]
                    neg_matches = np.where(neg_feature == v)[0]

                    pos_probs[v] = len(pos_matches) / len(pos_x)
                    neg_probs[v] = len(neg_matches) / len(neg_x)
            else:
                # discretize continuous attributes
                lb = np.min(X_T[i])
                ub = np.max(X_T[i])
                bin_size = (ub - lb) / self.NUM_BINS

                pos_probs, neg_probs = RangeDict(), RangeDict()
                prev_boundary = -np.inf
                for i in xrange(1, NUM_BINS + 1):
                    boundary = lb + (bin_size * i)

                    pos_matches = np.where(pos_x_feature > prev_boundary & pos_x_feature <= boundary)[0]
                    neg_matches = np.where(neg_x_feature > prev_boundary & neg_x_feature <= boundary)[0]

                    pos_probs[(prev_boundary, boundary)] = len(pos_matches) / len(pos_x)
                    neg_probs[(prev_boundary, boundary)] = len(neg_matches) / len(neg_x)

                    prev_boundary = boundary

            self.pos_probs[i] = pos_probs
            self.neg_probs[i] = neg_probs

        self.y_prob = len(pos_i) / len(y)

    def predict(self, X):
        predictions = np.apply_along_axis(
            lambda x: np.argmax(
                self.compute_running_probs(
                    x,
                    [self.neg_probs, self.pos_probs],
                ) / [1 - self.y_prob, self.y_prob]
            ) * 2 - 1,
            1,
            X,
        )
        return predictions

    def predict_proba(self, X):
        probs = np.apply_along_axis(
            self.compute_running_probs,
            1,
            X,
            [self.pos_probs],
        ) / self.y_prob
        return probs.reshape(len(X),)

    def compute_running_probs(self, x, probs):
        running_probs = np.ones(len(probs))
        for i in xrange(len(x)):
            running_probs = running_probs * [p[i][x[i]] for p in probs]
        return running_probs
