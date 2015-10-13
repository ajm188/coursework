"""
The Naive Bayes Classifier
"""
from __future__ import division
from __future__ import print_function

from collections import defaultdict

import numpy as np
import scipy


class NaiveBayes(object):

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
        self.y_prob = len(y[y == 1]) / len(y)
        positive_indices = np.where(y == 1)[0]
        negative_indices = np.where(y == -1)[0]
        positive_examples = X[positive_indices]
        negative_examples = X[negative_indices]
        pos_ex_cols = positive_examples.T
        neg_ex_cols = negative_examples.T
        for i in xrange(len(self._schema.feature_names)):
            pos_feature = pos_ex_cols[i]
            neg_feature = neg_ex_cols[i]
            # I'll deal with continuous attrs later
            for v in self._schema.nominal_values[i]:
                _v = int(v)
                pos_matches = np.where(pos_feature == _v)[0]
                neg_matches = np.where(neg_feature == _v)[0]
                self.pos_probs[i][_v] = \
                    len(pos_matches) / len(positive_examples)
                self.neg_probs[i][_v] = \
                    len(neg_matches) / len(negative_examples)

    def predict(self, X):
        predictions = []
        for x in X:
            running_probs = np.ones(2)
            for i in xrange(len(x)):
                running_probs = running_probs * [self.neg_probs[i][x[i]],
                                                 self.pos_probs[i][x[i]]]
            probs = running_probs / [1 - self.y_prob, self.y_prob]
            predictions.append((np.argmax(running_probs) * 2 - 1))
        return np.array(predictions)

    def predict_proba(self, X):
        pass  # add code here
