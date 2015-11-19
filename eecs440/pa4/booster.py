"""
An implementation of boosting as a wrapper for a classifier
"""
from __future__ import division

import numpy as np
import scipy

from ann import ArtificialNeuralNetwork
from dtree import DecisionTree
from logistic_regression import LogisticRegression
from nbayes import NaiveBayes

CLASSIFIERS = {
    'ann': ArtificialNeuralNetwork,
    'dtree': DecisionTree,
    'logistic_regression': LogisticRegression,
    'nbayes': NaiveBayes,
}


class Booster(object):

    def __init__(self, algorithm, iters, **params):
        """
        Boosting wrapper for a classification algorithm

        @param algorithm : Which algorithm to use
                            (dtree, ann, linear_svm, nbayes,
                            or logistic_regression)
        @param iters : How many iterations of boosting to do
        @param params : Parameters for the classification algorithm
        """
        self.algorithm = algorithm
        self.iters = iters
        self.params = params

    def fit(self, X, y):
        cls = CLASSIFIERS[self.algorithm]
        self.classifiers = [cls(**self.params) for _ in range(self.iters)]
        sample_weight = np.ones_like(y) / len(y)

        self.classifier_weights = {}
        for i, classifier in enumerate(self.classifiers):
            classifier.fit(X, y, sample_weight=sample_weight)
            preds = classifier.predict(X)
            indicators = (preds != y).astype('int')
            error = np.sum(indicators * sample_weight)
            if error == 1:
                self.classifier_weights[i] = 0
            else:
                self.classifier_weights[i] = 0.5 * np.log2((1 - error) / error)
            if error == 0 or error >= 0.5:
                break

            correct = np.where(preds == y)[0]
            incorrect = np.where(preds != y)[0]
            sample_weight[correct] = sample_weight[correct] * error
            sample_weight[incorrect] = sample_weight[incorrect] / error

        self.classifier_weights = \
            [self.classifier_weights[i]
             for i in sorted(self.classifier_weights.keys())]

    def predict(self, X):
        denom = sum(self.classifier_weights)
        weighted_preds = np.array(
            [c.predict(X) * w / denom
             for w, c in zip(self.classifier_weights, self.classifiers)],
        )
        try:
            preds = (np.sum(weighted_preds, axis=0) * 2).astype('int')
            preds[preds < 0] = -1
            preds[preds >= 0] = 1
        except:
            import pdb; pdb.set_trace()
        return preds

    def predict_proba(self, X):
        denom = sum(self.classifier_weights)
        weighted_probs = np.array(
            [c.predict_proba(X) * w / denom
             for w, c in zip(self.classifier_weights, self.classifiers)],
        )
        return np.sum(weighted_probs, axis=0)
