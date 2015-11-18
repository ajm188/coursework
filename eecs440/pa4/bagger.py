"""
An implementation of bagging as a wrapper for a classifier
"""
import numpy as np
import numpy.random
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


class Bagger(object):

    def __init__(self, algorithm, iters, **params):
        """
        Boosting wrapper for a classification algorithm

        @param algorithm : Which algorithm to use
                           (dtree, ann, linear_svm, nbayes,
                           or logistic_regression)
        @param iters : How many iterations of bagging to do
        @param params : Parameters for the classification algorithm
        """
        self.algorithm = algorithm
        self.bags = iters
        self.params = params

    def fit(self, X, y):
        cls = CLASSIFIERS[self.algorithm]
        self.classifiers = [cls(**self.params) for _ in range(self.bags)]
        for classifier in self.classifiers:
            bag = (len(X) * np.random.random_sample(len(X))).astype('int')
            X_bag = np.array([X[i] for i in bag])
            y_bag = np.array([y[i] for i in bag])
            classifier.fit(X_bag, y_bag)

    def predict(self, X):
        preds = np.array(
            [c.predict(X) for c in self.classifiers],
        )
        preds = np.sum(preds, axis=0)
        preds[preds >=0] = 1
        preds[preds < 0] = -1
        return preds

    def predict_proba(self, X):
        probs = np.array(
            [c.predict_proba(X) for c in self.classifiers],
        )
        probs = np.average(probs, axis=0)
        return probs
