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
        classifier = CLASSIFIERS[self.algorithm]
        self.classifiers = [classifiers(**self.params) for _ in self.bags]
        for classifier in self.classifiers:
            bag = ((len(X) + 1) * np.random_sample(len(X))).as_type('int')
            classifier.fit(X[bag], y[bag])

    def predict(self, X):
        preds = np.array(
            [c.predict(X) for c in self.classifiers],
        )
        pass

    def predict_proba(self, X):
        pass
