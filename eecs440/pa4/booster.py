"""
An implementation of boosting as a wrapper for a classifier
"""
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
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def predict_proba(self, X):
        pass
