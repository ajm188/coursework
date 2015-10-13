"""
The Naive Bayes Classifier
"""
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy


class NaiveBayes(object):

    def __init__(self, alpha=0, schema=None):
        """
        Constructs a Naive Bayes classifier

        @param m : Smoothing parameter (0 for no smoothing)
        """
        self._schema = schema

    def fit(self, X, y):
        pass  # add code here

    def predict(self, X):
        pass  # add code here

    def predict_proba(self, X):
        pass  # add code here
