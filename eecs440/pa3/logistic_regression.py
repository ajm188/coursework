# -*- coding: utf8 -*-
"""
The Logistic Regression Classifier
"""
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.linalg
import numpy.random
import scipy
import scipy.optimize

import stats
from folds import get_folds


def sigmoid(x):
    return (np.exp(-x) + 1) ** (-1)


def objective_func(w_and_b, X, y, _lambda):
    w, b = w_and_b[0:-1], w_and_b[-1]
    s = sigmoid(y * (np.dot(X, w) + b))
    return _lambda * 0.5 * (np.linalg.norm(w) ** 2) + np.sum(np.log(s ** (-1)))


def gradient(w_and_b, X, y, _lambda):
    w, b = w_and_b[0:-1], w_and_b[-1]
    s = sigmoid((-y) * (np.dot(X, w) + b))
    del_common = (-y) * s
    del_w = (_lambda * w) + np.sum(del_common[:, np.newaxis] * X, axis=0)
    del_b = np.sum(del_common)
    return np.concatenate([del_w, [del_b]])


def random_weights(dimensions, r):
    lower, upper = r
    size = upper - lower
    offset = lower
    return np.random.rand(*dimensions) * size + offset


class LogisticRegression(object):

    def __init__(self, schema=None, **kwargs):
        """
        Constructs a logistic regression classifier

        @param lambda : Regularisation constant parameter
        """
        self.schema = schema
        self.nominals = {}
        self._lambda = kwargs.pop('lambda')

    def fit(self, X, y):
        self._lambda = self.tune(X, y, [0, 0.001, 0.01, 0.1, 1, 10, 100])
        print('Chose {} for λ'.format(self._lambda))
        self._fit(X, y)

    def _fit(self, X, y):
        self._enable_unnominalization(X)
        X = self.unnominalize(X)

        self.means = np.mean(X, axis=0)
        self.stddevs = np.std(X, axis=0)
        X = self.normalize(X)

        res = scipy.optimize.minimize(
            objective_func,
            np.zeros(len(X[0]) + 1),
            method='BFGS',
            jac=gradient,
            args=(X, y, self._lambda),
            options={
                'disp': True,
                'maxiters': 2000,
            },
        )

        self.w, self.b = res.x[0:-1], res.x[-1]

    def tune(self, X, y, lambda_range):
        folds = get_folds(X, y, 5)
        AUCs = []
        for _lambda in lambda_range:
            print(_lambda)
            sm = stats.StatisticsManager()
            for train_X, train_y, test_X, test_y in folds:
                kwargs = {'schema': self.schema, 'lambda': _lambda}
                classifier = LogisticRegression(**kwargs)
                classifier._fit(train_X, train_y)
                preds = classifier.predict(test_X)
                probs = classifier.predict_proba(test_X)
                sm.add_fold(test_y, preds, probs, 0)
            A = sm.get_statistic('auc', pooled=True)
            AUCs.append(A)
        return lambda_range[np.argmax(AUCs)]

    def normalize(self, X):
        return (X - self.means) / self.stddevs

    def _enable_unnominalization(self, X):
        for i, _ in enumerate(self.schema.feature_names):
            if self.schema.is_nominal(i):
                self.nominals[i] = \
                    dict([(float(v), j)
                          for j, v in enumerate(self.schema.nominal_values[i])])

    def unnominalize(self, X):
        D = np.empty_like(X)
        for i, _ in enumerate(self.schema.feature_names):
            X_C = X[:, i]
            if self.schema.is_nominal(i):
                nom = self.nominals[i]
                for n, c in nom.iteritems():
                    D[:, i][np.where(X_C == n)[0]] = c
            else:
                D[:, i] = X_C
        return D

    def predict(self, X):
        X = self.normalize(self.unnominalize(X))
        predictions = np.dot(X, self.w) + self.b
        predictions[np.where(predictions > 0)[0]] = 1
        predictions[np.where(predictions < 0)[0]] = -1
        return predictions

    def predict_proba(self, X):
        X = self.normalize(self.unnominalize(X))
        dot_prods = -(np.dot(X, self.w[:, np.newaxis]) + self.b)
        frac = ((np.exp(dot_prods) + 1) ** (-1)).reshape(dot_prods.shape[0],)
        frac[np.where(frac == np.inf)[0]] = 1
        return frac
