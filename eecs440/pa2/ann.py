"""
The Artificial Neural Network
"""
from __future__ import division
from __future__ import print_function
import sys
if sys.version_info[0] == 3:
    xrange = range

import numpy as np
import numpy.random
import scipy


def random_weights(dimensions, range):
    lower, upper = *range
    size = upper - lower
    offset = lower
    return np.random.rand(*dimensions) * size + offset


def sigmoid(u):
    return 1 / (1 + np.exp(-u))


sigmoid_ufunc = np.vectorize(sigmoid)


class ArtificialNeuralNetwork(object):

    STARTING_WEIGHT_RANGE = [-0.1, 0.1]
    NU = 0.01

    def __init__(self,
                 gamma,
                 layer_sizes,
                 num_hidden,
                 epsilon=None,
                 max_iters=None,
                 **kwargs):
        """
        Construct an artificial neural network classifier

        @param gamma : weight decay coefficient
        @param layer_sizes:  Number of hidden layers
        @param num_hidden:  Number of hidden units in each hidden layer
        @param epsilon : cutoff for gradient descent
                         (need at least one of [epsilon, max_iters])
        @param max_iters : maximum number of iterations to run
                            gradient descent for
                            (need at least one of [epsilon, max_iters])
        """
        assert layer_sizes == 1
        assert epsilon is not None or max_iters is not None

        self._max_iters = max_iters
        self.epsilon = epsilon
        self._schema = kwargs['schema']
        self.num_hidden = num_hidden
        self.gamma = gamma
        self.weight_decay_term = 2 * self.gamma * ArtificialNeuralNetwork.NU

        num_attrs = len(self._schema.feature_names)
        self.hidden_weights = random_weights(
            [num_hidden, num_attrs],
            ArtificialNeuralNetwork.STARTING_WEIGHT_RANGE,
        )
        num_hidden = num_attrs if num_hidden == 0 else num_hidden
        self.output_weights = random_weights(
            [num_hidden],
            ArtificialNeuralNetwork.STARTING_WEIGHT_RANGE,
        )
        self._build_normalization_table()

    def fit(self, X, y, sample_weight=None):
        """
        Fit a neural network of layer_sizes * num_hidden hidden units using
        X, y.
        """
        X = np.apply_along_axis(self.normalize, 1, X.astype('float64'))
        self.means = np.mean(X, 0)
        nominal_indices = np.array(
            [i for i, v in enumerate(self._schema.nominal_values) if len(v)]
        )
        self.means[nominal_indices] = 0
        self.std_devs = np.std(X, 0)
        self.std_devs[nominal_indices] = 1
        X = np.apply_along_axis(self.standardize, 1, X)
        iterations = 0
        output_delta = np.inf
        hidden_deltas = np.array([np.inf for i in xrange(self.num_hidden)])
        while not self.stop_fitting(iterations,
                                    output_delta,
                                    hidden_deltas):
            # begin printing fanciness
            if iterations == 0:
                print("iters: {}".format(iterations), end='')
            else:
                print("\b" * len(str(iterations - 1)), end='')
                print(iterations, end='')
            sys.stdout.flush()
            iterations += 1
            for i, x in enumerate(X):
                # compute outputs of all nodes
                o_h, o_o = self.propagate(x)
                # compute deltas
                output_delta = o_o * (1 - o_o) * (o_o - y[i])
                if self.num_hidden > 0:
                    hidden_deltas = \
                        (1 - o_h) * (self.output_weights * output_delta)
                else:
                    o_h = x
                # update weights
                self.output_weights = self.output_weights - \
                    (ArtificialNeuralNetwork.NU * output_delta * o_h) - \
                    self.output_weights * self.weight_decay_term
                if self.num_hidden > 0:
                    self.hidden_weights = self.hidden_weights - \
                        (ArtificialNeuralNetwork.NU *
                         np.array([hidden_deltas[i] * x
                                   for i in xrange(len(hidden_deltas))])) - \
                        self.hidden_weights * self.weight_decay_term
        print()  # clear the newline we left over

    def propagate(self, x):
        if self.num_hidden > 0:
            hidden_outputs = sigmoid_ufunc(np.dot(self.hidden_weights, x))
        else:
            hidden_outputs = x
        output_output = sigmoid(np.dot(self.output_weights, hidden_outputs))
        return (hidden_outputs, output_output)

    def stop_fitting(self, num_iters, output_delta, hidden_deltas):
        if self._max_iters is not None:
            return num_iters >= self._max_iters
        return output_delta < self.epsilon and \
            np.all(hidden_deltas < self.epsilon)

    def predict(self, X):
        """ Predict -1/1 output """
        activations = self.predict_proba(X)
        activations[activations >= 0.5] = 1
        activations[activations < 0.5] = -1
        return activations

    def predict_proba(self, X):
        """ Predict probabilistic output """
        X = np.apply_along_axis(self.normalize, 1, X.astype('float64'))
        X = np.apply_along_axis(self.standardize, 1, X)
        return np.array([self.propagate(x)[1] for x in X])

    def _build_normalization_table(self):
        self.normalization_table = {}
        for i, values in enumerate(self._schema.nominal_values):
            if not values:
                continue
            sorted_values = sorted([int(s) for s in values])
            self.normalization_table[i] = \
                dict((float(orig), norm)
                     for norm, orig in enumerate(sorted_values, start=1))

    def normalize(self, X):
        return np.array([self._normalize_input(X, i) for i in xrange(len(X))])

    def _normalize_input(self, X, i):
        return self.normalization_table.get(i, {}).get(X[i], X[i])

    def standardize(self, X):
        return (X - self.means) / self.std_devs
