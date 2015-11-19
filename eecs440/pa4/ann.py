"""
The Artificial Neural Network
"""
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random
import scipy


def random_weights(dimensions, r):
    """
    Return a NumPy array with the specified :dimensions:, where each element
    is a random number within :range:.
    """
    lower, upper = r
    size = upper - lower
    offset = lower
    return np.random.rand(*dimensions) * size + offset


def sigmoid(u):
    """Returns the value of the sigmoid function for the argument :u:."""
    return 1 / (1 + np.exp(-u))


# Get a vectorized version of sigmoid so we can broadcast it to NumPy matrices.
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
        # First, normalize the traning set, non-destructively.
        X = np.apply_along_axis(self.normalize, 1, X.astype('float64'))
        # Next, compute the means and std_devs
        self.means = np.mean(X, 0)
        nominal_indices = np.array(
            [i for i, v in enumerate(self._schema.nominal_values) if len(v)]
        )
        # Set mean = 0 and std_dev = 1 for nominal attributes, so that
        # standardizing later doesn't actually effect the nominal attributes
        self.means[nominal_indices] = 0
        self.std_devs = np.std(X, 0)
        self.std_devs[nominal_indices] = 1
        X = self.standardize(X)
        iterations = 0
        output_delta = np.inf
        hidden_deltas = np.array([np.inf for i in range(self.num_hidden)])
        while not self.stop_fitting(iterations,
                                    output_delta,
                                    hidden_deltas):
            """
            This printing was super cool, but once I added parallelization,
            it got really ugly. So, I'll just leave this commented out.
            # begin printing fanciness
            if iterations == 0:
                print("iters: {}".format(iterations), end='')
            else:
                print("\b" * len(str(iterations - 1)), end='')
                print(iterations, end='')
            sys.stdout.flush()
            """
            iterations += 1
            # Stochastic gradient descent.
            for i, x in enumerate(X):
                # compute outputs of all nodes
                o_h, o_o = self.propagate(x)
                # compute deltas
                output_delta = o_o * (1 - o_o) * (o_o - y[i])
                if sample_weight is not None:
                    output_delta = output_delta * sample_weight[i]
                # Ignore the hidden layer if there isn't one
                if self.num_hidden > 0:
                    hidden_deltas = \
                        (1 - o_h) * (self.output_weights * output_delta)
                else:
                    o_h = x
                # update weights
                self.output_weights = self.output_weights - \
                    (ArtificialNeuralNetwork.NU * output_delta * o_h) - \
                    self.output_weights * self.weight_decay_term
                # Igore the hidden layer if there isn't one
                if self.num_hidden > 0:
                    self.hidden_weights = self.hidden_weights - \
                        (ArtificialNeuralNetwork.NU *
                         np.array([hidden_deltas[i] * x
                                   for i in range(len(hidden_deltas))])) - \
                        self.hidden_weights * self.weight_decay_term
        print()  # clear the newline we left over

    def propagate(self, x):
        """
        Propagate an example :x: through the network.
        Ignores the hidden layer if there isn't one.
        Returns the tuple of hidden_layer outputs, and output_layer output.
        """
        if self.num_hidden > 0:
            hidden_outputs = sigmoid_ufunc(np.dot(self.hidden_weights, x))
        else:
            hidden_outputs = x
        output_output = sigmoid(np.dot(self.output_weights, hidden_outputs))
        return (hidden_outputs, output_output)

    def stop_fitting(self, num_iters, output_delta, hidden_deltas):
        """
        Returns whether the ANN should stop doing gradient descent.

        This happens if the ANN was initialized with a maximum number of
        iterations to perform, and that number has been reached, or if the
        dL/dw for each node in the network is below some epsilon threshold.
        """
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
        X = self.standardize(X)
        return np.array([self.propagate(x)[1] for x in X])

    def _build_normalization_table(self):
        """
        Build up the nested dictionary needed to normalize nominal attributes
        later.

        The index of each nominal attribute gets mapped to a dictionary.
        In this nested dictionary, each possible value for the nominal feature
        gets mapped to a value in 1..k, where k is the number of possible
        values.

        Had to do some funky casting here, because the template code you give
        us has inconsistencies between the types loaded from the actual .mat
        files and the types implied by the schema.nominal_values lists. But,
        oh well, I made it work.
        """
        self.normalization_table = {}
        for i, values in enumerate(self._schema.nominal_values):
            if not values:
                continue
            sorted_values = sorted([int(s) for s in values])
            self.normalization_table[i] = \
                dict((float(orig), norm)
                     for norm, orig in enumerate(sorted_values, start=1))

    def normalize(self, X):
        """
        Normalize a set of examples, according to the schema.

        Returns a new nparray, so this operation is non-destructive.
        """
        return np.array([self._normalize_input(X, i) for i in range(len(X))])

    def _normalize_input(self, X, i):
        """
        Normalizes a single input. Uses maximum cleverness to ensure that
        only nominal attributes are affected by this operation.

        Basically, if we don't have a mapping which says what a feature value
        should be changed to, the feature value remains unchanged.
        """
        return self.normalization_table.get(i, {}).get(X[i], X[i])

    def standardize(self, X):
        """
        Standardizes a set of examples. Modifies the examples such that the
        mean feature value is 0 and the standard deviation of the feature is 1.
        """
        return (X - self.means) / self.std_devs
