"""
The Decision Tree Classifier
"""
from __future__ import print_function

import numpy as np
import numpy.random
import scipy
import scipy.stats


def continuous_counts(X, value):
    lhs = len(np.where(X <= value)[0])
    return np.array([lhs, len(X) - lhs])


def discrete_counts(X):
    return np.bincount(X.astype(int))


def H(Y, given=None, continuous_value=None):
    if given is not None:
        X = given
        X_counts = discrete_counts(X) if continuous_value is None else \
            continuous_counts(X, continuous_value)
        X_probs = X_counts / float(len(X))
        if continuous_value is not None:
            cond_entropies = np.array(
                [
                    H(Y[np.where(X <= continuous_value)]),
                    H(Y[np.where(X > continuous_value)]),
                ],
            )
        else:
            cond_entropies = np.array(
                [H(Y[np.where(X == i)[0]]) for i in range(len(X_counts))],
            )
        return -np.dot(X_probs, cond_entropies)

    if not len(Y):
        return 0
    counts = discrete_counts(Y) if continuous_value is None else \
        continuous_counts(Y, continuous_value)
    probabilities = counts / float(len(Y))
    log_probabilities = np.log2(probabilities)
    log_probabilities[log_probabilities == -np.inf] = 0
    return -np.dot(probabilities, log_probabilities)


def IG(X, y, continuous_value=None):
    return H(y) - H(y, given=X, continuous_value=continuous_value)


def gain_ratio(X, y, continuous_value=None):
    H_X = H(X, continuous_value=continuous_value)
    if not H_X:
        return 0
    return IG(X, y, continuous_value=continuous_value) / H_X


class DecisionTree(object):

    class Node(object):

        def __init__(self, feature=None, label=None, parent=None, split=None):
            self.feature = feature
            self.parent = parent
            self.label = label
            self.split = split
            if self.split:
                self.splits = []
            self.test = {}

        def add_test(self, v, n):
            if self.split:
                if len(self.splits) > 2:
                    message = "splitting is binary. \
                               there can only be 2 children"
                    raise Exception(message)
                # Based on the way partition is defined (below), the left
                # partition should always be added first.
                self.splits.append(n)
                # also, stick the node in self.test so "depth" and "size"
                # still work correctly. gross, I know, but hey it's getting
                # late
                self.test[len(self.splits)] = n
            else:
                if v in self.test:
                    raise Exception("can't set the same value twice")
                self.test[v] = n

        def predict(self, x):
            if self.label:
                return self.label
            elif self.split:
                # Since the left partition is always added first, if
                # x[feature] is less than the split, take the first node,
                # otherwise, take the second
                splits_index = 0 if x[self.feature] <= self.split else 1
                return self.splits[splits_index]
            else:
                try:
                    return self.test[x[self.feature]].predict(x)
                except:
                    # In the event we don't have a node for the given
                    # attribute of x (i.e. the partition would have been empty
                    # when constructing the decision tree) just pick between
                    # 1/-1 with probability 1/2. Note that this is the same
                    # behavior as if DecisionTree._fit had been called on the
                    # empty partition - it was just easier to implement it this
                    # way with the duplicated code
                    return (-1) ** np.random.random_integers(0, 1)

        def size(self):
            if not self.test:
                return 1
            return sum([child.size() for child in self.test.values()])

        def depth(self):
            if not self.test:
                return 1
            return 1 + max(
                map(
                    lambda child: child.depth(),
                    self.test.values(),
                )
            )

    def __init__(self, depth=None, **kwargs):
        """
        Constructs a Decision Tree Classifier

        @param depth=None : maximum depth of the tree,
                            or None for no maximum depth
        """
        self._depth = depth or None  # makes 0 None as well
        self._schema = kwargs['schema']
        self.root = None

    def fit(self, X, y, sample_weight=None):
        """ Build a decision tree classifier trained on data (X, y) """
        self.root = self._fit(
            X,
            (y + 1) / 2,  # convert 1/-1 to 1/0
            set(range(len(self._schema.feature_names))),
            {},
            0,
        )

    def _fit(self, X, y, features, used_splits, depth, parent=None):
        if not features or \
                self._depth == depth or \
                self.pure_partition(y):

            # convert 1/0 back to 1/-1
            return self.majority_node((y * 2) - 1, parent)

        _features = features
        best_feature, best_GR, best_split = None, -np.inf, None
        for feature in features:
            X_values = X[:, feature]
            if not self._schema.is_nominal(feature):
                sorted_indices = np.argsort(X_values)
                sorted_y = y[sorted_indices]
                test_values = set([X_values[i]
                                   for i in range(len(X_values) - 1) if
                                   sorted_y[i] != sorted_y[i + 1]])
                test_values = test_values - used_splits.get(feature, set())
                if not test_values:
                    # if there are no values left to test, stop checking
                    _features = _features - set([feature])
                for tv in test_values:
                    gr = gain_ratio(X_values, y, continuous_value=tv)
                    if gr > best_GR:
                        best_GR = gr
                        best_feature = feature
                        best_split = tv
            else:
                gr = gain_ratio(X_values, y)
                if gr > best_GR:
                    best_GR = gr
                    best_feature = feature
                    best_split = None
        if best_split:
            # If we ended up choosing a continuous feature, record the split
            # index we used so we don't use it again deeper in the subtree
            used_feature_splits = used_splits.get(best_feature, set())
            used_splits[best_feature] = used_feature_splits | set([best_split])

        n = DecisionTree.Node(
            feature=best_feature,
            split=best_split,
            parent=parent,
        )
        partition = self.partition(X, y, best_feature, split=best_split)
        if not best_split:
            # Only remove features from consideration if they are discrete
            _features = _features - set([best_feature])
        for x_part, y_part in partition:
            value = best_split or x_part[0][best_feature]
            n.add_test(
                value,
                self._fit(
                    x_part,
                    y_part,
                    _features,
                    dict(used_splits),
                    depth + 1,
                ),
            )
        return n

    def partition(self, X, y, feature, split=None):
        """
        Partition examples (X) and labels (y) based on the feature.
        Returns a list of tuples, one tuple for each "bucket" of the
        partition.
        """
        if split:
            left = np.where(X[:, feature] <= split)[0]
            right = np.where(X[:, feature] > split)[0]
            rows = [left, right]
        else:
            counts = np.bincount(X.astype(int)[:, feature])
            rows = [np.where(X[:, feature] == v)[0]
                    for v in xrange(len(counts)) 
                    if counts[v] != 0]

        return [(X[r], y[r]) for r in rows]

    def majority_node(self, y, parent):
        """
        Return a node which assigns the majority class label for an example.
        If the set of labels (y) is empty, pick between 1 and -1 randomly.
        """
        if len(y):
            label = scipy.stats.mode(y).mode[0]
        else:
            # Choose between 1 and -1 randomly if the set of labels is empty
            label = (-1) ** np.random.random_integers(0, 1)
        return DecisionTree.Node(label=label, parent=parent)

    def pure_partition(self, y):
        return len(np.unique(y)) <= 1

    def predict(self, X):
        """ Return the -1/1 predictions of the decision tree """
        return np.array([self.root.predict(x) for x in X])

    def predict_proba(self, X):
        """ Return the probabilistic output of label prediction """
        pass

    def size(self):
        """
        Return the number of nodes in the tree
        """
        if self.root is None:
            return 0
        return self.root.size()

    def depth(self):
        """
        Returns the maximum depth of the tree
        (A tree with a single root node has depth 0)
        """
        if self.root is None:
            return 0
        return self.root.depth() - 1
