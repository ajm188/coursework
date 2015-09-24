"""
The Decision Tree Classifier
"""
from __future__ import print_function

import numpy as np
import numpy.random
import scipy
import scipy.stats


def H(Y, given=None):
    if given is not None:
        X = given
        X_counts = np.bincount(X)
        X_probs = X_counts / float(len(X))
        cond_entropies = np.apply_along_axis(
            lambda i: H(Y[np.where(X == i)[0]]),
            0,
            np.where(X_counts != 0)[0],
        )
        return np.sum(cond_entropies)

    if not len(Y):
        return 0
    probabilities = np.bincount(Y) / float(len(Y))
    log_probabilities = np.log2(probabilities)
    log_probabilities[log_probabilities == -np.inf] = 0
    return -np.dot(probabilities, log_probabilities)


def IG(X, y):
    return H(y) - H(y, given=X)


def gain_ratio(X, y):
    H_X = H(X)
    if not H_X:
        return 0
    return IG(X, y) / H_X


class DecisionTree(object):

    class Node(object):

        def __init__(self, feature=None, label=None, parent=None):
            self.feature = feature
            self.parent = parent
            self.label = label
            self.test = {}

        def add_test(self, v, n):
            if v in self.test:
                raise Exception("can't set the same value twice")
            self.test[v] = n

        def predict(self, x):
            if self.label:
                return self.label
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


    def __init__(self, depth=None, **kwargs):
        """
        Constructs a Decision Tree Classifier

        @param depth=None : maximum depth of the tree,
                            or None for no maximum depth
        """
        self._depth = depth
        self._schema = kwargs['schema']
        self.root = None

    def fit(self, X, y, sample_weight=None):
        """ Build a decision tree classifier trained on data (X, y) """
        self.root = self._fit(
            X,
            (y + 1) / 2,  # convert 1/-1 to 1/0
            set(range(len(self._schema.feature_names))),
        )

    def _fit(self, X, y, features, parent=None):
        if not features or \
                self._depth - 1 == self.depth() or \
                self.pure_partition(y):

            # convert 1/0 back to 1/-1
            return self.majority_node((y * 2) - 1, parent)

        best_feature, best_GR = None, -np.inf
        for feature in features:
            X_values = X[:,feature]
            gr = gain_ratio(X_values, y)
            if gr > best_GR:
                best_GR = gr
                best_feature = feature
        n = DecisionTree.Node(feature=best_feature, parent=parent)
        partition = self.partition(X, y, best_feature)
        _features = features - set([best_feature])
        for x_part, y_part in partition:
            value = x_part[0][best_feature]
            n.add_test(value, self._fit(x_part, y_part, _features))
        return n

    def partition(self, X, y, feature):
        """
        Partition examples (X) and labels (y) based on the feature.
        Returns a list of tuples, one tuple for each "bucket" of the
        partition.
        """
        x_bins = np.bincount(X[:,feature])
        values = np.where(x_bins != 0)[0]
        rows = []
        for v in values:
            rows.append(np.where(X[:,feature] == v)[0])
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
        return [self.root.predict(x) for x in X]

    def predict_proba(self, X):
        """ Return the probabilistic output of label prediction """
        pass

    def size(self):
        """
        Return the number of nodes in the tree
        """
        pass

    def depth(self):
        """
        Returns the maximum depth of the tree
        (A tree with a single root node has depth 0)
        """
        pass
