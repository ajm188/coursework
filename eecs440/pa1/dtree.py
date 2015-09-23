"""
The Decision Tree Classifier
"""
import numpy
import numpy.random
import scipy
import scipy.stats


class DecisionTree(object):

    class Node(object):

        def __init__(self, feature=None, label=None, parent=None):
            self.feature = feature
            self.parent = parent
            self.label = label
            self.test = {}

        def predict(self, x):
            if self.label:
                return self.label
            else:
                return self.test[x[self.feature]].predict(x)

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
            y,
            list(range(len(self._schema.feature_names))),
        )

    def _fit(self, X, y, features, parent=None):
        if not features or \
                self._depth - 1 == self.depth() or \
                self.pure_partition(y):

            return self.majority_node(y, parent)

        best_feature, best_GR = None, -np.inf
        for feature in features:
            gr = self.gain_ratio(X, y, feature)
            if gr > best_GR:
                best_GR = gr
                best_feature = feature
        n = Node(parent=parent)
        X_part, y_part = self.partition(X, y, feature)
        _features = features - best_feature
        for i in range(len(self._schema.nominal_values(best_feature))):
            value = self._schema.nominal_values(best_feature)[i]
            n.map(value, self._fit(X_part[i], y_part[i], _features))
        return n

    def gain_ratio(self, X, y, feature):
        pass

    def information_gain(self, X, y, feature):
        pass

    def entropy(self, X, y, feature):
        prob_f = numpy.vectorize(lambda v: numpy.sum(X[x == v]) / len(X))
        probs = numpy.apply_along_axis(
            prob_f,
            feature,
            self._schema.nominal_values(feature),
        )
        H = numpy.vectorize(lambda p: 0 if not p else p * numpy.log2(p))
        return -numpy.sum(numpy.nan_to_num(H(probs)))

    def majority_node(self, y, parent):
        if y:
            label = scipy.stats.mode(y).mode[0]
        else:
            # Choose between 1 and -1 randomly if the set of labels is empty
            label = (-1) ** numpy.random.random_integers(0, 1)
        return Node(label=label, parent=parent)

    def pure_partition(self, y):
        return len(numpy.unique(y)) <= 1

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
