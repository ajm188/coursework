"""
Statistics Computations
"""
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy


class StatisticsManager(object):

    def __init__(self):
        self.true_labels = []
        self.predicted_labels = []
        self.prediction_scores = []
        self.training_times = []
        self.statistics = {
            'accuracy': (accuracy, self.predicted_labels),
            'precision': (precision, self.predicted_labels),
            'recall': (recall, self.predicted_labels),
            'auc': (auc, self.prediction_scores),
        }

    def add_fold(self, true_labels, predicted_labels,
                 prediction_scores, training_time):
        """
        Add a fold of labels and predictions for later statistics computations

        @param true_labels : the actual labels
        @param predicted_labels : the predicted binary labels
        @param prediction_scores : the real-valued confidence values
        @param training_time : how long it took to train on the fold
        """
        self.true_labels.append(true_labels)
        self.predicted_labels.append(predicted_labels)
        self.prediction_scores.append(prediction_scores)
        self.training_times.append(training_time)

    def get_statistic(self, statistic_name, pooled=True):
        """
        Get a statistic by name, either pooled across folds or not

        @param statistic_name : one of {accuracy, precision, recall, auc}
        @param pooled=True : whether or not to "pool" predictions across folds
        @return statistic if pooled, or (avg, std) of statistic across folds
        """
        if statistic_name not in self.statistics:
            raise ValueError('"%s" not implemented' % statistic_name)

        statistic, predictions = self.statistics[statistic_name]

        if pooled:
            predictions = np.hstack(map(np.asarray, predictions))
            labels = np.hstack(map(np.asarray, self.true_labels))
            return statistic(labels, predictions)
        else:
            stats = []
            for l, p in zip(self.true_labels, predictions):
                stats.append(statistic(l, p))
            return np.average(stats), np.std(stats)


def accuracy(labels, predictions):
    """
    Returns the accuracy of a set of predictions given a set of true labels.

    Raises an assertion error if there are ever a different number of labels
    and predictions.
    """
    assert len(labels) == len(predictions)

    true_positives = TP(labels, predictions)
    true_negatives = TN(labels, predictions)
    return (true_positives + true_negatives) / float(len(labels))


def precision(labels, predictions):
    """
    Returns the precision of a set of predictions given a set of true labels.

    Raises an assertion error if there are ever a different number of labels
    and predictions.
    """
    assert len(labels) == len(predictions)

    true_positives = TP(labels, predictions)
    false_positives = FP(labels, predictions)
    if true_positives + false_positives == 0:
        return 1
    return true_positives / (true_positives + false_positives)


def recall(labels, predictions):
    """
    Returns the recall of a set of predictions given a set of true labels.

    Raises an assertion error if there are ever a different number of labels
    and predictions.
    """
    assert len(labels) == len(predictions)

    true_positives = TP(labels, predictions)
    false_negatives = FN(labels, predictions)
    if true_positives + false_negatives == 0:
        return 1
    return true_positives / (true_positives + false_negatives)


def specificity(labels, predictions):
    """
    Returns the specificity of a set of predictions given a set of true labels.

    Raises an assertion error if there are ever a different number of labels
    and predictions.
    """
    assert len(labels) == len(predictions)

    true_negatives = TN(labels, predictions)
    false_positives = FP(labels, predictions)
    if true_negatives + false_positives == 0:
        return 1
    return true_negatives / (true_negatives + false_positives)


def auc(labels, predictions):
    """
    Returns the area under ROC curve of a set of predictions given a set of
    true labels.
    """
    ordering = np.argsort(predictions)
    ordering = np.array([i for i in reversed(ordering)])
    labels = labels[ordering]
    predictions = predictions[ordering]
    roc_points = []
    # Compute recall and 1 - specificity for each confidence threshold,
    # including 0 and 1.
    for i in xrange(len(predictions) + 1):
        preds = np.zeros(predictions.shape) - 1
        preds[np.arange(i)] = 1
        roc_points.append(
            (recall(labels, preds), 1 - specificity(labels, preds)),
        )

    # Compute the area by going pairwise along the x-axis, adding the value
    # of the small trapezoid captured by this segment of the ROC curve.
    area = 0
    for i in xrange(len(roc_points) - 1):
        j = i + 1

        trap_lower_height = roc_points[i][0]
        trap_upper_height = roc_points[j][0]
        base_lower = roc_points[i][1]
        base_upper = roc_points[j][1]
        h = (trap_lower_height + trap_upper_height) / 2
        area += h * (base_upper - base_lower)

    return area


def TP(labels, predictions):
    return ((labels > 0) & (predictions > 0)).sum()


def TN(labels, predictions):
    return ((labels < 0) & (predictions < 0)).sum()


def FP(labels, predictions):
    return ((labels < 0) & (predictions > 0)).sum()


def FN(labels, predictions):
    return ((labels > 0) & (predictions < 0)).sum()
