"""
Statistics Computations
"""
import numpy as np
import scipy


class StatisticsManager(object):

    def __init__(self):
        self.true_labels = []
        self.predicted_labels = []
        self.prediction_scores = []
        self.training_times = []
        self.statistics = {
            'accuracy' : (accuracy,  self.predicted_labels),
            'precision': (precision, self.predicted_labels),
            'recall'   : (recall,    self.predicted_labels),
            'auc'      : (auc,       self.prediction_scores),
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

    Computes the true positive count by taking the intersection of the indices
    where the true label was positive, and the indices where the prediction was
    positive. This intersection contains all the indices where the label and
    prediction were both positive, so its length gives the number of true
    positives. The true negative count is then obtained similarly.

    Raises an assertion error if there are ever a different number of labels
    and predictions.
    """
    assert len(labels) == len(predictions)

    true_positives = len(
        np.intersect1d(
            np.where(labels > 0)[0],
            np.where(predictions > 0)[0],
            assume_unique=True,
        )
    )
    true_negatives = len(
        np.intersect1d(
            np.where(labels < 0)[0],
            np.where(predictions < 0)[0],
            assume_unique=True,
        )
    )
    return (true_positives + true_negatives) / float(len(labels))

def precision(labels, predictions):
    pass

def recall(labels, predictions):
    pass

def auc(labels, predictions):
    pass
