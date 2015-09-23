#!/usr/bin/env python
"""
The main script for running experiments
"""
import time
import numpy as np

from stats import StatisticsManager
from data import get_dataset
import scipy
from dtree import DecisionTree
'''
from linear_svm import LinearSupportVectorMachine
from nbayes import NaiveBayes
from logistic_regression import LogisticRegression
from ann import ArtificialNeuralNetwork

from bagger import Bagger
from booster import Booster

from feature_selection import PCA
'''


CLASSIFIERS = {
    'dtree': DecisionTree,
    # 'linear_svm'                : LinearSupportVectorMachine,
    # 'nbayes'                    : NaiveBayes,
    # 'logistic_regression'       : LogisticRegression,
    # 'ann'                       : ArtificialNeuralNetwork,
}
'''
META_ALGORITHMS = {
    'bagging'           : Bagger,
    'boosting'          : Booster,
}

FS_ALGORITHMS = {
    'pca'               : PCA,
}
'''


def get_classifier(**options):
    """
    Create an instance of the classifier and return it. If using
    bagging/boosting, create an instance of that instead, using the classifier
    name and options
    """
    classifier_name = options.pop('classifier')
    if classifier_name not in CLASSIFIERS:
        raise ValueError('"%s" classifier not implemented.' % classifier_name)

    if "meta_algorithm" in options:
        meta = options.pop("meta_algorithm")
        iters = options.pop("meta_iters")
        return META_ALGORITHMS[meta](
            algorithm=classifier_name,
            iters=iters,
            **options,
        )
    else:
        return CLASSIFIERS[classifier_name](**options)


def get_folds(X, y, k):
    """
    Return a list of stratified folds for cross validation

    @param X : NumPy array of examples
    @param y : NumPy array of labels
    @param k : number of folds
    @return (train_X, train_y, test_X, test_y) for each fold
    """
    folds = {}
    # partition the examples into postive and negative sets
    positive_indices = np.argwhere(y == 1)
    negative_indices = np.argwhere(y == -1)
    assert len(positive_indices) + len(negative_indices) == len(y)
    for example, label in zip(X, y):
        l = positive if label == 1 else negative
        l.append((example, label))
    # number of pos/neg that should be in each fold
    positive_len = len(positive) / k
    negative_len = len(negative) / k
    # shuffle both lists
    np.random.shuffle(positive)
    np.random.shuffle(negative)
    # put len(pos) / k positives and len(neg) / k negatives in each fold
    for i in xrange(k):
        positive_start = i * positive_len
        negative_start = i * negative_len
        fold_positive = positive[positive_start:(positive_start + positive_len)]
        fold_negative = negative[negative_start:(negative_start + negative_len)]
        fold_X, fold_y = [], []
        for t in fold_positive + fold_negative:
            fold_X.append(t[0])
            fold_y.append(t[1])
        folds[i] = [fold_X, fold_y]

    # for pos/neg leftovers, randomly distribute into the folds
    leftovers = positive[(k-1)*positive_len:] + \
            negative[(k-1)*negative_len:]
    for leftover in leftovers:
        (example, label) = leftover
        fold_index = np.random.randint(0, k)
        fold = folds[fold_index]
        fold[0].append(example)
        fold[1].append(label)

    # for each fold, hold it aside for testing, and use the rest of the folds
    # as training data
    train_X, train_y, test_X, test_y = [], [], [], []
    for i in xrange(k):
        train_folds = [folds[j] for j in folds.keys() if i != j]
        test_fold = folds[i]
        train_examples, train_labels = [], []
        for f in train_folds:
            train_examples.extend(f[0])
            train_labels.extend(f[1])
        train_X.append(train_examples)
        train_y.append(train_labels)
        test_X.append(test_fold[0])
        test_y.append(test_fold[1])

    return zip(train_X, train_y, test_X, test_y)


def main(**options):
    dataset_directory = options.pop('dataset_directory', '.')
    dataset = options.pop('dataset')
    k = options.pop('k')

    if "meta_algorithm" in options and "meta_iters" not in options:
        """ Make sure they use --meta-iters if they want to do bagging/boosting """
        raise ValueError("Please indicate number of iterations for %s" % options["meta_algorithm"])

    fs_alg = None
    if "fs_algorithm" in options:
        fs_alg = options.pop("fs_algorithm")
        if "fs_features" not in options:
            raise ValueError("Please indicate number of features for %s" % fs_alg)
        fs_n = options.pop("fs_features")

    schema, X, y = get_dataset(dataset, dataset_directory)
    # put schema in the options so the classifier has it
    options['schema'] = schema
    folds = get_folds(X, y, k)
    stats_manager = StatisticsManager()
    import pdb;pdb.set_trace()
    for train_X, train_y, test_X, test_y in folds:

        # Construct classifier instance
        print options
        classifier = get_classifier(**options)

        # Train classifier
        train_start = time.time()
        if fs_alg:
            selector = FS_ALGORITHMS[fs_alg](n=fs_n)
            selector.fit(train_X)
            train_X = selector.transform(train_X)
        classifier.fit(train_X, train_y)
        train_time = (train_start - time.time())

        if fs_alg:
            test_X = selector.transform(test_X)
        predictions = classifier.predict(test_X)
        scores = classifier.predict_proba(test_X)
        if len(np.shape(scores)) > 1 and np.shape(scores)[1] > 1:
            scores = scores[:,1]    # Get the column for label 1
        stats_manager.add_fold(test_y, predictions, scores, train_time)

    print ('      Accuracy: %.03f %.03f'
        % stats_manager.get_statistic('accuracy', pooled=False))
    '''
    print ('     Precision: %.03f %.03f'
        % stats_manager.get_statistic('precision', pooled=False))

    print ('        Recall: %.03f %.03f'
        % stats_manager.get_statistic('recall', pooled=False))

    print ('Area under ROC: %.03f'
        % stats_manager.get_statistic('auc', pooled=True))
    '''   

                


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        prog='main.py',
        description='Runs experiments for EECS 440.',
        argument_default=argparse.SUPPRESS)

    parser.add_argument('--dataset_directory', dest='dataset_directory', metavar='DIR', default='.')
    parser.add_argument('--k', metavar='FOLDS', type=int, default=5)
    parser.add_argument('--dataset', metavar='DATASET', default='voting')
    parser.add_argument('--meta_algorithm', metavar='ALG', required=False, choices=['bagging', 'boosting'], help='Bagging or boosting, if desired')
    parser.add_argument('--meta_iters', metavar='N', type=int, required=False, help='Iterations for bagging or boosting, if applicable')
    parser.add_argument('--fs_algorithm', metavar='ALG', required=False, choices=['pca'], help='Feature selection algorithm, if desired')
    parser.add_argument('--fs_features', metavar='N', required=False, type=int, help='Number of feature to select, if applicable')

    subparsers = parser.add_subparsers(dest='classifier', help='Classifier options')
    
    dtree_parser = subparsers.add_parser('dtree', help='Decision Tree')
    dtree_parser.add_argument('--depth', type=int, help='Maximum depth for decision tree, 0 for None', default=2)

    ann_parser = subparsers.add_parser('ann', help='Artificial Neural Network')
    ann_parser.add_argument('--gamma', type=float, help='Weight decay coefficient', default=0.01)
    ann_parser.add_argument('--layer_sizes', type=int, help='Number of hidden layers', default=3)
    ann_parser.add_argument('--num_hidden', type=int, help='Number of hidden units in the hidden layers', default=40)
    ann_parser.add_argument('--epsilon', type=float, required=False)
    ann_parser.add_argument('--max_iters', type=int, required=False)

    svm_parser = subparsers.add_parser('linear_svm', help='Linear Support Vector Machine')
    svm_parser.add_argument('--c', type=float, help='Regularization parameter for SVM', default=1)

    nbayes_parser = subparsers.add_parser('nbayes', help='Naive Bayes')
    nbayes_parser.add_argument('--alpha', type=float, help='Smoothing parameter for Naive Bayes', default=1)

    lr_parser = subparsers.add_parser('logistic_regression', help='Logistic Regression')
    lr_parser.add_argument('--c', type=float, help='Regularization parameter for Logistic Regression', default=10)

    args = parser.parse_args()
    main(**vars(args))
