#!/usr/bin/env python
"""
The main script for running experiments
"""
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import numpy.random
import scipy

from ann import ArtificialNeuralNetwork
from bagger import Bagger
from booster import Booster
from data import get_dataset
from dtree import DecisionTree
from folds import get_folds
from logistic_regression import LogisticRegression
from nbayes import NaiveBayes
from stats import StatisticsManager


CLASSIFIERS = {
    'dtree': DecisionTree,
    'nbayes': NaiveBayes,
    'logistic_regression': LogisticRegression,
    'ann': ArtificialNeuralNetwork,
}

META_ALGORITHMS = {
    'bagging': Bagger,
    'boosting': Booster,
}


def get_classifier(**options):
    """
    Create an instance of the classifier and return it.
    If using bagging/boosting, create an instance of that instead, using the
    classifier name and options
    """
    classifier_name = options.pop('classifier')
    if classifier_name not in CLASSIFIERS:
        raise ValueError(
            '"{}" classifier not implemented.'.format(classifier_name),
        )

    if "meta_algorithm" in options:
        meta = options.pop("meta_algorithm")
        iters = options.pop("meta_iters")
        classifier = META_ALGORITHMS[meta](
            algorithm=classifier_name,
            iters=iters,
            **options
        )
        return classifier
    else:
        return CLASSIFIERS[classifier_name](**options)


def main(**options):
    dataset_directory = options.pop('dataset_directory', '.')
    dataset = options.pop('dataset')
    k = options.pop('k')

    if "meta_algorithm" in options and "meta_iters" not in options:
        # Make sure they use --meta-iters if they want to do bagging/boosting
        raise ValueError(
            "Please indicate number of iterations for {}".format(
                options["meta_algorithm"],
            ),
        )

    fs_alg = None
    if "fs_algorithm" in options:
        fs_alg = options.pop("fs_algorithm")
        if "fs_features" not in options:
            raise ValueError(
                "Please indicate number of features for {}".format(fs_alg),
            )
        fs_n = options.pop("fs_features")

    schema, X, y = get_dataset(dataset, dataset_directory)
    folds = get_folds(X, y, k)
    stats_manager = StatisticsManager()
    options['schema'] = schema
    for train_X, train_y, test_X, test_y in folds:

        # Construct classifier instance
        print(options)
        classifier = get_classifier(**options)

        # Train classifier
        train_start = time.time()
        if fs_alg:
            selector = FS_ALGORITHMS[fs_alg](n=fs_n)
            selector.fit(train_X)
            train_X = selector.transform(train_X)

        randoms = np.random.rand(len(train_y))
        train_y[randoms < options['flip_probs']] = \
            -1 * train_y[randoms < options['flip_probs']]
        classifier.fit(train_X, train_y)
        train_time = (train_start - time.time())

        if fs_alg:
            test_X = selector.transform(test_X)
        predictions = classifier.predict(test_X)
        print(len(np.where(predictions == test_y)[0]), len(test_y))
        scores = classifier.predict_proba(test_X)
        if len(np.shape(scores)) > 1 and np.shape(scores)[1] > 1:
            scores = scores[:, 1]   # Get the column for label 1
        stats_manager.add_fold(test_y, predictions, scores, train_time)

    print(
        '      Accuracy: %.03f %.03f'
        % stats_manager.get_statistic('accuracy', pooled=False),
    )

    print(
        '     Precision: %.03f %.03f'
        % stats_manager.get_statistic('precision', pooled=False),
    )

    print(
        '        Recall: %.03f %.03f'
        % stats_manager.get_statistic('recall', pooled=False),
    )

    print(
        'Area under ROC: %.03f'
        % stats_manager.get_statistic('auc', pooled=True),
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        prog='main.py',
        description='Runs experiments for EECS 440.',
        argument_default=argparse.SUPPRESS,
    )

    parser.add_argument(
        '--dataset_directory',
        dest='dataset_directory',
        metavar='DIR',
        default='data/',
    )
    parser.add_argument('--k', metavar='FOLDS', type=int, default=5)
    parser.add_argument('--dataset', metavar='DATASET', default='voting')
    parser.add_argument(
        '--meta_algorithm',
        metavar='ALG',
        required=False,
        choices=['bagging', 'boosting'],
        help='Bagging or boosting, if desired',
    )
    parser.add_argument(
        '--meta_iters',
        metavar='N',
        type=int,
        required=False,
        help='Iterations for bagging or boosting, if applicable',
    )
    parser.add_argument(
        '--fs_algorithm',
        metavar='ALG',
        required=False,
        choices=['pca'],
        help='Feature selection algorithm, if desired',
    )
    parser.add_argument(
        '--fs_features',
        metavar='N',
        required=False,
        type=int,
        help='Number of feature to select, if applicable',
    )
    parser.add_argument(
        '--flip_probs',
        type=float,
        default=0,
    )

    subparsers = parser.add_subparsers(
        dest='classifier',
        help='Classifier options',
    )

    dtree_parser = subparsers.add_parser('dtree', help='Decision Tree')
    dtree_parser.add_argument(
        '--depth',
        type=int,
        help='Maximum depth for decision tree, 0 for None',
        default=1,
    )

    ann_parser = subparsers.add_parser('ann', help='Artificial Neural Network')
    ann_parser.add_argument(
        '--gamma',
        type=float,
        help='Weight decay coefficient',
        default=0.01,
    )
    ann_parser.add_argument(
        '--layer_sizes',
        type=int,
        help='Number of hidden layers',
        default=3,
    )
    ann_parser.add_argument(
        '--num_hidden',
        type=int,
        help='Number of hidden units in the hidden layers',
        default=0,
    )
    ann_parser.add_argument('--epsilon', type=float, required=False)
    ann_parser.add_argument('--max_iters', type=int, required=False)

    svm_parser = subparsers.add_parser(
        'linear_svm',
        help='Linear Support Vector Machine',
    )
    svm_parser.add_argument(
        '--c',
        type=float,
        help='Regularization parameter for SVM',
        default=1,
    )

    nbayes_parser = subparsers.add_parser('nbayes', help='Naive Bayes')
    nbayes_parser.add_argument(
        '--alpha',
        type=float,
        help='Smoothing parameter for Naive Bayes',
        default=1,
    )

    lr_parser = subparsers.add_parser(
        'logistic_regression',
        help='Logistic Regression',
    )
    lr_parser.add_argument(
        '--lambda',
        type=float,
        help='Regularization parameter for Logistic Regression',
        default=10,
    )

    args = parser.parse_args()
    main(**vars(args))
