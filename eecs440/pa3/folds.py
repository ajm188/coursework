import numpy as np


def get_folds(X, y, k):
    """
    Return a list of stratified folds for cross validation

    @param X : NumPy array of examples
    @param y : NumPy array of labels
    @param k : number of folds
    @return (train_X, train_y, test_X, test_y) for each fold
    """
    # temporarily change the 1/-1 nature of y to 1/0
    _y = (y + 1) / 2
    # partition the examples into postive and negative sets
    positive_indices = np.where(_y)[0]
    negative_indices = np.where(_y - 1)[0]
    assert len(positive_indices) + len(negative_indices) == len(y)

    # shuffle both lists
    np.random.shuffle(positive_indices)
    np.random.shuffle(negative_indices)

    # create k buckets of indices of (approximately) equal size
    positive_folds_indices = \
        np.array(np.array_split(positive_indices, k))
    negative_folds_indices = \
        np.array(np.array_split(negative_indices, k))

    train_X, train_y, test_X, test_y = [], [], [], []
    for i in xrange(k):
        train_folds = np.concatenate((np.arange(0, i), np.arange(i+1, k)))
        pos_train_indices = np.concatenate(positive_folds_indices[train_folds])
        neg_train_indices = np.concatenate(negative_folds_indices[train_folds])
        pos_test_indices = positive_folds_indices[i]
        neg_test_indices = negative_folds_indices[i]

        train_X.append(
            np.concatenate((X[pos_train_indices], X[neg_train_indices]))
        )
        train_y.append(
            np.concatenate((y[pos_train_indices], y[neg_train_indices]))
        )
        test_X.append(
            np.concatenate((X[pos_test_indices], X[neg_test_indices]))
        )
        test_y.append(
            np.concatenate((y[pos_test_indices], y[neg_test_indices]))
        )

    return zip(train_X, train_y, test_X, test_y)
