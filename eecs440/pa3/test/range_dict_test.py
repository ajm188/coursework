import numpy as np

from ..range_dict import AlreadySpannedRangeException
from ..range_dict import RangeDict


def test_construction():
    RangeDict()


def test_insertion():
    rd = RangeDict()
    rd[(3, 4)] = 5


def test_duplicate_insertion():
    rd = RangeDict()
    rd[(3, 4)] = 5
    rd[(3, 4)] = 6


def test_many_inserts():
    rd = RangeDict()
    for i, j in zip(xrange(5), xrange(1, 6)):
        rd[(i, j)] = i + j


def test_bad_insert():
    rd = RangeDict()
    rd[(1, 4)] = 3
    try:
        rd[(2, 3)] = 5
        assert False, 'Did not raise AlreadySpannedRangeException'
    except AlreadySpannedRangeException:
        pass


def test_lb_neg_inf_insert_does_not_raise_TypeError():
    rd = RangeDict()
    try:
        rd[(-np.inf, 54.3)] = 4
        assert True
    except TypeError:
        assert False, 'Should not have raised TypeError'


def test_item_retrieval():
    rd = RangeDict()
    rd[(1, 4)] = 3
    assert rd[2] == 3


def test_item_retrieval_lower_bound_exclusive():
    rd = RangeDict()
    rd[(1, 4)] = 3
    try:
        rd[1]
        assert False, 'Did not raise KeyError'
    except KeyError:
        pass


def test_item_retrieval_upper_bound_inclusive():
    rd = RangeDict()
    rd[(1, 4)] = 3
    assert rd[4] == 3


def test_item_retrieveal_works_for_floats():
    rd = RangeDict()
    rd[(1, 4)] = 3
    assert rd[3.5] == 3


def test_complex_item_retrieval():
    rd = RangeDict()
    for i, j in zip(xrange(5), xrange(1, 6)):
        rd[(i, j)] = i + j
    assert rd[5] == 9
