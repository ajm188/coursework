from __future__ import print_function

import numexpr
import pandas as pd


class AlreadySpannedRangeException(Exception):
    pass


class RangeDict(object):

    def __init__(self):
        self.df = pd.DataFrame(columns=('lb', 'ub', 'val'))

    def __setitem__(self, r, v):
        lb, ub = r
        # do some bounds checking
        # do we already have this mapping?
        query = self.df.query('(lb == @lb) & (ub == @ub)'.format(lb, ub))
        if not query.empty:
            self.df.loc[query.index[0]]['val'] = v
        # is this range spanned by any other ranges in the dict?
        if not self.df.query(
            '((lb < @lb) & (ub > @lb)) | ((lb < @ub) & (ub > @ub))',
        ).empty:
            raise AlreadySpannedRangeException
        self.df = self.df.append(
            {'lb': lb, 'ub': ub, 'val': v},
            ignore_index=True,
        )

    def __getitem__(self, k):
        query = self.df.query('lb < {} <= ub'.format(k))
        if query.empty:
            raise KeyError
        return self.df.loc[query.index[0]]['val']

    def get(self, k, default=None):
        try:
            return self[k]
        except KeyError:
            return default
