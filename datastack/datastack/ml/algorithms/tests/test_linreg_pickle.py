#!/usr/bin/env python
'''
Tests for the HLI DS ridge regression linear model.

Created on Jan 21, 2016

@author: twong
'''

import logging
import numpy as np
import os.path
import pandas as pd
import pickle
import unittest

_BASENAME = os.path.splitext(os.path.realpath(__file__))[0]
_TEST_MODEL = _BASENAME + '.model'
_TEST_DATA = _BASENAME + '.data'


class TestLinregPickle(unittest.TestCase):
    """Test pickled data from the original home-grown version of the ridge
    regression linear model against newer
    """

    def setUp(self):
        unittest.TestCase.setUp(self)
        with open(_TEST_MODEL) as f:
            self._models = pickle.load(f)
        with open(_TEST_DATA) as f:
            self._data_df = pickle.load(f)
        self._X = self._data_df[
            [c for c in self._data_df.columns if not c.startswith('result')]]
        self._y = self._data_df[
            [c for c in self._data_df.columns if c.startswith('result')]]

    def test_linreg_pickle_against_9641c15(self):
        """Test pickled data against the refactored ridge regression linear model @ GitHub commit 9641c15.
        """
        # Retitle the observed y-frame columns so that we can compare the
        # observed and predicted y-frames directly.
        self._y.columns = range(0, self._y.columns.shape[0])
        # Fill out a predicted y-frame and compare.
        results = [pd.DataFrame(m.predict(self._X)) for m in self._models]
        y_pred = pd.concat(results, axis=1, ignore_index=True)
        self.assertEqual(
            self._y.shape,
            y_pred.shape,
            'Got misshapen predicted value data frame')
        self.assertTrue(np.allclose(self._y, y_pred))


if __name__ == '__main__':
    logging.basicConfig()
    unittest.main()
