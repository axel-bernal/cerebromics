#!/usr/bin/env python
'''
Created on May 17, 2016

@author: twong
'''

import datastack.cerebro.cross_validation as cross_validation
import datastack.cerebro.linear_model.ridge

import os
import pandas as pd
import sklearn.grid_search
import sklearn.linear_model
import sklearn.metrics
import sys
import unittest


_BASENAME = os.path.splitext(os.path.realpath(__file__))[0]
_TEST_DATA = _BASENAME + '.data'


_TEST_CLASSIFICATION_GENDER_TARGET = 'facepheno.Sex'

_TEST_CLASSIFICATION_PARAM_GRID = {
    'C': [
        0.0001,
        0.0005,
        0.001,
        0.005,
        0.01,
        0.05,
        0.1,
        0.5,
        1,
        10,
        100,
        1000,
        10000,
    ]
}

_TEST_REGRESSION_HEIGHT_TARGET = 'facepheno.height'
_TEST_REGRESSION_WEIGHT_TARGET = 'facepheno.weight'
_TEST_REGRESSION_TARGETS = [_TEST_REGRESSION_HEIGHT_TARGET, _TEST_REGRESSION_WEIGHT_TARGET]

_TEST_REGRESSION_PARAM_GRID = {
    'alpha': [
        0.0001,
        0.0005,
        0.001,
        0.005,
        0.01,
        0.05,
        0.1,
        0.5,
        1,
        5,
        10
    ],
    'fit_intercept': [
        True,
        False
    ]
}


class TestCrossValidationInnerBasic(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._n_jobs = 1
        if ('darwin' not in sys.platform) or ('conda' in sys.version):
            cls._n_jobs = 8
        cls._voice_keys = {'voice': ['dynamic.FACE.voice.v2_visit1.complete.coeff{0:02d}'.format(i) for i in range(100)]}
        cls._df = pd.read_pickle(_TEST_DATA)
        cls._kf = cross_validation.HashedKfolds(cls._df)
        cls._verbose = False

    def _fit_estimator(self, estimator, param_grid, X, y, scoring=None):
        X_train = self._kf.df[X]
        y_train = self._kf.df[y]
        cv = cross_validation.CrossValidation(
            estimator,
            scoring=scoring,
            param_grid=param_grid,
            n_jobs=self._n_jobs,
            verbose=self._verbose,
            inner_loop=cross_validation.CrossValidationInnerBasic(),
        )
        # Our backward-compatibility inner loop will converge on the same
        # best set of hyperparameter values as GridSearchCV, but returns
        # a different score because of the way it retrains estimators with
        # the best set for each train/test split.
        cv.fit(X_train, y_train, with_legacy_grid_score=True)
        gs = sklearn.grid_search.GridSearchCV(estimator, param_grid, scoring=scoring, n_jobs=self._n_jobs, verbose=self._verbose,)
        gs.fit(X_train, y_train)
        return (gs, cv)

    def testInnerBasicVsGridSearchClassifyGender(self):
        """Test that the basic inner method works the same way as sklearn grid search when classifying gender.
        """
        estimator = sklearn.linear_model.LogisticRegression()
        gs, cv = self._fit_estimator(estimator, _TEST_CLASSIFICATION_PARAM_GRID, self._voice_keys['voice'], _TEST_CLASSIFICATION_GENDER_TARGET)
        self.assertEqual(gs.best_score_, cv.best_score_, 'Expected {}, got {}'.format(gs.best_score_, cv.best_score_))

    def testInnerBasicVsGridSearchRegressHeight(self):
        """Test that the basic inner method works the same way as sklearn grid search when regressing height.
        """
        estimator = datastack.cerebro.linear_model.ridge.Ridge()
        gs, cv = self._fit_estimator(estimator, _TEST_REGRESSION_PARAM_GRID, self._voice_keys['voice'], _TEST_REGRESSION_HEIGHT_TARGET)
        self.assertEqual(gs.best_score_, cv.best_score_, 'Expected {}, got {}'.format(gs.best_score_, cv.best_score_))

    def testInnerBasicVsGridSearchWithLasso(self):
        """Test that the basic inner method works the same way as sklearn grid search when using a different estimator.
        """
        estimator = sklearn.linear_model.Lasso()
        gs, cv = self._fit_estimator(estimator, _TEST_REGRESSION_PARAM_GRID, self._voice_keys['voice'], _TEST_REGRESSION_HEIGHT_TARGET)
        self.assertEqual(gs.best_score_, cv.best_score_, 'Expected {}, got {}'.format(gs.best_score_, cv.best_score_))

    def testInnerBasicVsGridSearchRegressWeight(self):
        """Test that the basic inner method works the same way as sklearn grid search when regressing weight.
        """
        estimator = datastack.cerebro.linear_model.ridge.Ridge()
        gs, cv = self._fit_estimator(estimator, _TEST_REGRESSION_PARAM_GRID, self._voice_keys['voice'], _TEST_REGRESSION_WEIGHT_TARGET)
        self.assertEqual(gs.best_score_, cv.best_score_, 'Expected {}, got {}'.format(gs.best_score_, cv.best_score_))

    def testInnerBasicVsGridSearchRegressMultiOutput(self):
        """Test that the basic inner method works the same way as sklearn grid search when regressing a multi-output target.
        """
        estimator = datastack.cerebro.linear_model.ridge.Ridge()
        scoring = sklearn.metrics.make_scorer(sklearn.metrics.r2_score, multioutput='variance_weighted')
        gs, cv = self._fit_estimator(estimator, _TEST_REGRESSION_PARAM_GRID, self._voice_keys['voice'], _TEST_REGRESSION_TARGETS, scoring=scoring)
        self.assertEqual(gs.best_score_, cv.best_score_, 'Expected {}, got {}'.format(gs.best_score_, cv.best_score_))

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
