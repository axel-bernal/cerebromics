#!/usr/bin/env python
"""
Tests for the HLI DS ridge regression linear model.

Created on Oct 30, 2015

@author: twong
"""

import datastack.cerebro.linear_model as linear_model
import datastack.ml.baseregress as baseregress
import datastack.serializer as serializer
import datastack.settings as settings
import logging
import numpy as np
import os.path
import pandas as pd
import time
import unittest
import sys

from sklearn.grid_search import GridSearchCV

_BASENAME = os.path.splitext(os.path.realpath(__file__))[0]
_TEST_DATA = _BASENAME + '.data'
_TEST_SERIALIZED_DATA = _BASENAME + '.szd'

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

_TEST_RIDGE_PARAM_GRID_SMALL = {
    "alpha": [
        0.001,
        0.01,
        0.1,
        1.01,
        10.0]}

_TEST_COVARIATES = ['facepheno.height', 'dynamic.FACE.pheno.v1.bmi']
_TEST_TARGET_SINGLE = 'facepheno.hand.strength.right.m1'
_TEST_TARGET_MULTI = [
    'facepheno.hand.strength.right.m1',
    'facepheno.hand.strength.right.m1']


class TestRidge(unittest.TestCase):
    """Test the more accurate home-grown ridge regression linear model.
    """
    # twong: I generated the golden values using clippert's original ridge
    # implementation.
    _SINGLE_SCORE_GOLDEN = 0.3082724292438539
    _MULTI_SCORE_GOLDEN = 0.30827242924385401

    _GRID_ALPHA_GOLDEN = 10
    _GRID_FIT_INTERCEPT_GOLDEN = True
    _GRID_SCORE_GOLDEN = 0.17432301509314338

    _BASELINE_METRICS_GOLDEN = [0.252616867289, 7.6623632162, 92.8043484436]

    @classmethod
    def setUpClass(cls):
        super(TestRidge, cls).setUpClass()
        df = pd.read_pickle(_TEST_DATA)
        # The original dataframe doesn't have consented or related information,
        # which might trigger a call-out to Rosetta within the base regression
        # library.
        for c in settings.QUICKSILVER_KFOLDS_HOLDOUT_COLUMNS:
            df[c] = False
        for c in settings.QUICKSILVER_KFOLDS_TOGETHER_COLUMNS:
            df[c] = [[] for _ in range(0, df.shape[0])]
        cls._data_df = df
        cls._X = cls._data_df[_TEST_COVARIATES]
        cls._Y_single = cls._data_df[_TEST_TARGET_SINGLE]
        cls._Y_multi = cls._data_df[_TEST_TARGET_MULTI]

    @classmethod
    def tearDownClass(cls):
        super(TestRidge, cls).tearDownClass()
        if os.path.exists(_TEST_SERIALIZED_DATA):
            os.unlink(_TEST_SERIALIZED_DATA)

    def testRegressionSingleTarget(self):
        """Test the basic ridge regression model for a single target.
        """
        model = linear_model.Ridge()
        fitted_model = model.fit(self._X, self._Y_single)
        score = fitted_model.score(self._X, self._Y_single)
        self.assertTrue(np.allclose(self._SINGLE_SCORE_GOLDEN, score))

    def testRegressionSingleDataframeTarget(self):
        """Test the basic ridge regression model for a single DATAFRAME target.
        """
        model = linear_model.Ridge()
        _df = pd.DataFrame(self._Y_single)
        fitted_model = model.fit(self._X, _df)
        score = fitted_model.score(self._X, _df)
        self.assertTrue(np.allclose(self._SINGLE_SCORE_GOLDEN, score))

    def testRegressionMultiOutputTarget(self):
        """Test the basic ridge regression model for multi-output targets.
        """
        model = linear_model.Ridge()
        fitted_model = model.fit(self._X, self._Y_multi)
        # Specify multiple output weighting to guard against changes to the
        # default between sklearn 0.17 and 0.18.
        score = fitted_model.score(
            self._X,
            self._Y_multi,
            multioutput='variance_weighted')
        self.assertTrue(np.allclose(self._MULTI_SCORE_GOLDEN, score))

    def testGridSearchRandomData(self):
        """Test cross-validation using the ridge regression model with all or some regularized coefficients.
        """
        N = 100
        D = 100000
        P = 2
        ITERATIONS = 10
        idx_covariates = np.zeros(D, dtype=np.bool)
        idx_covariates[0] = True
        idx_covariates[-1] = True
        seed = int(time.time())
        np.random.seed(seed=seed)
        for i in range(0, ITERATIONS):
            # Generate some random predictor and target values
            X = np.random.randn(N, D)
            beta = 0.04 * np.random.randn(D, P)
            beta[0, :] *= 1000.0
            Y = np.random.randn(N, P) * 50.0 + X.dot(beta)
            Y[:, 0] += 4
            Y[:, 0] += 40
            # If we're not running under OS X, parallelize the grid search
            # to reduce the test time.
            n_jobs = 1
            if ('darwin' not in sys.platform) or ('conda' in sys.version):
                n_jobs = 4
            # Regularize all then all but two of the coefficients
            gridsearch = GridSearchCV(
                estimator=linear_model.Ridge(),
                param_grid=_TEST_RIDGE_PARAM_GRID_SMALL)
            gridsearch.fit(X=X, y=Y[:, 0])
            gridsearch_unregularized = GridSearchCV(
                estimator=linear_model.Ridge(),
                fit_params={
                    "idx_covariates": idx_covariates},
                param_grid=_TEST_RIDGE_PARAM_GRID_SMALL,
                n_jobs=n_jobs)
            gridsearch_unregularized.fit(X=X, y=Y[:, 0])
            # Compare the R2 score from each.
            self.assertNotEqual(
                gridsearch.best_score_,
                gridsearch_unregularized.best_score_,
                'Got equal grid search scores with different estimators: '
                'Seed %d, iteration %d' % (seed, i))

    def testGridSearchSingleTarget(self):
        """Test cross-validation using the ridge regression model for a single target.
        """
        grid = GridSearchCV(
            linear_model.Ridge(),
            _TEST_REGRESSION_PARAM_GRID)
        grid.fit(self._X, self._Y_single)
        self.assertEqual(
            self._GRID_ALPHA_GOLDEN,
            grid.best_params_['alpha'],
            'Expected alpha of 10')
        self.assertEqual(
            self._GRID_FIT_INTERCEPT_GOLDEN,
            grid.best_params_['fit_intercept'],
            'Expected %s fit-to-intercept flag' % (self._GRID_FIT_INTERCEPT_GOLDEN))
        self.assertTrue(
            np.allclose(
                self._GRID_SCORE_GOLDEN,
                grid.best_score_),
            'Got an unexpected R2 value')

    def testBaseRegressionMultiOutputTarget(self):
        """Test our baseline regression suite using the ridge regression model for multiple targets.
        """
        covariates = {'covariates': _TEST_COVARIATES}
        r = baseregress.BaseRegress(
            _TEST_TARGET_MULTI,
            covariates=covariates,
            use_predicted_baseline_covariates=baseregress.BL_NEITHER,
            dataframe=self._data_df)
        estimator = linear_model.Ridge()
        r.run(
            estimator=estimator,
            params=_TEST_REGRESSION_PARAM_GRID,
            run_keys=['covariates'],
            with_aggregate_covariates=False,
            with_bootstrap=False)

        # Get just the metrics for ridge.
        metrics = r.metrics_df[
            r.metrics_df.Model.str.match(
                estimator.__class__.__name__)]
        metrics_ridge = [metrics.R2.values[0], metrics.MAE.values[0], metrics.MSE.values[0]]
        self.assertTrue(
            np.allclose(
                self._BASELINE_METRICS_GOLDEN,
                metrics_ridge),
            'Got an unexpected loss metric value: Expected {}, got {}'.format(
                self._BASELINE_METRICS_GOLDEN,
                metrics_ridge))

    def testGridSearchSingleTargetAndSave(self):
        """Test that we can save and load a trained ridge model.
        """
        grid = GridSearchCV(linear_model.Ridge(), _TEST_REGRESSION_PARAM_GRID)
        grid.fit(self._X, self._Y_single)
        y_pred = grid.best_estimator_.predict(self._X)
        with open(_TEST_SERIALIZED_DATA, 'w') as f:
            serializer.save(grid.best_estimator_, f)
        ridge = serializer.load(_TEST_SERIALIZED_DATA)
        self.assertTrue((y_pred == ridge.predict(self._X)).all())

if __name__ == '__main__':
    logging.basicConfig()
    unittest.main()
