#!/usr/bin/env python
'''
Tests for classes that ultimately depend on
datastack.cerebro.kfolds

Created on Nov 2, 2015

@author: twong
'''

import datastack.ml.cross_validation as cv
import datastack.ml.algorithms.linreg as hli_lm
import logging
import numpy as np
import os.path
import pandas as pd
import pickle
import sklearn.linear_model as sk_lm
import unittest

from bson.objectid import ObjectId

_BASENAME = os.path.splitext(os.path.realpath(__file__))[0]
_TEST_DATA = _BASENAME + '.data'

# Expected output of using a classifier produced from cross-validated
# training
CROSS_VALIDATION_CLASSIFICATION_Y_PRED_GOLDEN_RAW = \
    [u'Male',
     u'Male',
     u'Female',
     u'Male',
     u'Female',
     u'Male',
     u'Male',
     u'Female',
     u'Male',
     u'Female',
     u'Male',
     u'Male',
     u'Female',
     u'Male',
     u'Male',
     u'Male',
     u'Male',
     u'Female',
     u'Male',
     u'Male']

# Expected output of using a regression model produced from
# cross-validated training
CROSS_VALIDATION_REGRESSION_Y_PRED_GOLDEN_RAW = \
    ['0x1.54c40246a39b4p+7',
     '0x1.4fdb78b470864p+7',
     '0x1.55ad5eebb659ep+7',
     '0x1.551e99d3ae2c4p+7',
     '0x1.594246e78f98bp+7',
     '0x1.578890528e998p+7',
     '0x1.58771b6aef5a6p+7',
     '0x1.57a4e4415964bp+7',
     '0x1.4c574633cb5b0p+7',
     '0x1.4a56b3e1dd0a6p+7',
     '0x1.3f6e58b6f6c73p+7',
     '0x1.57e056efef804p+7',
     '0x1.3fb8bb5ba3b59p+7',
     '0x1.5816c17fd5541p+7',
     '0x1.5a33e72e21cdbp+7',
     '0x1.5515bcdf5dfa3p+7',
     '0x1.54d2a70e45967p+7',
     '0x1.557dac62b25adp+7',
     '0x1.58e35ca2eb083p+7',
     '0x1.4c7e4c5cd3c84p+7']

# Expected output of the classification evaluation
EVALUATION_CLASSIFICATION_Y_PRED_GOLDEN_RAW = \
    {ObjectId('55c6fddbc37a457d277112db'): u'Female',
     ObjectId('55c6fddbc37a457d277112e9'): u'Male',
     ObjectId('55c6fddcc37a457d277112ed'): u'Male',
     ObjectId('55c6fdddc37a457d27711348'): u'Female',
     ObjectId('55c6fdddc37a457d27711372'): u'Male',
     ObjectId('55c6fdddc37a457d27711386'): u'Female',
     ObjectId('55c6fddec37a457d277113d8'): u'Male',
     ObjectId('55c6fddfc37a457d2771141d'): u'Male',
     ObjectId('55c6fddfc37a457d27711428'): u'Male',
     ObjectId('55c6fde0c37a457d2771143a'): u'Female',
     ObjectId('55c6fde0c37a457d27711478'): u'Male',
     ObjectId('55c6fde0c37a457d27711479'): u'Female',
     ObjectId('55c6fde1c37a457d27711487'): u'Male',
     ObjectId('55c6fde1c37a457d2771148f'): u'Male',
     ObjectId('55c6fde1c37a457d277114a3'): u'Male',
     ObjectId('55c6fde1c37a457d277114a4'): u'Female',
     ObjectId('55c6fde2c37a457d277114e0'): u'Male',
     ObjectId('55c6fde4c37a457d27711596'): u'Female',
     ObjectId('55c6fde4c37a457d27711599'): u'Female',
     ObjectId('55c6fde4c37a457d2771159a'): u'Male',
     ObjectId('55c6fde4c37a457d2771159b'): u'Male',
     ObjectId('55c6fde4c37a457d277115a2'): u'Female',
     ObjectId('55c6fde4c37a457d277115a3'): u'Female',
     ObjectId('55c6fde4c37a457d277115a4'): u'Female',
     ObjectId('55c6fde4c37a457d277115a5'): u'Female',
     ObjectId('55c6fde4c37a457d277115a6'): u'Female',
     ObjectId('55c6fde4c37a457d277115be'): u'Male',
     ObjectId('55c6fde5c37a457d277115fa'): u'Female',
     ObjectId('55c6fdf2c37a457d27711a57'): u'Female',
     ObjectId('55c6fdf2c37a457d27711a5d'): u'Male',
     ObjectId('55c6fe28c37a457d277128f4'): u'Male',
     ObjectId('55c6fe28c37a457d277128f5'): u'Female',
     ObjectId('55c6fe28c37a457d277128fa'): u'Male',
     ObjectId('55c6fe28c37a457d277128fc'): u'Male',
     ObjectId('55c6fe28c37a457d277128fd'): u'Male',
     ObjectId('55c6fe28c37a457d277128fe'): u'Female',
     ObjectId('55c6fe28c37a457d277128ff'): u'Male',
     ObjectId('55c6fe28c37a457d27712911'): u'Male',
     ObjectId('55c6fe28c37a457d27712913'): u'Female',
     ObjectId('55c6fe28c37a457d27712914'): u'Male',
     ObjectId('55c6fe28c37a457d27712915'): u'Male',
     ObjectId('55c6fe28c37a457d27712918'): u'Male',
     ObjectId('55c6fe28c37a457d2771291a'): u'Female',
     ObjectId('55c6fe29c37a457d27712943'): u'Male',
     ObjectId('55c6fe29c37a457d27712944'): u'Male',
     ObjectId('55c6fe29c37a457d27712945'): u'Male',
     ObjectId('55c6fe29c37a457d27712946'): u'Male',
     ObjectId('55c6fe29c37a457d27712947'): u'Female',
     ObjectId('55c6fe30c37a457d27712c4b'): u'Male',
     ObjectId('55c6fe30c37a457d27712c4c'): u'Male',
     ObjectId('55c6fe30c37a457d27712c4f'): u'Female',
     ObjectId('55c6fe30c37a457d27712c50'): u'Female',
     ObjectId('55c6fe30c37a457d27712c52'): u'Male',
     ObjectId('55c6fe30c37a457d27712c55'): u'Male',
     ObjectId('55c6fe30c37a457d27712c57'): u'Female',
     ObjectId('55c6fe30c37a457d27712c59'): u'Female',
     ObjectId('55c6fe30c37a457d27712c5a'): u'Female',
     ObjectId('55c6fe30c37a457d27712c64'): u'Female',
     ObjectId('55c6fe30c37a457d27712c65'): u'Male',
     ObjectId('55c6fe30c37a457d27712c66'): u'Male',
     ObjectId('55c6fe30c37a457d27712c67'): u'Male',
     ObjectId('55c6fe30c37a457d27712c68'): u'Male',
     ObjectId('55c6fe30c37a457d27712c69'): u'Male',
     ObjectId('55c6fe30c37a457d27712c6a'): u'Male',
     ObjectId('55c6fe30c37a457d27712c6b'): u'Male',
     ObjectId('55c6fe30c37a457d27712c6c'): u'Male',
     ObjectId('55c6fe30c37a457d27712c6d'): u'Male',
     ObjectId('55c6fe30c37a457d27712c6e'): u'Male',
     ObjectId('55c6fe30c37a457d27712c6f'): u'Male',
     ObjectId('55c6fe30c37a457d27712c70'): u'Male',
     ObjectId('55c6fe30c37a457d27712c71'): u'Male',
     ObjectId('55c6fe30c37a457d27712c72'): u'Male',
     ObjectId('55c6fe30c37a457d27712c7d'): u'Male',
     ObjectId('55c6fe30c37a457d27712c7e'): u'Male',
     ObjectId('55c6fe30c37a457d27712c7f'): u'Male',
     ObjectId('55c6fe30c37a457d27712c80'): u'Female',
     ObjectId('55c6fe30c37a457d27712c81'): u'Female',
     ObjectId('55c6fe30c37a457d27712c84'): u'Male',
     ObjectId('55c6fe30c37a457d27712c85'): u'Female',
     ObjectId('55c6fe30c37a457d27712c86'): u'Female',
     ObjectId('55c6fe30c37a457d27712c87'): u'Female',
     ObjectId('55c6fe30c37a457d27712c88'): u'Female',
     ObjectId('55c6fe31c37a457d27712c89'): u'Male',
     ObjectId('55c6fe31c37a457d27712c8a'): u'Male',
     ObjectId('55c6fe31c37a457d27712c8c'): u'Female',
     ObjectId('55c6fe31c37a457d27712c8d'): u'Female',
     ObjectId('55c6fe31c37a457d27712c8e'): u'Male',
     ObjectId('55c6fe31c37a457d27712c8f'): u'Male',
     ObjectId('55c6fe31c37a457d27712c92'): u'Female',
     ObjectId('55c6fe31c37a457d27712ca8'): u'Male',
     ObjectId('55c6fe31c37a457d27712ca9'): u'Male',
     ObjectId('55c6fe31c37a457d27712cab'): u'Male',
     ObjectId('55c6fe31c37a457d27712cbe'): u'Male',
     ObjectId('55c6fe31c37a457d27712cd3'): u'Male',
     ObjectId('55c6fe31c37a457d27712cd7'): u'Male',
     ObjectId('55c6fe32c37a457d27712cf0'): u'Male',
     ObjectId('55c6fe32c37a457d27712cfc'): u'Female',
     ObjectId('55c6fe32c37a457d27712d04'): u'Male',
     ObjectId('55c6fe32c37a457d27712d09'): u'Female',
     ObjectId('55c6fe34c37a457d27712d6d'): u'Female'}

# Expected output of the regression evaluation
EVALUATION_REGRESSION_Y_PRED_GOLDEN_RAW = \
    {ObjectId('55c6fddbc37a457d277112db'): '0x1.5609d8dd36931p+7',
     ObjectId('55c6fddbc37a457d277112e9'): '0x1.565f3dc63f689p+7',
     ObjectId('55c6fddcc37a457d277112ed'): '0x1.5232ef288df45p+7',
     ObjectId('55c6fdddc37a457d27711348'): '0x1.54e7f9a265e92p+7',
     ObjectId('55c6fdddc37a457d27711372'): '0x1.57fece1123da8p+7',
     ObjectId('55c6fdddc37a457d27711386'): '0x1.526e220f69d10p+7',
     ObjectId('55c6fddec37a457d277113d8'): '0x1.557107615d0b1p+7',
     ObjectId('55c6fddfc37a457d2771141d'): '0x1.5ad204c31855ap+7',
     ObjectId('55c6fddfc37a457d27711428'): '0x1.52ad327eb8294p+7',
     ObjectId('55c6fde0c37a457d2771143a'): '0x1.54b2d69ade3cfp+7',
     ObjectId('55c6fde0c37a457d27711478'): '0x1.528158bd2aa6cp+7',
     ObjectId('55c6fde0c37a457d27711479'): '0x1.5bdabd23ca63bp+7',
     ObjectId('55c6fde1c37a457d27711487'): '0x1.56cbb527ec084p+7',
     ObjectId('55c6fde1c37a457d2771148f'): '0x1.51dc89136c4a8p+7',
     ObjectId('55c6fde1c37a457d277114a3'): '0x1.5969aa432a110p+7',
     ObjectId('55c6fde1c37a457d277114a4'): '0x1.4279af3394542p+7',
     ObjectId('55c6fde2c37a457d277114e0'): '0x1.57ebf6f300cf5p+7',
     ObjectId('55c6fde4c37a457d27711596'): '0x1.3f24ff67167b1p+7',
     ObjectId('55c6fde4c37a457d27711599'): '0x1.544493cba89f6p+7',
     ObjectId('55c6fde4c37a457d2771159a'): '0x1.552ca5e699239p+7',
     ObjectId('55c6fde4c37a457d2771159b'): '0x1.5afed978d8cffp+7',
     ObjectId('55c6fde4c37a457d277115a2'): '0x1.54b39bf35665ep+7',
     ObjectId('55c6fde4c37a457d277115a3'): '0x1.54c07d5df93fdp+7',
     ObjectId('55c6fde4c37a457d277115a4'): '0x1.4e56df21a867fp+7',
     ObjectId('55c6fde4c37a457d277115a5'): '0x1.520801e12e616p+7',
     ObjectId('55c6fde4c37a457d277115a6'): '0x1.563f4ac9a635bp+7',
     ObjectId('55c6fde4c37a457d277115be'): '0x1.546fc9be485fcp+7',
     ObjectId('55c6fde5c37a457d277115fa'): '0x1.45243c670ba00p+7',
     ObjectId('55c6fdf2c37a457d27711a57'): '0x1.592a1c39dec83p+7',
     ObjectId('55c6fdf2c37a457d27711a5d'): '0x1.56435abed1078p+7',
     ObjectId('55c6fe28c37a457d277128f4'): '0x1.57977d8475493p+7',
     ObjectId('55c6fe28c37a457d277128f5'): '0x1.4cffd1f434e56p+7',
     ObjectId('55c6fe28c37a457d277128fa'): '0x1.55ec5af5da07dp+7',
     ObjectId('55c6fe28c37a457d277128fc'): '0x1.4b36be618f14ep+7',
     ObjectId('55c6fe28c37a457d277128fd'): '0x1.5454391a9ba25p+7',
     ObjectId('55c6fe28c37a457d277128fe'): '0x1.5b3d8a49b397cp+7',
     ObjectId('55c6fe28c37a457d277128ff'): '0x1.59088c11648d0p+7',
     ObjectId('55c6fe28c37a457d27712911'): '0x1.536974d45ef1ep+7',
     ObjectId('55c6fe28c37a457d27712913'): '0x1.5419874b32ee8p+7',
     ObjectId('55c6fe28c37a457d27712914'): '0x1.51e8e3510cb72p+7',
     ObjectId('55c6fe28c37a457d27712915'): '0x1.4d521d07fbaecp+7',
     ObjectId('55c6fe28c37a457d27712918'): '0x1.51d62c55b70b2p+7',
     ObjectId('55c6fe28c37a457d2771291a'): '0x1.557c0aedf173dp+7',
     ObjectId('55c6fe29c37a457d27712943'): '0x1.3e642e9d76172p+7',
     ObjectId('55c6fe29c37a457d27712944'): '0x1.54d07ca32e22ap+7',
     ObjectId('55c6fe29c37a457d27712945'): '0x1.4d47cb1ad54d3p+7',
     ObjectId('55c6fe29c37a457d27712946'): '0x1.53dfed0095264p+7',
     ObjectId('55c6fe29c37a457d27712947'): '0x1.55b2679d10eeap+7',
     ObjectId('55c6fe30c37a457d27712c4b'): '0x1.583775b70e67cp+7',
     ObjectId('55c6fe30c37a457d27712c4c'): '0x1.53586687ecd9ap+7',
     ObjectId('55c6fe30c37a457d27712c4f'): '0x1.5df8817338025p+7',
     ObjectId('55c6fe30c37a457d27712c50'): '0x1.58fdfb557f555p+7',
     ObjectId('55c6fe30c37a457d27712c52'): '0x1.53083915c876bp+7',
     ObjectId('55c6fe30c37a457d27712c55'): '0x1.55267b028c664p+7',
     ObjectId('55c6fe30c37a457d27712c57'): '0x1.58e7be5ed2e3cp+7',
     ObjectId('55c6fe30c37a457d27712c59'): '0x1.422c71a28b1d4p+7',
     ObjectId('55c6fe30c37a457d27712c5a'): '0x1.3f5db43c53fd2p+7',
     ObjectId('55c6fe30c37a457d27712c64'): '0x1.4b3327aa4de9cp+7',
     ObjectId('55c6fe30c37a457d27712c65'): '0x1.59e6cc410e9bbp+7',
     ObjectId('55c6fe30c37a457d27712c66'): '0x1.4e97b8d1e6c22p+7',
     ObjectId('55c6fe30c37a457d27712c67'): '0x1.3dc1a0addea92p+7',
     ObjectId('55c6fe30c37a457d27712c68'): '0x1.5774d5e826277p+7',
     ObjectId('55c6fe30c37a457d27712c69'): '0x1.573bc837e5193p+7',
     ObjectId('55c6fe30c37a457d27712c6a'): '0x1.53c5d0b35e472p+7',
     ObjectId('55c6fe30c37a457d27712c6b'): '0x1.576813f5e5f3cp+7',
     ObjectId('55c6fe30c37a457d27712c6c'): '0x1.56fa2aa6f3955p+7',
     ObjectId('55c6fe30c37a457d27712c6d'): '0x1.3ce9cfe2b7315p+7',
     ObjectId('55c6fe30c37a457d27712c6e'): '0x1.56856463a9042p+7',
     ObjectId('55c6fe30c37a457d27712c6f'): '0x1.5301b7f473ad8p+7',
     ObjectId('55c6fe30c37a457d27712c70'): '0x1.57bfdbed2f572p+7',
     ObjectId('55c6fe30c37a457d27712c71'): '0x1.584c08031281dp+7',
     ObjectId('55c6fe30c37a457d27712c72'): '0x1.59462e2fb0c60p+7',
     ObjectId('55c6fe30c37a457d27712c7d'): '0x1.55946eff475a7p+7',
     ObjectId('55c6fe30c37a457d27712c7e'): '0x1.537df7062bb29p+7',
     ObjectId('55c6fe30c37a457d27712c7f'): '0x1.5500b4d06884cp+7',
     ObjectId('55c6fe30c37a457d27712c80'): '0x1.3d43d8299e6bbp+7',
     ObjectId('55c6fe30c37a457d27712c81'): '0x1.55db248c9b922p+7',
     ObjectId('55c6fe30c37a457d27712c84'): '0x1.5482b682be93ep+7',
     ObjectId('55c6fe30c37a457d27712c85'): '0x1.3c78b9ad3d40dp+7',
     ObjectId('55c6fe30c37a457d27712c86'): '0x1.511fe96ac5423p+7',
     ObjectId('55c6fe30c37a457d27712c87'): '0x1.5908845842dabp+7',
     ObjectId('55c6fe30c37a457d27712c88'): '0x1.529750c11e9f7p+7',
     ObjectId('55c6fe31c37a457d27712c89'): '0x1.584dbdc899059p+7',
     ObjectId('55c6fe31c37a457d27712c8a'): '0x1.5a5b4a74f7dc0p+7',
     ObjectId('55c6fe31c37a457d27712c8c'): '0x1.4da33dcc0012ep+7',
     ObjectId('55c6fe31c37a457d27712c8d'): '0x1.54bd58b9583e0p+7',
     ObjectId('55c6fe31c37a457d27712c8e'): '0x1.54c74e5db2ba1p+7',
     ObjectId('55c6fe31c37a457d27712c8f'): '0x1.47b9d1cf75969p+7',
     ObjectId('55c6fe31c37a457d27712c92'): '0x1.545e668bfac15p+7',
     ObjectId('55c6fe31c37a457d27712ca8'): '0x1.55159b9c69615p+7',
     ObjectId('55c6fe31c37a457d27712ca9'): '0x1.55e56dbfcd902p+7',
     ObjectId('55c6fe31c37a457d27712cab'): '0x1.4f5171e9bb4fdp+7',
     ObjectId('55c6fe31c37a457d27712cbe'): '0x1.555b5130481efp+7',
     ObjectId('55c6fe31c37a457d27712cd3'): '0x1.4831ba77d5bd3p+7',
     ObjectId('55c6fe31c37a457d27712cd7'): '0x1.5289c55c6dd5fp+7',
     ObjectId('55c6fe32c37a457d27712cf0'): '0x1.548d330ea6b65p+7',
     ObjectId('55c6fe32c37a457d27712cfc'): '0x1.59d2abcefad8bp+7',
     ObjectId('55c6fe32c37a457d27712d04'): '0x1.56a7116722daap+7',
     ObjectId('55c6fe32c37a457d27712d09'): '0x1.4d0d50d357f8cp+7',
     ObjectId('55c6fe34c37a457d27712d6d'): '0x1.512ace8e6e71bp+7'}


class TestCrossValidation(unittest.TestCase):
    """Test the wrapper around sklearn.gridsearch.GridSearchCV.
    """

    # We have to sort a lot to ensure that reordered floating-point
    # operations don't cause us to accumulate rounding errors
    _CROSS_VALIDATION_CLASSIFICATION_Y_PRED_GOLDEN = np.array(
        CROSS_VALIDATION_CLASSIFICATION_Y_PRED_GOLDEN_RAW)
    _CROSS_VALIDATION_REGRESSION_Y_PRED_GOLDEN = np.array(
        [float.fromhex(f) for f in CROSS_VALIDATION_REGRESSION_Y_PRED_GOLDEN_RAW])

    def setUp(self):
        super(TestCrossValidation, self).setUp()
        with open(_TEST_DATA) as f:
            self._data_df = pickle.load(f)

    def testClassification(self):
        """Test the classification interface to the evaluation algorithm.
        """
        xy_generator = (
            lambda df:
            (
                pd.DataFrame(df['facepheno.height']),
                df['facepheno.Sex']
            )
        )
        kfolds = cv.KFoldPredefined(
            data=self._data_df,
            keep_in_holdout_columns=None,
            keep_together_columns=None)
        classifier = cv.CrossValidationClassification(xy_generator, kfolds)
        X_holdout, _ = kfolds.get_data_holdout(xy_generator)
        y_holdout_pred = classifier.grid.best_estimator_.predict(X_holdout)
        self.assertTrue(
            (self._CROSS_VALIDATION_CLASSIFICATION_Y_PRED_GOLDEN == y_holdout_pred).all())

    def _testRegression_predict(self, estimator=None):
        """
        """
        xy_generator = (
            lambda df:
            (
                df[['dynamic.FACE.genome.v1.pc1',
                    'dynamic.FACE.genome.v1.pc2',
                    'dynamic.FACE.genome.v1.pc3']],
                df['facepheno.height']
            )
        )
        kfolds = cv.KFoldPredefined(
            data=self._data_df,
            keep_in_holdout_columns=None,
            keep_together_columns=None)
        linear_model = cv.CrossValidationRegression(
            xy_generator,
            kfolds=kfolds,
            estimator=estimator)
        X_holdout, _ = kfolds.get_data_holdout(xy_generator)
        return linear_model.grid.best_estimator_.predict(X_holdout.values)

    def testRegressionHliRidge(self):
        """Test the regression interface to the cross-validation algorithm using the HLI ridge regression.
        """
        y_holdout_pred = self._testRegression_predict(
            estimator=hli_lm.Ridge())
        if len(y_holdout_pred.shape) == 2:
            assert y_holdout_pred.shape[1] == 1, "This functionality is only compatible with 1-D predictions, " \
                                                 "found %i." % y_holdout_pred.shape[
                                                     1]
            y_holdout_pred = y_holdout_pred[:, 0]
        self.assertTrue(
            np.allclose(self._CROSS_VALIDATION_REGRESSION_Y_PRED_GOLDEN, y_holdout_pred))

    def testRegressionSklearnRidge(self):
        """Test the regression interface to the cross-validation algorithm using the sk-learn ridge regression.
        """
        y_holdout_pred = self._testRegression_predict(
            estimator=sk_lm.Ridge())
        self.assertTrue(
            np.allclose(self._CROSS_VALIDATION_REGRESSION_Y_PRED_GOLDEN, y_holdout_pred))


class TestEvaluation(unittest.TestCase):
    """Test the underlying double-loop cross-validation evaluation
    algorithm for classification and regression machine learning
    problems.
    """

    # We have to sort a lot to ensure that reordered floating-point
    # operations don't cause us to accumulate rounding errors
    _EVALUATION_CLASSIFICATION_Y_PRED_GOLDEN = pd.Series(
        EVALUATION_CLASSIFICATION_Y_PRED_GOLDEN_RAW).sort_index(inplace=False)
    _EVALUATION_REGRESSION_Y_PRED_GOLDEN = pd.Series(
        EVALUATION_REGRESSION_Y_PRED_GOLDEN_RAW).sort_index(inplace=False)

    def setUp(self):
        super(TestEvaluation, self).setUp()
        with open(_TEST_DATA) as f:
            self._data_df = pickle.load(f)
        self._EVALUATION_REGRESSION_Y_PRED_GOLDEN = self._EVALUATION_REGRESSION_Y_PRED_GOLDEN.apply(
            lambda f: float.fromhex(f))

    def testClassification(self):
        """Test the classification interface to the evaluation algorithm.
        """
        xy_generator = (
            lambda df:
            (
                pd.DataFrame(df['facepheno.height']),
                df['facepheno.Sex']
            )
        )
        kfolds = cv.KFoldPredefined(
            data=self._data_df,
            keep_in_holdout_columns=None,
            keep_together_columns=None)
        evaluator = cv.EvaluationClassification(
            xy_generator=xy_generator,
            kfolds=kfolds)
        y_pred = evaluator.get_predicted()
        cols = list(y_pred.columns)
        self.assertTrue(len(cols) == 1)
        yseries = y_pred[cols[0]]
        self.assertTrue(
            (self._EVALUATION_CLASSIFICATION_Y_PRED_GOLDEN == yseries).all())

        self._commonTests(evaluator)

    def _testRegression_predict(self, estimator=None):
        xy_generator = (
            lambda df:
            (
                df[['dynamic.FACE.genome.v1.pc1',
                    'dynamic.FACE.genome.v1.pc2',
                    'dynamic.FACE.genome.v1.pc3']],
                df['facepheno.height']
            )
        )
        kfolds = cv.KFoldPredefined(
            data=self._data_df,
            keep_in_holdout_columns=None,
            keep_together_columns=None)
        evaluator = cv.EvaluationRegression(
            xy_generator=xy_generator,
            kfolds=kfolds,
            estimator=estimator)
        self._commonTests(evaluator)
        return evaluator.get_predicted()

    def testRegressionHliRidge(self):
        """Test the regression interface to the evaluation algorithm using the HLI ridge regression.
        """
        y_pred = self._testRegression_predict(estimator=hli_lm.Ridge())
        col = y_pred.columns[0]
        yser = pd.to_numeric(y_pred[col])
        self.assertTrue(
            np.allclose(self._EVALUATION_REGRESSION_Y_PRED_GOLDEN, yser))

    def testRegressionSklearnRidge(self):
        """Test the regression interface to the evaluation algorithm using the sk-learn ridge regression.
        """
        y_pred = self._testRegression_predict(estimator=sk_lm.Ridge())
        col = y_pred.columns[0]
        yser = pd.to_numeric(y_pred[col])
        self.assertTrue(
            np.allclose(self._EVALUATION_REGRESSION_Y_PRED_GOLDEN, yser))

    def _commonTests(self, evaluator):
        # test: predictions are returned in same order as input
        y_pred_all = evaluator.get_predicted(with_holdout=True)
        self.assertTrue(
            (y_pred_all.index == self._data_df.index).all())

        # test: get_predicted() with X_data param
        X_all, _ = evaluator.xy_generator(self._data_df)
        y_pred_data = evaluator.get_predicted(X_data=X_all)
        self.assertTrue(
            (y_pred_data.index == self._data_df.index).all())

if __name__ == '__main__':
    logging.basicConfig()
    unittest.main()
