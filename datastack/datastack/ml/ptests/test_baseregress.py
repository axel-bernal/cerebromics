# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 18:01:20 2015

Copyright (C) 2015 Human Longevity Inc. Written by Peter Garst.
"""

import datastack.ml.baseregress as br
import datastack.ml.baseclassify as bc
import sklearn.metrics as metrics
import sklearn.linear_model as linear_model
import numpy as np
import datastack.utilities.typeutils as typeutils

# Names of important columns
import keys
import sys


def _testKey(base, key):
    """
    After running a test, we should have results information tied
    to each key.
    This method tests the information attached to a single key.
    """
    eval = base.evaluators[key]
    cv = base.cv[key]

    # Make sure the cv object exists
    cv.get_estimator()

    assert eval.r2 < 1.1, '*** Bad r2 value for key ' + str(key)


def test_basic(frame1):
    """
    Run a simple test - it makes sure we can connect to Rosetta, get data,
    and run a basic test with reasonable results.
    """
    base = br.BaseRegress(keys.right)
    base.addData(frame1)
    assert base.run(), '*** Basic test internal error'
    _testKey(base, ('Age', keys.right, 0))
    _testKey(base, ('Gender', keys.right, 0))
    _testKey(base, ('Ethnicity', keys.right, 0))

    # Test that we agglomerate as the default
    _testKey(base, ('AGE', keys.right, 0))


def _testCovarCheck(b):
    """
    We have a complex set of results coming out of the testCovar run.
    Check the keys for the results we expect, and check the data frame
    for the right structure.
    """
    cov = ['Age', 'Gender', 'Ethnicity', 'AGE', 'size', 'eyecolor', 'AGE + size', 'AGE + eyecolor']
    targ = ['right', 'left']
    for c in cov:
        for t in targ:
            _testKey(b, (c, t, 0))

    df = b.metrics_df
    cols = list(df.columns)
    errs = ['R2', 'MAE', 'MSE']
    for t in targ:
        for e in errs:
            probe = t + ': ' + e
            assert probe in cols, '*** Data frame column ' + probe + ' not found'

    clist = list(df['Covariates'])
    for c in cov:
        assert c in clist, '*** Data frame row for covariate ' + c + ' not found'


def test_covar(frame1):
    """
    This is the main test method.
    Specify a number of targets and a number of covariates.
    Pass in a number of data frames with required data.
    """
    # Set up two named targets
    targets = {'right': keys.right, 'left': keys.left}
    base2 = br.BaseRegress(targets)

    size = [keys.height, keys.bmi]

    base2.covariates['size'] = size
    base2.covariates['eyecolor'] = keys.eyes
    base2.addData(frame1)
    assert base2.run(), '*** testCovar internal error'

    _testCovarCheck(base2)


def test_agg(frame1):
    """
    Test use of aggregate covariates.
    """
    # Set up two named targets
    targets = {'right': keys.right, 'left': keys.left}
    base2 = br.BaseRegress(targets)

    size = [keys.height, keys.bmi]

    base2.covariates['size'] = size
    base2.covariates['eyecolor'] = keys.eyes
    base2.addData(frame1)
    assert base2.run(with_aggregate_covariates=False), '*** testAgg internal error'

    covs = list(base2.metrics_df['Covariates'])
    tage = ['AGE' in c for c in covs]
    assert not any(tage), '*** testAgg still have aggregates'


def _testEstCheck(b):
    """
    Check the results of the multiple estimators test.
    """
    checkEst(b, 'Ridge', 0)
    checkEst(b, 'Lasso', 1)


def checkEst(b, name, pos):
    key = ('Age', 'right', pos)
    cv = b.cv[key]
    est = cv.get_estimator()

    if hasattr(est, 'underlyingName'):
        tname = est.underlyingName()
    else:
        tname = str(type(est))

    assert name in tname, '*** Missing estimator ' + name + ' at position ' + str(pos)


def test_est(frame1):
    """
    Test specification of estimators.
    Specify several estimators for one covariate, and make sure we get
    the properly labeled rows in the output.
    This test isn't as thorough as it could be - if we passed in custom
    estimators we could check that it is actually calling the right
    estimators.
    """
    targets = {'right': keys.right, 'left': keys.left}
    base = br.BaseRegress(targets)
    base.addData(frame1)

    params = {'alpha': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]}
    est1 = {'est': linear_model.Ridge(), 'params': params}
    est2 = {'est': linear_model.Lasso(), 'params': params}
    base.estimatorsToRun['Age'] = [est1, est2]
    assert base.run(), '*** testEst internal error'

    _testEstCheck(base)


def test_wild(frame1):
    """
    Covariates should accept wildcard specifications.
    """
    base = br.BaseRegress(keys.height)
    base.addData(frame1)
    base.covariates['noreg'] = [keys.right, keys.left]
    base.covariates['reg1'] = [keys.wcard1]
    base.covariates['reg2'] = [keys.wcard2]
    assert base.run(), '*** Wild test internal error'

    ev0 = base.evaluators[('noreg', keys.height, 0)].r2
    ev1 = base.evaluators[('reg1', keys.height, 0)].r2
    ev2 = base.evaluators[('reg2', keys.height, 0)].r2
    diff1 = abs(ev0 - ev1)
    assert diff1 <= 0.001, '*** First wildcard test fails'
    diff2 = abs(ev0 - ev2)
    assert diff2 <= 0.001, '*** Second wildcard test fails'


def test_idx(frame1):
    """
    We can use idx_covariates to specify a subset of columns which
    should not be subject to regularization penalties.
    This is just a smoke test, to make sure it doesn't fail
    spectacularly.
    """
    base = br.BaseRegress(keys.right)
    base.addData(frame1)

    # Should be a better way to do this, but the covariate is not set up
    # in base until we run it.
    base.setIdxCovariates([keys.gender])

    assert base.run(), '*** Idx test internal error'
    _testKey(base, ('Age', keys.right, 0))
    _testKey(base, ('Gender', keys.right, 0))
    _testKey(base, ('Ethnicity', keys.right, 0))

    # Test that we agglomerate as the default
    _testKey(base, ('AGE', keys.right, 0))


def test_rows(frame1):
    size = [keys.height, keys.bmi]
    eyecolor = keys.eyes

    targets = {'right': keys.right, 'left': keys.left}
    base2 = br.BaseRegress(targets)
    base2.addData(frame1)
    base2.covariates['size'] = size
    base2.covariates['eyecolor'] = eyecolor
    base2.run(run_keys=['AGE + size', 'eyecolor'])

    n = len(base2.metrics_df)
    assert n == 2, '*** Restricted rows, got ' + str(n)


def test_classify(frame1):
    basec = bc.BaseClassify(keys.health, nopreproc=True)
    basec.addData(frame1)
    assert basec.run(), '*** Classify internal error'

# Just make sure it doesn't crash


def test_extra(frame1):
    targets = {'right': 'facepheno.hand.strength.right.m1', 'left': 'facepheno.hand.strength.left.m1'}
    size = ['facepheno.height', 'dynamic.FACE.pheno.v1.bmi']
    eyecolor = ['dynamic.FACE.neyecolor.v1_visit1.*']
    base = br.BaseRegress(targets)
    base.addData(frame1)

    base.extracols['median'] = lambda cv: metrics.median_absolute_error(cv.y, cv.get_predicted())

    base.covariates['size'] = size
    base.covariates['eyecolor'] = eyecolor
    base.run(with_aggregate_covariates=False)
    base.display()


def test_std(frame1):
    targets = {'right': 'facepheno.hand.strength.right.m1', 'left': 'facepheno.hand.strength.left.m1'}
    size = ['facepheno.height', 'dynamic.FACE.pheno.v1.bmi']
    eyecolor = ['dynamic.FACE.neyecolor.v1_visit1.*']
    base = br.BaseRegress(targets)
    base.addData(frame1)

    base.addCol('STD(R2)')

    base.covariates['size'] = size
    base.covariates['eyecolor'] = eyecolor
    base.run(with_aggregate_covariates=False)
    base.display()


def test_holdout(frame1):
    base = br.BaseRegress(keys.right)
    base.addData(frame1)
    base.run(with_bootstrap=False)

    cv_object = base.cv['AGE', keys.right, 0]
    output = cv_object.get_predicted(with_holdout=True)
    output = typeutils.makeList(output)
    res = [np.isfinite(x) for x in output]
    assert all(res), '*** Bad prediction on holdout set'

if __name__ == "__main__":
    import conftest
    df = conftest.frame1(None)

    test_holdout(df)

    print 'Test rows in run'
    test_rows(df)

    print 'Test a classification problem'
    test_classify(df)

    print 'Test std column'
    test_std(df)

    print 'Test use of idx_covariates'
    test_idx(df)

    print 'Holdout test'
    test_holdout(df)

    print 'Test extra error columns'
    test_extra(df)

    print 'Test basic regression of data in Rosetta'
    test_basic(df)

    print 'Test specification of estimators'
    test_est(df)

    print 'Test specification of targets and covariates'
    test_covar(df)

    print 'Test with_aggregate_covariates'
    test_agg(df)

    print 'Test wildcards'
    test_wild(df)
