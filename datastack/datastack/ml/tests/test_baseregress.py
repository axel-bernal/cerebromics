# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 18:01:20 2015

Copyright (C) 2015 Human Longevity Inc. Written by Peter Garst.
"""

import datastack.dbs.rdb
import datastack.ml.baseregress as br
import datastack.ml.baseclassify as bc
from test_brpreproc import testPreproc
import sklearn.metrics as metrics
import numpy as np
import datastack.utilities.typeutils as typeutils

# Names of important columns
import cols

rdb = None

def testPrep ():
    """
    Set up resources for the tests
    """
    try:
        global rdb
        rdb = datastack.dbs.rdb.RosettaDBMongo()
        rdb.initialize(namespace='hg19')
    except:
        print '*** Unable to open Rosetta'
        return False
    return True

def testKey (base, key):
    """
    After running a test, we should have results information tied
    to each key. 
    This method tests the information attached to a single key.
    Use print rather than assert because we don't want to stop the
    test on the first failure.
    """
    try:
        eval = base.evaluators[key]
        cv = base.cv[key]
        
        # Make sure the cv object exists
        cv.get_estimator()
    except:
        print '*** Unable to access results for key ', str(key)
        return
    
    if not eval.r2 < 1.1:
        print '*** Bad r2 value for key ', str(key)

def testBasic ():
    """
    Run a simple test - it makes sure we can connect to Rosetta, get data,
    and run a basic test with reasonable results.
    """
    base = br.BaseRegress(cols.right_key)
    if not base.run():
        print '*** Basic test internal error'
        return
    testKey(base, ('Age', cols.right_key, 0))
    testKey(base, ('Gender', cols.right_key, 0))
    testKey(base, ('Ethnicity', cols.right_key, 0))
    
    # Test that we agglomerate as the default
    testKey(base, ('AGE', cols.right_key, 0))
    
def testCovarCheck (b):
    """
    We have a complex set of results coming out of the testCovar run.
    Check the keys for the results we expect, and check the data frame
    for the right structure.
    """
    cov = ['Age', 'Gender', 'Ethnicity', 'AGE', 'size', 'eyecolor', 'AGE + size', 'AGE + eyecolor']
    targ = ['right', 'left']
    for c in cov:
        for t in targ:
            testKey(b, (c, t, 0))
    
    df = b.metrics_df
    cols = list(df.columns)
    errs = ['R2', 'MAE', 'MSE']
    for t in targ:
        for e in errs:
            probe = t + ': ' + e
            if not probe in cols:
                print '*** Data frame column ', probe, ' not found'
    
    clist = list(df['Covariates'])
    for c in cov:
        if not c in clist:
            print '*** Data frame row for covariate ', c, ' not found'
 
def testCovar ():
    """
    This is the main test method.
    Specify a number of targets and a number of covariates.
    Pass in a number of data frames with required data.
    """
    # Set up two named targets
    targets = {'right':cols.right_key, 'left':cols.left_key}
    base2 = br.BaseRegress(targets)
    
    size = [cols.height_key, cols.bmi_key]
    col1 = [cols.key_key] + cols.eye_keys
    frame1 = rdb.query(col1)
    col2 = [cols.key_key] + size
    frame2 = rdb.query(col2)
    
    base2.covariates['size'] = size
    base2.covariates['eyecolor'] = cols.eye_keys
    base2.addData(frame1)
    base2.addData(frame2)
    if not base2.run():
        print '*** testCovar internal error'
        return
        
    testCovarCheck(base2)
 
def testAgg ():
    """
    This is the main test method.
    Specify a number of targets and a number of covariates.
    Pass in a number of data frames with required data.
    """
    # Set up two named targets
    targets = {'right':cols.right_key, 'left':cols.left_key}
    base2 = br.BaseRegress(targets)
    
    size = [cols.height_key, cols.bmi_key]
    col1 = [cols.key_key] + cols.eye_keys
    frame1 = rdb.query(col1)
    col2 = [cols.key_key] + size
    frame2 = rdb.query(col2)
    
    base2.covariates['size'] = size
    base2.covariates['eyecolor'] = cols.eye_keys
    base2.addData(frame1)
    base2.addData(frame2)
    if not base2.run(with_aggregate_covariates=False):
        print '*** testAgg internal error'
        return
        
    covs = list(base2.metrics_df['Covariates'])
    tage = ['AGE' in c for c in covs]
    if any(tage):
        print '*** testAgg still have aggregates'
    
def testEstCheck (b):
    """
    Check the results of the multiple estimators test.
    """
    checkEst(b, 'Ridge', 0)
    checkEst(b, 'Lasso', 1)
    
def checkEst (b, name, pos):
    key = ('Age', 'right', pos)
    cv = b.cv[key]
    est = cv.get_estimator()
    
    if hasattr(est, 'underlyingName'):
        tname = est.underlyingName()
    else:
        tname = str(type(est))
    
    if not name in tname:
        print '*** Missing estimator ', name, ' at position ', str(pos)
    
def testEst ():
    """
    Test specification of estimators.
    Specify several estimators for one covariate, and make sure we get
    the properly labeled rows in the output.
    This test isn't as thorough as it could be - if we passed in custom
    estimators we could check that it is actually calling the right
    estimators.
    """
    import sklearn.linear_model as linear_model
    targets = {'right':cols.right_key, 'left':cols.left_key}
    base = br.BaseRegress(targets)
    
    params = {'alpha': [ 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]}
    est1 = {'est' : linear_model.Ridge(), 'params' : params}
    est2 = {'est' : linear_model.Lasso(), 'params' : params}
    base.estimatorsToRun['Age'] = [est1, est2]
    if not base.run():
        print '*** testEst internal error'
        return
    
    testEstCheck(base)
    
def testWild ():
    """
    Covariates should accept wildcard specifications.
    """
    base = br.BaseRegress(cols.height_key)
    base.covariates['noreg'] = [cols.right_key, cols.left_key]
    base.covariates['reg1'] = [cols.wcard1]
    base.covariates['reg2'] = [cols.wcard2]
    if not base.run():
        print '*** Wild test internal error'
        return
    ev0 = base.evaluators[('noreg', cols.height_key, 0)].r2
    ev1 = base.evaluators[('reg1', cols.height_key, 0)].r2
    ev2 = base.evaluators[('reg2', cols.height_key, 0)].r2
    diff1 = abs(ev0 - ev1)
    if diff1 > 0.001:
        print '*** First wildcard test fails'
    diff2 = abs(ev0 - ev2)
    if diff2 > 0.001:
        print '*** Second wildcard test fails'
        
def testIdx ():
    """
    We can use idx_covariates to specify a subset of columns which
    should not be subject to regularization penalties.
    This is just a smoke test, to make sure it doesn't fail
    spectacularly.
    """
    base = br.BaseRegress(cols.right_key)
    
    # Should be a better way to do this, but the covariate is not set up
    # in base until we run it.
    base.setIdxCovariates([cols.gender_key])
    
    if not base.run():
        print '*** Idx test internal error'
        return
    testKey(base, ('Age', cols.right_key, 0))
    testKey(base, ('Gender', cols.right_key, 0))
    testKey(base, ('Ethnicity', cols.right_key, 0))
    
    # Test that we agglomerate as the default
    testKey(base, ('AGE', cols.right_key, 0))

def testRows ():
    size = ['facepheno.height', 'dynamic.FACE.pheno.v1.bmi']
    eyecolor = ['dynamic.FACE.eyecolor.v1_visit1.*']
    
    targets = {'right':cols.right_key, 'left':cols.left_key}
    base2 = br.BaseRegress(targets)
    base2.covariates['size'] = size
    base2.covariates['eyecolor'] = eyecolor
    base2.run(rows=['size', 'eyecolor'])
    
    n = len(base2.metrics_df)
    if n != 2:
        print '*** Restricted rows, got ', n

def testClassify ():
    basec = bc.BaseClassify('facepheno.health.status')
    if not basec.run():
        print '*** Classify internal error'
        return

# Just make sure it doesn't crash
def testExtra ():
    targets = {'right':'facepheno.hand.strength.right.m1', 'left':'facepheno.hand.strength.left.m1'}
    size = ['facepheno.height', 'dynamic.FACE.pheno.v1.bmi']
    eyecolor = ['dynamic.FACE.eyecolor.v1_visit1.*']
    base = br.BaseRegress(targets)
    
    base.extracols['median'] = lambda cv : metrics.median_absolute_error(cv.y, cv.get_predicted())
    
    base.covariates['size'] = size
    base.covariates['eyecolor'] = eyecolor
    base.run(with_aggregate_covariates=False)
    base.display()
    
def testStd ():
    targets = {'right':'facepheno.hand.strength.right.m1', 'left':'facepheno.hand.strength.left.m1'}
    size = ['facepheno.height', 'dynamic.FACE.pheno.v1.bmi']
    eyecolor = ['dynamic.FACE.eyecolor.v1_visit1.*']
    base = br.BaseRegress(targets)
    
    base.addCol('STD(R2)')
    
    base.covariates['size'] = size
    base.covariates['eyecolor'] = eyecolor
    base.run(with_aggregate_covariates=False)
    base.display()

def testHoldout ():    
    base = br.BaseRegress('facepheno.hand.strength.right.m1')
    base.run(with_bootstrap=False)
    
    cv_object= base.cv['AGE','facepheno.hand.strength.right.m1',0]
    output = cv_object.get_predicted(with_holdout=True)
    output = typeutils.makeList(output)
    res = [np.isfinite(x) for x in output]
    if not all(res):
        print '*** Bad prediction on holdout set'

if __name__ == "__main__":
    if not testPrep():
        print '*** Unable to set up test resources'
        exit
        
    print 'Holdout test'
    testHoldout()
    
    print 'Test std column'
    testStd()
    
    print 'Test extra error columns'
    testExtra()
               
    print 'Test basic regression of data in Rosetta'
    testBasic()
    
    print 'Test a classification problem'
    testClassify()   
        
    print 'Test rows in run'
    testRows()
          
    print 'Test specification of estimators'
    testEst()
 
    print 'Test use of idx_covariates'
    testIdx()
    
    print 'Test preprocessing'
    testPreproc()
    
    print 'Test specification of targets and covariates'
    testCovar()
 
    print 'Test with_aggregate_covariates'
    testAgg()

    print 'Test wildcards'
    testWild()

    print'BaseRegress tests finished'
