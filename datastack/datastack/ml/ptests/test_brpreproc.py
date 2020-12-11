"""
Created on Wed Dec 16 08:59:02 2015

Copyright (C) 2015 Human Longevity Inc. Written by Peter Garst.

Test the preprocessing tools built into the base regression material.
This is called out of test_baseregress.
"""

import datastack.ml.baseregress as br
import datastack.ml.prespec as ps
import pandas as pd
import numpy as np

import keys

def _dropFrame (X):
    """
    Go to underlying matrix
    """
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        X = X.values
        
    sh = X.shape
    if len(sh) == 2 and sh[1] == 1:
        X = X[:,0]
    return X

def checkSame (X, est):
    """
    Predict in two ways - directly from X, and by pulling out the
    preprocessing and underlying estimator and running those.
    The results should be the same.
    """
    y0 = _dropFrame(est.predict(X))
    
    uest = est.get_estimator()
    pspec = est.get_pspec()
    X0 = ps.runPreproc(pspec, X)
    y1 = _dropFrame(uest.predict(X0))
    assert np.array_equal(y0, y1), '*** Unwinding preprocessing produces different results.'
  
def _testOne (df):
    """
    Run this frame with preprocessing "off".
    Get the estimator from CV, and extract the underlying estimator
    and preprocessing.
    Check that the preprocessing is dropping missing values.
    Check that we get the same prediction results split out and combined.
    """
    b = br.BaseRegress(keys.right)
    b.setDoPreproc(False)
    b.addData(df)
    covs = [keys.gender,keys.height]
    b.covariates['test'] = covs
    assert b.run(), '*** testOne test internal error'

    cv = b.cv[('test', keys.right, 0)]
    est = cv.get_estimator()
    pspec = est.get_pspec()
    
    # Do the test only on active columns - don't want to drop for 
    # something unused.
    allc = list(covs)
    allc.append(keys.right)
    dfx = df[allc]
    
    df0 = dfx.dropna()
    df1 = ps.runPreproc(pspec, dfx)
    assert df0.equals(df1), '*** Preprocessing off not dropping missing.'
    
    checkSame(df[covs], est)

def test_preproc (frame1):
    """
    Test many aspects of the default preprocessing.
    Well, maybe not so many yet. 
    The regular base regression test now uses default
    preprocessing, so that is also a test.
    """
    _testOne(frame1.copy())
    
if __name__ == "__main__":
    """
    This version you can invoke from an IDE for debugging purposes.
    """
    import conftest
    df = conftest.frame1(None)
    test_preproc(df)
