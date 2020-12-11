# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 08:59:02 2015

Copyright (C) 2015 Human Longevity Inc. Written by Peter Garst.

Test the preprocessing tools built into the base regression material.
This is called out of test_baseregress.
"""

import datastack.common.settings as settings
import datastack.face as face
import datastack.ml.baseregress as br
import datastack.ml.prespec as ps
import pandas as pd
import numpy as np

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
    if not np.array_equal(y0, y1):
        print '*** Unwinding preprocessing produces different results.'
        print np.array_str(y0)
        print np.array_str(y1)

def testOne (df):
    """
    Run this frame with preprocessing "off".
    Get the estimator from CV, and extract the underlying estimator
    and preprocessing.
    Check that the preprocessing is dropping missing values.
    Check that we get the same prediction results split out and combined.
    """
    b = br.BaseRegress('strength')
    b.setDoPreproc(False)
    b.addData(df)
    covs = ['gender','height']
    b.covariates['test'] = covs
    if not b.run():
        print '*** testOne test internal error'
        return
    cv = b.cv[('test', 'strength', 0)]
    est = cv.get_estimator()
    pspec = est.get_pspec()
    
    # Check that pspec drops missing values
    df0 = df.dropna()
    df1 = ps.runPreproc(pspec, df)
    if not df0.equals(df1):
        print '*** Preprocessing off not dropping missing.'
    
    # Check same results combined or split
    checkSame(df[covs], est)

def testPreproc ():
    """
    Test many aspects of the default preprocessing.
    Well, maybe not so many yet. 
    The regular base regression test now uses default
    preprocessing, so that is also a test.
    """
    df = pd.read_csv('testpp.csv')
    # The original dataframe doesn't have consented or related information,
    # which causes heartburn within the base regression library.
    df[face.FACE_CONSENTED_COLUMN] = False
    df[face.FACE_RELATED_COLUMN] = [[] for _ in range(0, df.shape[0])]

    testOne(df.copy())
    
if __name__ == "__main__":
    testPreproc()
