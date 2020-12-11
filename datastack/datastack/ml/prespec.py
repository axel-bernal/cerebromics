# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 09:05:52 2015

Copyright (C) 2015 Human Longevity Inc. Written by Peter Garst.

This class specifies the preprocessing options to apply to data.
It is used in _BaseRegress, where we can attach an instance of this object
to each column to specify how it should be handled.
"""

import numpy as np
from datastack.tools.vdbwrapper import MissingOptions
from outliers import Outliers
from dataset import DataSet
from scipy import stats
import pandas as pd
import numbers
import re

class PreSpec (object):
    """
    Specify preprocessing for one dimension. In the future we may do things
    in multiple dimensions.
    Depending on the parameters, we may need to fit this.
    Currently the input X is always a series, which is what we get by
    extracting a column from a frame.
    """
    
    def __init__ (self, missing=MissingOptions.mean, stand=True, ostd = 4):
        """
        For the first pass, we specify preprocessing as follows:
    
        Missing data - mean, mode, drop - default is mean
        Standardize - yes or no - default is yes
        Outlier std - default is 4, use < 0 to turn off
        """
        self.missing = missing
        self.stand = stand
        self.ostd = ostd
        
        # Values we may fit depending on the above options
        self.mean = None
        self.mode = None
        self.std = None
        
    def clone (self):
        """
        When we clone we can drop the fitted material, since we are 
        generating this for a new variable.
        """
        return PreSpec(missing=self.missing, stand=self.stand, ostd=self.ostd)
    
    def runPreproc (self, X):
        """
        Run the preprocessing on one column.
        If we are dropping missing values, we have already done it.
        This only applies to numeric columns - we may get things like sample
        keys in here, and we want to ignore those.
        """
        if not self._isNumeric(X):
            return X
        X = self._doMissing(X)
        X = self._doClip(X)
        X = self._doStandardize(X)
        return X
    
    def _isNumeric (self, X):
        x = X.values[0]
        return isinstance(x, numbers.Number)
    
    def _doMissing (self, X):
        """
        We have already handled the drop case externally, because we can't
        do that a columns at a time.
        """
        if self.missing == MissingOptions.mean:
            X = X.fillna(self.mean)
        elif self.missing == MissingOptions.mode:
            X = X.fillna(self.mode)
        return X
    
    def _doClip (self, X):
        """
        X is a series, but we need a data frame.
        """
        if self.ostd <= 0:
            return X
            
        out = Outliers(stddevs = self.ostd)
        XX = pd.DataFrame(X)
        out.check(XX, XX, list(XX.columns))
        return XX[XX.columns[0]]
        
    def _doStandardize (self, X):
        if not self.stand:
            return X
            
        X = (X - self.mean) / self.std
        return X
        
    def undoPreproc (self, X):
        if not self._isNumeric(X):
            return X
        if not self.stand:
            return X
            
        X = (X * self.std) + self.mean
        return X
        
    def fitPreproc (self, X):
        if not self._isNumeric(X):
            return
        X = X.values
        
        # Mode is very expensive - we should only do it when we need to
        if self.missing == MissingOptions.mode:
            self.mode = stats.mode(X)
        
        self.mean = np.nanmean(X)
        self.std = np.nanstd(X)


stdPreproc = PreSpec()
"""
This is the standard preprocessing specification. It is applied to all data
in BaseRegress for which the user did not specify something different,
and you can use it in other contexts.
"""

dropnaPreproc = PreSpec(missing=MissingOptions.drop, stand=False, ostd=-1)
"""
Only drop missing values. This is our previous standard preprocessing.
"""

"""
In operation we usually have a dataframe and a dictionary with preprocessing
specified for various columns. These functions break that down into the individual
variables.
"""
   
def _settleOne (spec, allvars, pdict):
    """
    We have one specification. Find all columns to which it applies,
    and put in the preprocessing spec. When we do we want to clone so
    we can fit them independently.
    It is normal to replace previous specifications.
    """
    cols, pp = spec
    if type(cols) is list:
        for c in cols:
            _settleOne((c, pp), vars, pdict)
        return
    
    # We have a string, may be a regex
    pat = re.compile(cols)
    fcols = [d for d in allvars if pat.search(d)]
    for fc in fcols:
        pdict[fc] = pp.clone()     
    
def resolvePP (pspec, cols):
    """
    We have the preprocessing spec in the form of a list, where each
    element looks like (spec, pp). spec can be a regular expression, list of
    expressions, etc, and pp is the preprocessing we want to apply.
    They don't have to be disjoint - we can do one for everything, and then
    add more specific specifications.
    
    cols is the list of column names for which we need to know preprocessing.
    Return a dictionary mapping each column name to the preprocessing spec.
    """
    pdict = {}
    for spec in pspec:
        _settleOne(spec, cols, pdict)
    return pdict
    
def _doMissing (X, cols, pspec):
    """
    We may want to do some preprocessing on the frame as a whole
    before looking at individual columns. Currently, dropping missing
    values falls into this category - if we have that specified for
    a column, then we want to drop all columns in rows where it is
    missing.
    """
    dropc = [c for c in cols if c in pspec and pspec[c].missing == MissingOptions.drop]
    if len(dropc) == 0:
        return X
    if isinstance(X, pd.DataFrame):
        return X.dropna(subset=dropc)
    return X.dropna()
    
def runPreproc (pspec, X):
    if pspec is None:
        return X
        
    iss = False
    if isinstance(X, pd.DataFrame):
        cols = list(X.columns)
    else:   # Assume series
        cols = [X.name]
        iss = True
        
    X0 = X.copy()
    X0 = _doMissing(X0, cols, pspec)
   
    for c in cols:
        if c in pspec:
            if iss:
                X0 = pspec[c].runPreproc(X0)
            else:
                X0[c] = pspec[c].runPreproc(X0[c])

    return X0
    
def undoPreproc (pspec, X, cols=None):
    if pspec is None:
        return X
        
    iss = isinstance(X, pd.Series)
    if cols is None:
        if iss:
            cols = [X.name]           
        else:   
            cols = list(X.columns)
 
    X0 = X.copy()
    for c in cols:
        if c in pspec:
            if iss:
                X0 = pspec[c].undoPreproc(X0)
            else:
                X0[c] = pspec[c].undoPreproc(X0[c])
    return X0

def fitPreproc (pspec, X):
    if pspec is None:
        return
        
    iss = False
    if isinstance(X, pd.DataFrame):
        cols = list(X.columns)
    else:   # Assume series
        iss = True
        cols = [X.name]
        
    for c in cols:
        if c in pspec:
            XX = X
            if not iss:
                XX = X[c]
            pspec[c].fitPreproc(XX)
            
def transData (pspec, ds):
    """
    We have x and y test and training data.
    Fit preprocessing on the training data, and apply it to all the data.
    Return a new data set with preprocessed data.
    """
    fitPreproc(pspec, ds.X_training)
    fitPreproc(pspec, ds.y_training)
    
    xtr = runPreproc(pspec, ds.X_training)
    xts = runPreproc(pspec, ds.X_test)
    ytr = runPreproc(pspec, ds.y_training)
    yts = runPreproc(pspec, ds.y_test)
    
    dsp = DataSet(xtr, xts, ytr, yts)
    return dsp