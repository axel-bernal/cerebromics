# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 09:11:25 2015

Copyright (C) 2015 Human Longevity Inc. Written by Peter Garst.
"""

import datastack.ml.features.mrmr.mrmr as mrmrw
import datastack.ml.algorithms.linreg as linreg
import sys

class MrSelect (object):
    """
    This is a feature selection wrapper for estimators.
    The base estimator is one that we might want to put into cross validation.
    This wrapper first does MrMr feature selection on the training set, 
    and then the tools can test it on the test set.
    """
    
    verbose = False
    
    def __init__ (self, est=None, core=None, ndim=None, **kwargs):
        """
        est is the underlying classifier. Default is Christoph's ridge regression.
        
        core is a core set of features which we always use.
        
        ndim is the target number of dimensions after feature selection, 
        in addition to the base features, if any. Can be 0 to verify that
        we get the proper baseline performance.
        """
        self.est = est
        if est is None:
            self.est = linreg.Ridge()
        self.core = core
        self.ndim = ndim
        self.set_params(**kwargs) 

        self.mrmr = mrmrw.MrMr()
        
    def fit (self, X, y):
        """
        First do feature selection to find the ndim best columns to
        use in the model; then fit the underlying estimator.
        """
        if self.verbose:
            print 'Start mrselect fit'
            sys.stdout.flush()
        self.columns = self.mrmr.selectCols(X, y, self.ndim, self.core)
        if self.verbose:
            # nc = set(self.columns) - set(self.core)
            # print 'Core columns', self.core
            print 'Selected columns', self.columns
            sys.stdout.flush()
        X0 = X[self.columns].dropna()
        
        res = self.est.fit(X0, y)
        if hasattr(self.est, 'classes_'):
            self.classes_ = self.est.classes_
        return res
        
    def predict (self, X):
        """
        Predict the X input data, after selecting the columns we want.
        """
        kframe = X[self.columns]      
        return self.est.predict(kframe)
  
    # Wrapping things that might or might not have predict_proba is a mess
        
    def score (self, X, y):
        """
        Return the mean accuracy, reported by the underlying model.
        """
        X0 = X[self.columns]
        return self.est.score(X0, y)
        
    def set_params (self, **params):
        """
        May want to pass parameters on to kest.
        Do we ever set our own parameters?
        """
        if 'mr_est' in params:
            self.est = params['mr_est']
            params.pop('mr_est', None)
        if 'mr_ndim' in params:
            self.ndim = params['mr_ndim']
            params.pop('mr_ndim', None)
        if 'mr_core' in params:
            self.core = params['mr_core']
            params.pop('mr_core', None)
            
        if not self.est is None:
            self.est.set_params(**params)
            if hasattr(self.est, 'classes_'):
                self.classes_ = self.est.classes_
                
        return self
        
    def get_params (self, deep=None):
        """
        Return mapping of param names to values.
        """
        params = self.est.get_params(deep=deep)
        params['mr_est'] = self.est
        params['mr_ndim'] = self.ndim
        params['mr_core'] = self.core
        return params
    