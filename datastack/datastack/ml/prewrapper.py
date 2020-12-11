# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 15:25:14 2015

Copyright (C) 2015 Human Longevity Inc. Written by Peter Garst.

This wraps other estimators, and implements preprocessing before
passing data on to them.

To be specific, to work with cross validation we need to intercept fit, predict,
predict_proba and score, and preprocess the data appropriately before
sending it to the underlying estimator.
We also need set and get params to work with grid search, because they
need to handle the preprocessing specification as well.

We need to assume data frame input in order to properly match the 
preprocessing to the covariate.

When the user calls get_estimator we need to return an instance of this
class also, because the preprocessing needs to travel with the model.

This class essentially replaces the basic methods like fit in the underlying
class, but it may have other properties the user wants to keep.
So, we allow the user to get the underlying estimator and the preprocessor
specification and handle them directly.

We could try doing something fancier, like importing the methods and fields
from the wrapped estimator - essentially subclassing at run time - but that
gets very ugly and very brittle.
"""

import prespec
import pandas as pd
import numpy as np

class PreWrapper (object):
    
    def __init__(self, pp_estimator=None, pp_pspec=None, **kwargs):
        """
        est is an estimator consistent with scikit learn. It must take data
        frames.
        
        pspec is a dictionary which specifies preprocessing for each column.
        To be completely general, we'll accept None as a value here.
        
        This object wraps the estimator, but does appropriate preprocessing
        for each column before using fit, predict and so on.
        
        The active flag indicates whether the preprocessing has already 
        been done. In cross validation we want to do the preprocessing in an
        outer loop for efficiency.
        """
        self.active = True
        self.preproc = None
        self.estimator = pp_estimator
        self.pspec = pp_pspec
        
        if (not self.estimator is None) and hasattr(self.estimator, 'classes_'):
            self.classes_ = self.estimator.classes_
        
        """
        Do to poor design in this material, we are obliged to make this
        class match the old based only on keyword arguments.
        """
        self.set_params(**kwargs)
    
    def setActive (self, val):
        self.active = val
   
    def get_estimator (self):
        """
        Get the underlying estimator.
        The user should handle this carefully, as the input and output
        may be transformed.
        """
        return self.estimator
        
    def underlyingName (self):
        """
        Return the name of the underlying estimator for display purposes
        """
        return self.estimator.__class__.__name__
    
    def get_pspec (self):
        """
        Return the information the user needs to understand and use the
        preprocessing if they want to do that separately.
        """
        return self.preproc
        
    def fit (self, X, y, **kwargs):
        """
        Fit the preprocessing as well as the model.
        Some of the models allow sample_weight - do we need to worry
        about that?
        """
                
        if self.pspec is None:
            res = self.estimator.fit(X, y, **kwargs)
            return res
            
        if isinstance(y, pd.DataFrame):
            self.targets = list(y.columns)
        else:
            self.targets = [y.name]
            
        if isinstance(X, pd.DataFrame):
            cols = list(X.columns) + self.targets
        else:
            cols = [X.name] + self.targets
        
        # Resolve the preproc regular expressions and such
        if self.active:
            self.preproc = prespec.resolvePP(self.pspec, cols)
            
            prespec.fitPreproc(self.preproc, X)
            prespec.fitPreproc(self.preproc, y)
            
            X = prespec.runPreproc(self.preproc, X)
            y = prespec.runPreproc(self.preproc, y)
            
        res = self.estimator.fit(X, y, **kwargs)
        if hasattr(self.estimator, 'classes_'):
            self.classes_ = self.estimator.classes_
        return res
   
    def predict (self, X):
        """
        Run the preprocessing and then pass it on to the underlying estimator.
        When we get the results, we may have to unpreprocess it - for example,
        if we are producing standardized outputs we may need to reverse that.
        """
        if self.pspec is None:
            return self.estimator.predict(X)
      
        if self.active:
            X = prespec.runPreproc(self.preproc, X)
            
        Y = self.estimator.predict(X)
        if isinstance(Y, np.ndarray):
            if len(Y.shape) == 1:
                Y = pd.Series(Y)
                Y.name = self.targets[0]
            elif (len(Y.shape) > 1) and (Y.shape[1] == 1):
                Y = pd.Series(Y[:, 0])
                Y.name = self.targets[0]
            else:
                Y = pd.DataFrame(Y)
                Y.columns = self.targets
                Y['_nindex'] = X.index
                Y = Y.set_index('_nindex')
        
        # Inside here we need to know the preprocessing spec
        if self.active:
            Y = prespec.undoPreproc(self.preproc, Y, cols=self.targets)
        return Y
        
    def really_has_proba (self):
        """
        Other classes call hasattr on predict_proba to see if they should
        go down that path. But, if the child estimator does not have it, 
        disaster strikes.
        This method returns true if the child estimator has it.
        """
        if self.estimator is None:
            return False
        if hasattr(self.estimator, 'classes_'):
            self.classes_ = self.estimator.classes_
        return hasattr(self.estimator, 'predict_proba')
    
    def predict_proba (self, X):
        """
        What should we really return if the underlying estimator does not have
        this method?
        Big mess - other classes tend to use hasattr to see if they should
        call this - naughty-naughty.
        """
        if not hasattr(self.estimator, 'predict_proba'):
            return None
        if self.pspec is None:
            return self.estimator.predict_proba(X)
            
        if self.active:
            X = prespec.runPreproc(self.preproc, X)
        return self.estimator.predict_proba(X)
        
    def score (self, X, y):
        """
        Call the scoring function on the preprocessed data.
        """
        if self.pspec is None:
            return self.estimator.score(X, y)
        
        if self.active:
            X = prespec.runPreproc(self.preproc, X)
            y = prespec.runPreproc(self.preproc, y)
        return self.estimator.score(X, y)
        
    # get_params on new has to match get_params on old after constructor, 
    # passing in only keyword arguments
    def get_params (self, deep=True):
        """
        To successfully clone this, we need the params for the estimator
        as well as the local parameters.
        """
        params = self.estimator.get_params(deep=deep)
        params['pp_pspec'] = self.pspec
        params['pp_preproc'] = self.preproc
        params['pp_estimator'] = self.estimator
        params['pp_active'] = self.active
        return params
    
    def set_params (self, **params):
        """
        Need to set the estimator params as well as the local ones.
        """
        if 'pp_pspec' in params:
            self.pspec = params['pp_pspec']
            params.pop('pp_pspec', None)
        if 'pp_preproc' in params:
            self.preproc = params['pp_preproc']
            params.pop('pp_preproc', None)
        if 'pp_estimator' in params:
            self.estimator = params['pp_estimator']
            params.pop('pp_estimator', None)
        if 'pp_active' in params:
            self.active = params['pp_active']
            params.pop('pp_active', None)
        
        self.estimator.set_params(**params)
        if hasattr(self.estimator, 'classes_'):
            self.classes_ = self.estimator.classes_
        return self
        