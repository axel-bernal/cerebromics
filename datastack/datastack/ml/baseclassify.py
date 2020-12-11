# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 06:49:11 2015

Copyright (C) 2015 Human Longevity Inc. Written by Peter Garst.

Provide a simple interface to baseline regression material.
"""

import baselines
import datastack.ml.cross_validation as cross_validation
import logging
import pandas as pd
import datastack.settings as settings

import baseregress

_logger = logging.getLogger(__name__)

pd.options.mode.chained_assignment = None

def setDebugLevel(level):
    baseregress.setDebugLevel(level)
    _logger.setLevel(level)

class BaseClassify (baseregress._BaseRegress):
    """
    Provide one line (well, maybe three) evaluation of classification problems.
    
    The simplest use case is: instantiate this object with one or more
    target variables; call run and then display to show a basic classification
    table.
    """
    
    def __init__ (self, targets, ordinals = None, covariates=None, dataframe=None, nopreproc=False, 
                  rversion=settings.ROSETTA_VERSION, namespace=settings.ROSETTA_NAMESPACE,
                  use_predicted_baseline_covariates=baseregress.BL_PREDICTED):
        """
        targets is the required input.
        It can be one value, or a list of values, or a dictionary with
        names for the targets. 
        
        The thing that is different than regression is that we can supply
        ordinal maps - that is, maps from categorical to ordinal values.
        We can optionally supply a map for each target variable.
        
        If we just have one target, we can supply one ordinal map.
        Or, we can supply a dictionary with entries like {t : map} where t
        is the name of one of the targets and map is the map for that target.
        If you assign the field later, it has to be in the dictionary form.

        By default use genetically predicted age and gender among
        baseline covariates; set , use_predicted_baseline_covariates to
        False to use subject-reported age and gender.
        """    
        self._proctype = 'Classify'
        super(BaseClassify, self).__init__(targets, covariates, dataframe, nopreproc=nopreproc, rversion=rversion, namespace=namespace, use_predicted_baseline_covariates=use_predicted_baseline_covariates)
         
        if ordinals is None:
            self.ordinals = {}
        elif len(targets) == 1:
            key = self.targets.keys()[0]
            if key in ordinals:
                self.ordinals = ordinals
            else:
                self.ordinals = {key : ordinals}
        else:
            self.ordinals = ordinals

    def _runSlice (self, key, targ, tord, fit_params=None, covariate_widget=None, kfold_widget=None, *args, **kwargs):
        """
        Run the regression tests for a particular target and covariate set.
        """
        tval = self.targets[targ]
        if key in self.estimatorsToRun:
            elist = self.estimatorsToRun[key]
            eind = 0
            for est in elist:
                params = None
                if 'params' in est:
                    params = est['params']
                e = baselines._evaluate_classification(
                    tval,
                    self.kfolds,
                    ordinals = tord,
                    nopreproc=self.nopreproc,
                    covariates=self.covariates[key],
                    estimator = est['est'],
                    params = params,
                    fit_params = fit_params, 
                    pspec=self._preproc,
                    covariate_widget=covariate_widget,
                    kfold_widget=kfold_widget,
                    *args,
                    **kwargs);
                
                # Use the name of the covariates rather than the value
                e.metrics_df.Covariates = e.metrics_df.Covariates.apply(lambda c: key)
                self.evaluators[(key, targ, eind)] = e
                eind = eind + 1
        else:      
            e = baselines._evaluate_classification(
                tval,
                self.kfolds,
                ordinals = tord,
                nopreproc=self.nopreproc,
                covariates=self.covariates[key],
                fit_params = fit_params, 
                pspec=self._preproc,
                covariate_widget=covariate_widget,
                kfold_widget=kfold_widget,
                *args,
                **kwargs);
            
            # Use the name of the covariates rather than the value
            e.metrics_df.Covariates = e.metrics_df.Covariates.apply(lambda c: key)
            self.evaluators[(key, targ, 0)] = e
        
    def run (
             self,
             rows=None,
             with_aggregate_covariates=True,
             with_bootstrap=True,
             kfargs=None,
             fit_params=None,
             outlier_filter=None,
             covariate_widget=None,
             kfold_widget=None,
             *args,
             **kwargs
             ):
        """
        Run the regression problems we have specified.
        
        If you specify a covariate set or list of covariate sets in the rows parameter,
        we will run only those rows.
          
        If with_aggregate_covariates is True, the default, the tools run your covariate sets 
        as you specify them and also with the age, gender and ethnicity information added.
        
        kfargs is an optional dictionary of arguments for the KFoldPredefined method.
           
        fit_params is an optional dictionary of arguments for the underlying fit method.
         
        covariate_widget and kfold_widget are optional UI elements for showing progress.
        
        Variable arguments are passed down to the estimators.
        """
        self._addStandardCovars()
        self._findAllData()
        
        # May want to aggregate our test features with the basic features      
        self.aggregate_covariate_label = ''.join([c[0] for c in self.basic])
        if with_aggregate_covariates:
            self.getAggregates()

        # Currently drop missing data - should make this more flexible.
        # But, be careful to drop missing data only for the columns we need for this
        # particular test.
        # Does this column selection make a copy even if it is all columns?
        testframe = self.dataframe[self.allvars]        
        testframe = self.dropCovars(rows, testframe)
        testframe = testframe.dropna()
        if outlier_filter is not None and not self.multi:
            outlier_filter.check(testframe, testframe, self.targets.values())

        if kfargs is None:
            self.kfolds = cross_validation.KFoldPredefined(data=testframe)
        else:
            self.kfolds = cross_validation.KFoldPredefined(data=testframe, **kfargs)
                     
        # We might want to access this outside
        self.testframe = testframe
                      
        # Figure out preprocessing for all columns
        self._settlePreproc()
        
        # If we have a rows specification, we want to do only those.
        # Assume baseline is not there.
        # No regular expressions.
        if (not rows is None) and (not type(rows) is list):
            rows = [rows]
        
        self.baseline = {}
        for t in self.targets: 
            tord = None
            if t in self.ordinals:
                tord = self.ordinals[t]
            if rows is None:
                _logger.info('Evaluating baseline models for target %s...' % (t))
                self.baseline[t] = cross_validation.EvaluationClassificationBaseline(self.targets[t], self.kfolds, ordinals=tord)
            _logger.info('Evaluating classification models for target %s...' % (t))
            for key in self.covariates:
                if (rows is None) or (key in rows):
                    self._runSlice(key, t, tord, fit_params, covariate_widget, kfold_widget, *args, **kwargs)             
          
        # get CV objects also
        self._getCV()
        self._tabulate_metrics(self.baseline)

        # If bootstrap requested, run our '**' estimator to identify
        # significant covariates
        if with_bootstrap == True:
            _logger.info('Bootstrap aggregating...')
            self._bootstrap(self.baseline, with_aggregate_covariates=with_aggregate_covariates)
            
        return True

# For debugging
if __name__ == "__main__":
    health = 'facepheno.health.status'
    basec = BaseClassify(health, nopreproc=True)
    basec.run()
    cv = basec.cv[('Gender', health, 0)]
    pred = cv.get_predicted_proba()
    print pred.head()
