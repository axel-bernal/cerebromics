# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 12:05:07 2015

Copyright (C) 2015 Human Longevity Inc. Written by Peter Garst.

The evaluation and cross validation classes are largely intertwined.
This unified base class allows us to handle their methods properly.
For example, if a user calls get_predicted on the cross validation
class, it really belongs in the evaluation class to avoid sharing
information through hyperparameters.

The reason for this structure to to provide backward compatibility.
This material is widely used in the group, and as long as people are using
the public interfaces we want to guarantee that their notebooks will run
without modification, although the results might change a bit if the
corrections give more solid results for their data.

In the previous version the two Evaluation classes derived from _Evaluation,
and the two CV classes from _CrossValidation. Now they all derive from
the _Unified class, although there is some internal ugliness to support
what people expect from both paths.
"""

import cross_validation
import datastack.ml.algorithms.linreg as linreg
import inspect
import numpy as np
import pandas as pd
import json
import pickle
import sklearn
import os
import logging

import face as face


import warnings


from sklearn.grid_search import GridSearchCV
from scipy.stats import pearsonr

import prewrapper
import datastack.utilities.typeutils as typeutils
from dataset import DataSet
import prespec

# By default use our local Ridge implementation.
# _REGRESSION_ESTIMATOR_DEFAULT = sklearn.linear_model.Ridge()
_REGRESSION_ESTIMATOR_DEFAULT = linreg.Ridge()





class _Unified (object):
    """
    This class subsumes the old _CrossValidation and _Evaluation classes.
    There is some unpleasant code which does things one way in the CV case
    and another way in the eval case - that's the cost of maintaining the
    current UI, we could do it much more cleanly if we obsoleted a lot of stuff.
    """

    def __init__ (self,
                  xy_generator=None,
                  kfolds=None,
                  isRegression=True,
                  isCvobj = True,
                  estimator=None,
                  params=None,
                  fit_params=None,
                  idx_covariates=None,
                  dtype=None,
                  pspec=None,
                  nopreproc=False,
                  widget=None,
                  VERSION = None,
                  TYPE = None,
                  COVARIATES = None,
                  PREDICTED_COVARIATES = None,
                  HOLDOUT =False,
                  *args,
                  **kwargs):
        """
        Prepare to handle calls to both cv and eval objects.
        """
        self.nopreproc = nopreproc
        self.params = params

        self.VERSION = VERSION
        self.TYPE =TYPE
        self.COVARIATES = COVARIATES
        self.PREDICTED_COVARIATES = PREDICTED_COVARIATES
        self.HOLDOUT = HOLDOUT

        self.face_model = face._Face(VERSION = VERSION,TYPE =TYPE,COVARIATES = COVARIATES,PREDICTED_COVARIATES = PREDICTED_COVARIATES,HOLDOUT = HOLDOUT)

        if self.params is None:
            raise ValueError('Invalid parameter grid: %s' % self.params)

        self.isRegression = isRegression
        if isRegression:
            self.cross_validator = cross_validation.CrossValidationRegression
        else:
            self.cross_validator = cross_validation.CrossValidationClassification

        self.setKfolds(kfolds, xy_generator)

        self._idx_covariates = idx_covariates
        self._setEstimator(estimator, pspec=pspec)

        if widget is not None:
            widget.max = len(self.kfolds)
            widget.value = 0
        self.widget = widget


        self.args = args
        self.kwargs = kwargs
        self.fit_params = fit_params
        self.grid = None
        self.cv_all = None
        self.y_pred = None

        if isCvobj:
            if params is None:
                raise ValueError('Invalid parameter grid: %s' % params)
            self._setGrid(params, args, kwargs)
        else:
            self._evalInit(args, kwargs)

    def _setEstimator (self, estimator, pspec=None):
        """
        If the user supplied an estimator, check it; otherwise,
        instantiate the default.
        Currently on the classification path the level above passes in the
        default estimator if the user did not supply one, but on the
        regression path we get None, because we have to set up the idx covariate
        columns here before instantiating the default estimator.
        Otherwise, instantiate the default estimator.

        Call this after setting up the kfolds, which initialized self.X.
        """
        if not estimator is None:
            self.estimator = estimator
            if inspect.isclass(self.estimator):
                raise TypeError('Invalid estimator: Expected an instance of an estimator,',
                            'got a type of estimator (try calling the estimator class constructor')
            if not self.nopreproc:
                self.estimator = self._addPreproc(pspec)
            return

        # Set up the default
        if self._idx_covariates is None:
            self.estimator = _REGRESSION_ESTIMATOR_DEFAULT
        else:
            idx = list(self._idx_covariates)
            cols = self.X.columns
            idx = [cols.get_loc(c) for c in idx if c in cols]
            self.estimator = linreg.Ridge(idx_covariates=idx)

        if not self.nopreproc:
            self.estimator = self._addPreproc(pspec)

    def _addPreproc (self, pspec):
        """
        Add a preprocessing wrapper around the estimator if required.

        Be careful - the user could pass in an already wrapped class, and we
        might be wanting different preprocessing. In that case pull out the
        underlying estimtator, and wrap it in the current preprocessing.
        It gets messy in the cross validation, though - if pspec is none,
        just keep current preprocessing.
        """
        if isinstance(self.estimator, prewrapper.PreWrapper):
            if pspec is None:
                return self.estimator
            self.estimator = self.estimator.get_estimator()
            return self._addPreproc(pspec)

        # If pspec is none, still want this class with the None inside.
        return prewrapper.PreWrapper(self.estimator, pspec)

    def _setGrid (self, params, args, kwargs):
        """
        A key routine - the place where we actually create and fit the
        grid search object.
        """
        #self.estimator.setActive(True)
        self.grid = GridSearchCV(
            self.estimator,
            params,
            fit_params=self.fit_params,
            cv=self.kfolds,
            *args,
            **kwargs)
        # print 'Estimator', str(type(self.estimator))
        self.grid.fit(self.X, self.y)
        self.grid_params = [gs.parameters for gs in self.grid.grid_scores_]      

    def _evalInit (self, args, kwargs):
        """
        Initialization for the eval branch.
        We may call this later on the cv branch if we need to get predicted
        values or do other eval-type tasks.

        In order to get acceptable efficiency, do preprocessing in the outer
        loop here, and thus turn it off in the inner loop.
        """
        y=pd.DataFrame()
        if isinstance(self.y, pd.DataFrame):
            y=self.y
            y.index = self.X.index
        else:
            y = pd.DataFrame(self.y, index=self.X.index,columns=['y'])

        self.y_pred = pd.DataFrame(index=self.X.index,columns=y.columns)


        if self.COVARIATES is None:
            self.covariates = pd.DataFrame(index=self.X.index)
            self.pred_covariates = pd.DataFrame(index=self.X.index)
        else:
            self.covariates = pd.DataFrame(index=self.X.index,columns=self.COVARIATES)
            if self.PREDICTED_COVARIATES is not None:
                self.pred_covariates = pd.DataFrame(index=self.X.index,columns=self.PREDICTED_COVARIATES)
            else:
                self.pred_covariates = pd.DataFrame(index=self.X.index)


        self.cvfold = pd.DataFrame(index=self.X.index,columns=['fold'])
        self.IDs = pd.DataFrame(index=self.X.index,columns=['ds.index.sample_name','subject_id','PC1','PC2'])

        self.params_path = []
        self.score_training_path = []
        self.score_test_path = []
        self.test_size_test_path = []
        self.metrics_df = None

        doproba = hasattr(self.estimator, 'predict_proba')
        self.y_prob = None

        # We will do preprocessing in the outer loop
        if self.nopreproc:
            pspec = None
        else:
            self.estimator.setActive(False)
            pspec = self.estimator.pspec

        # keeping track of the number of the cross-validation fold
        idx_cv = 1

        for training_index, test_index in self.kfolds:
            new_training_index = self.X.index[training_index]
            new_test_index = self.X.index[test_index]

            X_training = self.X.loc[new_training_index]
            X_test = self.X.loc[new_test_index]

            self.ds_sample_key_training = list(self.kfolds.df.ix[new_training_index,'ds.index.sample_name'].values)
            self.ds_sample_key_test = list(self.kfolds.df.ix[new_test_index,'ds.index.sample_name'].values)

            if isinstance(self.y, pd.DataFrame):

                # over-write y_training and y_test with values from .csv Embedding for FACE if needed - to prevent overfitting
                if self.VERSION is not None and self.TYPE is not None:
                    res_frame,model_pcs=self.face_model._getSVDValuesPerFoldFace(idx_cv,self.y.shape[1])
                    
                    y_training = res_frame.loc[self.ds_sample_key_training,model_pcs]
                    y_test = res_frame.loc[self.ds_sample_key_test,model_pcs]
                    self.y.ix[new_test_index] = y_test.values
                    
                    self.IDs.ix[new_test_index] = res_frame.loc[self.ds_sample_key_test,['ds.index.sample_name','subject_id',model_pcs[0],model_pcs[1]]].values
                    idx_cv+=1
                else:
                    y_training=self.y.loc[new_training_index,:]
                    y_test=self.y.loc[new_test_index,:]
            else:
                y_training = self.y.iloc[training_index]
                y_test = self.y.iloc[test_index]

            ds = DataSet(X_training, X_test, y_training, y_test)
            dsp = prespec.transData(pspec, ds)

            kfolds_training = sklearn.cross_validation.KFold(len(dsp.X_training), n_folds=5)
            cv = self.cross_validator(
                (dsp.X_training, dsp.y_training),
                kfolds=kfolds_training,
                estimator=self.estimator,
                nopreproc=self.nopreproc,
                params=self.params,
                fit_params=self.fit_params,
                *args,
                **kwargs)
            model = cv.grid.best_estimator_

            if isinstance(self.y, pd.DataFrame):
                
                res = model.predict(dsp.X_test)
                
                res = prespec.undoPreproc(pspec, res)
                self.y_pred.ix[new_test_index] = res
                
                # if doing face run copy fold and covariates values
                if self.VERSION is not None and self.TYPE is not None:
                    self.cvfold.ix[new_test_index]=res_frame.loc[self.ds_sample_key_test,['fold']].values

                if self.PREDICTED_COVARIATES is not None:
                    self.pred_covariates.ix[new_test_index] = res_frame.loc[self.ds_sample_key_test,self.PREDICTED_COVARIATES].values
               

                if self.COVARIATES is not None:
                    self.covariates.ix[new_test_index] = res_frame.loc[self.ds_sample_key_test,self.COVARIATES].values

            else:
                res = model.predict(dsp.X_test)
                
                res = prespec.undoPreproc(pspec, res)
                res = typeutils.makeList(res)
                self.y_pred.ix[new_test_index] = pd.DataFrame(res, index=new_test_index,columns=['y'])

            if doproba:
                res = model.predict_proba(dsp.X_test)

                # We can get None if we are wrapping an estimator with no
                # such method. It might be better to remove the function.
                if not res is None:
                    if self.y_prob is None:
                        # print 'Classes', model.classes_
                        # print 'Shape', res.shape
                        if hasattr(model, 'classes_'):
                            self.y_prob = pd.DataFrame(
                                np.zeros((len(self.kfolds.df), res.shape[1])),
                                index=self.X.index, columns=model.classes_)
                        else:
                            self.y_prob = pd.DataFrame(
                                np.zeros((len(self.kfolds.df), res.shape[1])),
                                index=self.X.index)
                    
                    # The crappy scikit estimators have a bug for which they
                    # return different numbers of columns depending on the
                    # input data. We just catch it ang flag it here.
                    try:
                        if isinstance(self.y, pd.DataFrame):
                            self.y_prob.ix[new_test_index] = res
                        else:
                            fr = pd.DataFrame(res, index=new_test_index) # ,columns=['y'])
                            self.y_prob.ix[new_test_index] = fr     
                    except:
                        self.y_prob = None
                        warnings.warn('Logistic regression bugs in scikit trashed your proba prediction.', UserWarning)

            self._update_score_path(
                cv.grid.scorer_,
                model,
                X_training,
                X_test,
                y_training,
                y_test)
            if self.widget is not None:
                self.widget.value += 1   

        if not self.nopreproc:
            self.estimator.setActive(True)

        # also do a grid fit on the entire data minus holdout, we will use this for generating predictions
        self.cv_all = self.cross_validator(self.xy_generator, self.kfolds, estimator=self.estimator, nopreproc=self.nopreproc, params=self.params, fit_params=self.fit_params)

        # over-write here if FACE wo/ holdout all but holdout values or with holdout values depending on the HOLDOUT flag
        
        if self.VERSION is not None and self.TYPE is not None:
            res_frame,model_pcs = self.face_model._getSVDValuesFace(self.y.shape[1])
            new_index = self.X.index
            self.ds_sample_key_all = list(self.kfolds.df.ix[new_index,'ds.index.sample_name'].values)

            y_training = res_frame.loc[self.ds_sample_key_all,model_pcs]
            self.y.ix[new_index] = y_training.values

        # Redundant - called in cv constructor
        # self.cv_all.grid.fit(self.X, self.y)

        if isinstance(self.y, pd.DataFrame):
            self.y_pred_all = pd.DataFrame(index=self.kfolds._df.index) #index = input data index
        else:
            self.y_pred_all = pd.DataFrame(index=self.kfolds._df.index, columns=self.y_pred.columns) #index = input data index

        self.y_pred_all.ix[self.y_pred.index] = self.y_pred
        if self.X_holdout is not None and len(self.X_holdout) > 0:
            if isinstance(self.y, pd.DataFrame):
                self.y_pred_all.ix[self.X_holdout.index] = self.cv_all.grid.predict(self.X_holdout)
            else:
                res = self.cv_all.grid.predict(self.X_holdout)
                # Probably need to do something for multivariate case.
                # The popularity of python is an enduring mystery.
                res = pd.Series(data=list(res), index=self.X_holdout.index)
                res = pd.DataFrame(res, index=self.X_holdout.index,columns=['y'])
                self.y_pred_all.ix[self.X_holdout.index] = res

        if doproba and (not self.y_prob is None):
            #self.y_prob_all = pd.DataFrame(np.zeros((len(self.kfolds._df), self.y_prob.shape[1])),
                                           # index=self.kfolds._df.index) #index = input data index
            self.y_prob_all = self.y_prob.copy()
            # self.y_prob_all[self.y_prob.index] = self.y_prob
            if self.X_holdout is not None and len(self.X_holdout) > 0:
                # self.y_prob_all[self.X_holdout.index] = self.cv_all.grid.predict_proba(self.X_holdout)
                xtra = self.cv_all.grid.predict_proba(self.X_holdout)
                npd = pd.DataFrame(data=xtra, index=self.X_holdout.index)
                self.y_prob_all.append(npd)

        if hasattr(self, '_update_metrics'):
            self._update_metrics()

    def get_predicted(self, with_holdout=False, X_data=None):
        """
        Returns predictions on X_data if provided. Otherwise returns out-of-sample predictions on the kfolds data.
        @param X_data: data frame with covariates as columns
        @param with_holdout: if True, include predictions on holdout samples. Ignored when X_data is provided
        @return Series, indexed and sorted same as the input data
        """
        if self.cv_all is None:
            self._evalInit(self.args, self.kwargs)


        if X_data is not None: # get predictions on X_data
            y_pred_data = self.cv_all.grid.predict(X_data)
            return pd.DataFrame(index=X_data.index, data = y_pred_data)

        return self.y_pred_all if with_holdout == True else self.y_pred


    def get_predicted_proba(self, with_holdout=False, X_data=None):
        """
        We want predicted probabilities for the input data, but we want to do it
        in a safe manner where each predicted value is generated by a model
        which was trained on other data.

        **Returns**
        ***
        **y_probcv** Cross validated probability predictions from the training data. Sets
        this field for later reference as well.
        This returns a data frame - would we prefer an array?
        """
        if not hasattr(self.estimator, 'predict_proba'):
            raise TypeError('The estimator must provide a predict_proba method')

        if self.cv_all is None:
            self._evalInit(self.args, self.kwargs)

        if X_data is not None: # get predictions on X_data
            y_prob_data = self.cv_all.grid.predict_proba(X_data)
            return pd.DataFrame(index=X_data.index, data = y_prob_data)

        return self.y_prob_all if with_holdout == True else self.y_prob

    def save_best_estimator(self, filename, metadata=None):
        '''Save the best estimator from the grid fit across the entire
        dataset. The output file is a pickled dictionary.

        Args:
            filename: The name for the saved estimator file
            metadata: Arbitrary metadata to include with the model
        '''
        with open(filename, 'w') as f:
            container = {
                         'estimator' : self.cv_all.grid.best_estimator_,
                         'metadata' : metadata
                         }
            pickle.dump(container, f)

    def _update_score_path(
            self, scorer, model, X_training, X_test, y_training, y_test):
        """
        Old version uses ndarray

        if isinstance(self.y, pd.DataFrame):
            self.score_training_path.append(scorer(model, X_training.values, y_training.values))
            self.score_test_path.append(scorer(model, X_test.values, y_test.values))
            self.test_size_test_path.append(len(y_test.values))
        else:
        """
        self.score_training_path.append(scorer(model, X_training, y_training))
        self.score_test_path.append(scorer(model, X_test, y_test))
        self.test_size_test_path.append(len(y_test))

    def setKfolds(self, kfolds, xy_generator):
        self.kfolds = kfolds
        self.xy_generator = xy_generator
        if not issubclass(self.kfolds.__class__, sklearn.cross_validation._PartitionIterator):
            raise TypeError(
                'Invalid k-fold type: Expected a partition iterator, got %s' %
                self.kfolds.__class__)

        self.X_holdout = None
        self.y_holdout = None
        if hasattr(self.kfolds, 'get_data') and hasattr(self.xy_generator, '__call__'):
            self.X, self.y = self.kfolds.get_data(self.xy_generator)

            if hasattr(self.kfolds, 'get_data_holdout'):
                self.X_holdout, self.y_holdout = self.kfolds.get_data_holdout(self.xy_generator)
        elif type(self.xy_generator) in [list, tuple]:
            if (len(self.xy_generator) == 2) and (
                    len(self.xy_generator[0]) == len(self.xy_generator[1])):
                self.X, self.y = self.xy_generator
            else:
                raise ValueError("Expected (X, y) tuple with vectors of equal lengths")
        else:
            raise TypeError(
                'Invalid X,y type: Expected a callable (X,y) generator or (X, y) tuple, got %s' %
                type(self.xy_generator))

        if self.X.shape[0] != self.y.shape[0]:
            raise IndexError(
                'X and y matrices have incompatible dimensions: X is %s, y is %s' %
                (str(self.X.shape), str(self.y.shape)))

        """
        It is only now that we know which columns we want for idx_covariates.
        """


    def __repr__(self):
        if self.kfolds is None:
            return '%s on %s with GridSearchCV default cross-validation' % (
                self.__class__.__name__, self.estimator)
        else:
            return '%s on %s with %s' % (
                self.__class__.__name__, self.estimator, self.kfolds)

    def get_score_path(self):
        """Get the estimator prediction scores for each training and test
        split as a function of the candidate parameter set.

        Returns
            A list of scores for the training splits and a list of
        scores for the corresponding test splits, both as a function of
        the candidate parameter set
        """
        return self.score_training_path, self.score_test_path

    def get_r2(self, model, X, y):
        return self._get_r2(model.predict(X), y)

    @staticmethod
    def return_1d_array(a):
        if len(a.shape) > 1:
            if isinstance(a, np.ndarray):
                return a[:, 0]
            else:
                return a.iloc[:, 0]
        return a

    def _get_r2(self, v1, v2):
        v1 = self.return_1d_array(v1)
        v2 = self.return_1d_array(v2)

        tokeep = pd.Series((v1 == v1) & (v2 == v2)).values

        rval = pearsonr(v1[tokeep], v2[tokeep])[0]
        assert not np.isnan(rval), "Pearson R should not be null"
        return rval * rval

    def _plot_something_vs_params(self, axes, y_data, y_label, param, x_log10):
        if self.grid is None:
            self._setGrid(self.params, self.args, self.kwargs)

        if set([len(p.keys()) for p in self.grid_params]) == {1} and len(set([p.keys()[
                0] for p in self.grid_params])) == 1 and set([len(p.values()) for p in self.grid_params]) == {1}:
            # The convoluted condition above checks to see if the
            # parameters grid only tests one parameter, and that the
            # parameter is a single-valued element
            param = self.grid_params[0].keys()[0]
        x_best_index = self.grid_params.index(self.grid.best_params_)
        if param is not None:
            x_label = param
            x_data = [p[param] for p in self.grid_params]
            x_best = self.grid_params[x_best_index][param]
        else:
            x_label = 'Parameter set index'
            x_data = range(0, len(self.grid_params))
            x_best = x_best_index
        np.array(x_data)
        if x_log10 is True:
            x_label = 'log10(%s)' % x_label
            x_data = np.log10(x_data)
            x_best = np.log10(x_best)
        axes.plot(x_data, y_data, ':')
        y_data_mean = [y.mean() for y in y_data]
        axes.plot(
            x_data,
            y_data_mean,
            'k',
            label='Average across the k-folds',
            linewidth=2)
        axes.axvline(
            x_best,
            linestyle='--',
            color='k',
            label='%s: Best estimate' %
            (x_label))
        axes.legend(loc="best")
        axes.set_xlabel(x_label)
        axes.set_ylabel(y_label)
        axes.axis('tight')

    def get_df(self):
        """Get the dataframe underlying the k-fold split used to train
        the estimator
        """
        return self.kfolds.df.copy()

    def get_estimator(self, version='unknown'):
        """
        Get the refit estimator so we can save it, etc.

        We have to be careful how we handle preprocessing. The estimator
        we trained might be for standardized values, for example.
        """
        if self.grid is None:
            self._setGrid(self.params, self.args, self.kwargs)

        est = self.grid.best_estimator_
        if est is None:
            return None
        if not version is None:
            est.hli_version = version

        # Try to put the feature list in the model also
        try:
            feats = list(self.X.columns)
            est.hli_features = feats
        except:
            print 'Unable to append feature list'

        return est

    def get_estimator_name(self):
        """Get the name of the trained estimator
        """
        return self.estimator.__class__.__name__

    def _get_scores_vs_params(self):
        return [gs.cv_validation_scores for gs in self.grid.grid_scores_]

    def plot_scores_vs_params(self, axes, score_label='Score', param=None, x_log10=False):
        """Plot a graph of the score/loss function against each evaluated
        hyperparameter combination for estimator training. If we can
        detect that evaluated a single numerical hyperparameter, plot
        against each evaluated value.
        """
        self._plot_something_vs_params(
            axes,
            self._get_scores_vs_params(),
            score_label,
            param,
            x_log10)

    def plot_observed_vs_predicted(
            self, axes, target_name, y=None, y_pred=None, *args, **kwargs):
        raise NotImplementedError
