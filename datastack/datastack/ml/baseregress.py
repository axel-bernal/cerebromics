
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 06:49:11 2015

Copyright (C) 2015 Human Longevity Inc. Written by Peter Garst.

Provide a simple interface to baseline regression material.
"""

import datastack.ml.baselines as baselines
import datastack.cerebro.bootstrap as bootstrap
import datastack.dbs.rdb
import datastack.ml.cross_validation as cross_validation
import datastack.settings as settings
import datastack.utilities.gui as gui
import logging
import numpy as np
import pandas as pd
import os

from datastack.ml.shuffletest import shuffleRegress
import datastack.ml.prespec as prespec
from datastack.tools.vdbwrapper import MissingOptions

import re
import warnings

"""
For the age and gender baseline covariates, we might want to use the actual
value or the predicted value, or both, or neither.
If you want neither, you should probably be using run_keys instead.
These are backward-compatible constants which control this decision.
"""
BL_REPORTED = 0
BL_PREDICTED = 1
BL_BOTH = 2
BL_NEITHER = 3

_logger = logging.getLogger(__name__)

pd.options.mode.chained_assignment = None

def onAmazon ():
    """
    Return true if we are running on an Amazon instance.
    We need this because we need different urls for rosetta and vdb
    from inside and outside amazon.
    We might want to put this in a more public place.
    """
    uufile = '/sys/hypervisor/uuid'
    if not os.path.exists(uufile):
        return False
    with open(uufile) as f:
        content = f.readlines()
        if len(content) == 0:
            return False
        return 'ec2' in content[0]

def setDebugLevel(level):
    _logger.setLevel(level)

class _BaseRegress (object):
    """
    Base class for BaseRegress and BaseClassify.
    A lot of the facilities are shared between them and are located
    in this class.
    """

    # Names for important columns

    samp_key = settings.ROSETTA_INDEX_KEY
    samp_name_key = settings.ROSETTA_SUBJECT_NAME_KEY
    consented_key = settings.QUICKSILVER_KFOLDS_HOLDOUT_COLUMNS
    related_key = settings.QUICKSILVER_KFOLDS_TOGETHER_COLUMNS

    def __init__ (self, targets, covariates=None, dataframe=None, nopreproc=False, 
                  rversion=settings.ROSETTA_VERSION, namespace=settings.ROSETTA_NAMESPACE, 
                  use_predicted_baseline_covariates=BL_PREDICTED, multi = False,
                  VERSION=None,TYPE=None,COVARIATES=None,PREDICTED_COVARIATES=None,HOLDOUT=False):

        """
        targets is the required input.
        It can be one value, or a list of values, or a dictionary with
        names for the targets.
        """
        self.nopreproc = nopreproc
        self.kfolds = None
        self.allvars = None
        self.rows = None
        self.VERSION=VERSION
        self.TYPE=TYPE
        self.COVARIATES=COVARIATES
        self.PREDICTED_COVARIATES=PREDICTED_COVARIATES
        self.HOLDOUT=HOLDOUT

        self._idx_covariates = None
        self.basic = []
        self.multi = multi
        """
        We are using Christoph's ridge regression by default, which takes
        a set of columns which we do not want to penalize with the regularization
        parameter.
        Use the setter routine below to set it.
        We may want to generalize this in the future.
        """

        self.rversion = rversion
        self.namespace = namespace

        if (type(targets) is str) or (type(targets) is unicode):
            targets = {str(targets) : str(targets)}
        if type(targets) is list:
            targets = {t : t for t in targets}

        # Backward compatibility: if given a boolean value, select the
        # appropriate ordinal value.
        if use_predicted_baseline_covariates == True:
            self.blcovar = BL_PREDICTED
        elif use_predicted_baseline_covariates == False:
            self.blcovar = BL_REPORTED
        else:
            self.blcovar = use_predicted_baseline_covariates

        # This section is for explicit targets, not the baseline covariates
        if 'Age' in targets:
            targets['Age'] = 'dynamic.FACE.age.v1.value'
        if 'Gender' in targets:
            targets['Gender'] = 'dynamic.FACE.gender.v1.value'
        if 'Ethnicity' in targets:
            _logger.error('Downstream code cannot handle a list of columns as targets')
        self.targets = targets

        _logger.debug('Dictionary of targets is now %s' % (str(self.targets)))

        # Python default values are broken
        # Not only that, but 'self.covariates = covariates' passes a
        # mutable reference from the caller, which causes changes inside
        # the class to become visible outside, in a way the caller didn't
        # expect. So instead, we make a copy
        if covariates is not None:
            self.covariates = dict(covariates)
        else:
            self.covariates = {}
        """
        This is a dictionary that includes named covariate sets. It will
        include standard predefined sets for Age, Ethnicity and Gender, and
        you can modify those or add other things you want to test.
        Regular expressions are acceptable.
        """

        # A bug in python corrupts this field if we default the parameter to {}
        if self.covariates is None:
            self.covariates = {}
        # Saved for bootstrap computations
        self.covariates_user = self.covariates.keys()

        self.dataframe = dataframe
        self.evaluators = {}
        """
        After running this test, evaluators is a dictionary containing an
        EvaluationRegression object for each (covariate, target) pair.
        """

        self.cv = {}
        """
        After running this test, cv is a dictionary containing a
        CrossValidationRegression object for each (covariate, target, estimator) tuple.
        """

        self.estimatorsToRun = {}
        """
        For any covariate for which you want to specify a list of estimators
        to run, assign the list of estimators to the covariate key here.
        Each estimator is a dictionary, with entry 'est' the estimator itself,
        and an optional entry 'params' which supplies the hyperparameter search
        space for sklearn grid search. For example,
        params = {'alpha': [ 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]}
        est1 = {'est' : linear_model.Lasso(), 'params' : params, 'fit_params': fparams}
        base.estimatorsToRun['size'] = [est1]
        """

        self.extracols = {}
        """
        If you want to attach extra columns to the resulting data frame,
        besides the standard R2 and so on, add entries to this hash table.
        The key will be the name of the new column, with target added if appropriate,
        and the value is an object with a getValue function, which must take
        a cross validation object and return a number to put in the table.
        For example, in the categorical eye color predictions we use this to
        get a top two accuracy column in the resulting data frames.
        """
        self._preproc = [('.*', prespec.dropnaPreproc)]
        self._savepp = [('.*', prespec.stdPreproc)]
        self._ppenable = False
        """
        We can specify preprocessing, like standardization and outlier
        correction, for each variable.
        This is a three state system:
        1. Use standard preprocessing by default.
        2. If the user turns preprocessing "off", drop missing for backward
        compatibility.
        3. The user can set preprocessing, including None.
        """
        self._addedcol = []
        self._dropcol = []

        self.metrics_df = None
        """
        This is a data frame showing the results of the regression test.
        Use the display function to show this table with proper precision
        and other improvements over the default ipython display.
        """

        self.rdb = None
        self.rversion=rversion
        """
        This is a Rosetta database connector. It is opened and used internally
        only if there are variables which the user does not supply in a
        data frame.
        """

    def setNoPreproc (self):
        self.nopreproc = True

    def setDoPreproc (self, val):
        """
        A single switch to turn preprocessing on or "off".
        In the previous version, the preprocessing policy was just to drop
        missing values, so to allow before and after comparisons by
        turning off preprocessing we really mean just drop missing values.
        """
        if val:
            if self._ppenable:
                return
            self._preproc = self._savepp
            self._ppenable = True
        else:
            if not self._ppenable:
                return
            self._savepp = self._preproc
            self._preproc = [('.*', prespec.dropnaPreproc)]
            self._ppenable = False

    def setPreproc (self, pp, cols):
        """
        Set the preprocessing policy for one or more columns.

        pp is a PreSpec object which specifies what preprocessing to do.
        It can be None to turn off preprocessing for the columns.

        cols is a string or a list. Regular expressions are allowed. The
        regular expression '*' means that pp should be the standard processing
        for all columns.

        This turns preprocessing on also.
        This can be a little complex because of regular expressions - for example
        we can do a big regular expression and then override a subset.
        So, we keep a list, in submission order, and resolve it later.
        """
        self.setDoPreproc(True)
        self._preproc.append((cols, pp))

    def _settlePreproc (self):
        """
        We are ready to run the estimators.
        allvars holds the currently active list of columns, and we have the
        data frame.
        Figure out the preprocessing we want for each column.
        We will actually do the preprocessing later, in the cross validation process.
        No preprocessing for ordinal columns.
        """
        if self.nopreproc:
            self._ppdict = None
        else:
            self._ppdict = prespec.resolvePP(self._preproc, self.allvars)
            self._initPP()

    def _initPP (self):
        """
        We may want to do some preprocessing on the frame as a whole
        before looking at individual columns. Currently, dropping missing
        values falls into this category - if we have that specified for
        a column, then we want to drop all columns in rows where it is
        missing.
        """
        cols = list(self.testframe.columns)
        dropc = [c for c in cols if c in self._ppdict and self._ppdict[c].missing == MissingOptions.drop]
        self.testframe = self.testframe.dropna(subset=dropc)

    def getPreproc (self, cols):
        """
        Get the PreSpec objects associated with the columns.
        Return a dictionary with the object for each input column.
        This is only available after you run - is that a problem?
        """
        if self.allvars is None:
            return None

        self._settlePreproc()
        cols = self._expandCovar(cols)
        res = {k: self._ppdict[k] for k in cols}
        return res

    def dropCovars (self, run_keys, testframe):
        """
        If we are restricting covariates with run keys, drop others now.
        We also want to drop columns we don't use, so that we don't later
        drop samples for missing data in unused columns.
        """
        if run_keys is None:
            return testframe
            
        keys = self.covariates.keys()
        for k in keys:
            if not k in run_keys:
                del self.covariates[k]

        allc = [self.samp_key] + self.consented_key + self.related_key
        if self.samp_name_key in testframe.columns:
            allc += [self.samp_name_key]
        allc = allc + self.targets.values()
        for k in self.covariates.keys():
            allc = allc + self.covariates[k]
        allc = list(set(allc))
        
        testframe = testframe[allc]
        return testframe

    def setIdxCovariates (self, idx_covariates):
        """
        Set the idx covariates: the list of column names which will not be penalized
        in the regularization.
        The user specifies a list of column names.
        At a lower level these are converted into column indices whenever
        we use the default regressor in the tests. We need this structure
        because the particular columns we apply this to may vary depending on
        the covariates we use.
        """
        self._idx_covariates = list(idx_covariates)


    def _requireAge (self, df):
        """
        Drop rows if the covariates are the tail end of a larger row with AGE.
        """
        covars = list(df['Covariates'])
        keep = [(not ('AGE + ' + str(x)) in covars) for x in covars]
        df = df[keep]
        return df

    def display (self, frame=None, fmtdict=None, rows=None, dobold=True):
        """
        Use the gui tools to print with proper precision, etc.
        We get reasonable default output if we specify no parameters.

        frame is the data frame to display, default is metrics_df generated by
        the regression or classification test.

        fmtdict specifies precision for different columns.

        rows allows you to specify a few common options for which rows to
        display. Currently this is only supports:
           the default, no rows parameter, is to show all rows
           'age': show your covariate sets only in conjunction with age
           'noage': drop the AGE + rows, only show your covariates bare
        If you want to get fancier than this, you can pull out metrics_df, do
        whatever filtering you want, and pass it back into this method for
        display.
        """
        tname = " ".join(self.targets.keys())
        df = self.metrics_df
        if not frame is None:
            df = frame

        if not rows is None:
            if rows == 'age':
                df = self._requireAge(df)
            if rows == 'noage':
                df = df[df.apply(lambda x : not 'AGE +' in x['Covariates'], axis=1)]
        gui.display_metrics(tname, df, self.kfolds.df.shape[0], fmtdict=fmtdict, proctype=self._proctype, dobold=dobold)

    def addData (self, frame):
        """
        The parameter is a data frame which must include the index column.
        For any columns we do not already have, add them to our internal
        data frame.
        """
        if self.dataframe is None:
            self.dataframe = frame.copy()
        else:
            newcol = list(set(frame.columns) - set(self.dataframe.columns))
            _logger.debug('Merging %s columns into original %s columns' % ([self.samp_key] + newcol, self.dataframe.columns))
            self.dataframe = self.dataframe.merge(frame[[self.samp_key] + newcol])
            _logger.debug('Merged %s columns into dataframe; now have %d rows' % (str(newcol), self.dataframe.shape[0]))

    def dropCol (self, colname):
        """
        We may want to drop columns, like MSE.
        """
        self._dropcol.append(colname)

    def addCol (self, colname):
        """
        Currently the only valid column name is "STD(R2)", for the standard
        deviation of the R2 estimate.
        We may add other valid columns to add later.
        """
        if colname != 'STD(R2)':
            raise ValueError('Unknown added column')
        self._addedcol.append(colname)

    def _addStandardCovars (self):
        """
        We have standard covariates for age, ethnicity and gender.
        If it is an empty list, that means we want to skip it.
        Also, make sure each covariate is described by a list, not just a string.
        Also, don't regress a target on itself.

        We have a problem because the basic covariates keep changing under us.
        The regression tests use data frames to remain consistent, so we don't
        catch it there.

        Ignore run keys here and add everything; later we will drop what we
        don't want.
        """
        self.basic = ['Age', 'Gender', 'Ethnicity']
        # Drop a standard covariate if it is one of the targets
        self.basic = [b for b in self.basic if not b in self.targets]
        # Add standard covariates that are not already in the list of
        # requested covariates
        self.addthese = [b for b in self.basic if not b in self.covariates]

        # foor == reported
        # foop == predicted
        ager = 'dynamic.FACE.age.v1.value'
        agep = settings.ROSETTA_AGE
        genderr = 'dynamic.FACE.gender.v1.value'
        genderp = settings.ROSETTA_GENDER
        ethnic = ['dynamic.FACE.genome.v1.pc1', 'dynamic.FACE.genome.v1.pc2', 'dynamic.FACE.genome.v1.pc3']

        # User can drop these by setting empty list for covariates
        for b in self.basic:
            if (b in self.covariates) and (len(self.covariates[b]) == 0):
                self.covariates.pop(b)

        if 'Ethnicity' in self.addthese and self.blcovar != BL_NEITHER:
            self.covariates['Ethnicity'] = ethnic

        if self.blcovar == BL_BOTH or self.blcovar == BL_PREDICTED:
            if 'Age' in self.addthese:
                self.covariates['Age'] = [agep]
            if 'Gender' in self.addthese:
                self.covariates['Gender'] = [genderp]
        if self.blcovar == BL_BOTH or self.blcovar == BL_REPORTED:
            if 'Age' in self.addthese:
                self.covariates['Age-R'] = [ager]
            if 'Gender' in self.addthese:
                self.covariates['Gender-R'] = [genderr]

        for k in self.covariates.keys():
            val = self.covariates[k]
            if isinstance(val, str):
                self.covariates[k] = [val]

        _logger.debug('Covariate dictionary is now %s' % str(self.covariates))
        
    def _expandCovar (self, val):

        """
        We have a list of covariates, some of which may be regular expressions.
        Expand any regular expressions to get a plain text list.
        A regular expression may represent one or more columns already present
        in the data frame, or it may represent some set of Rosetta columns.
        We assume it is not both - if we find matches in the data frame we don't check
        Rosetta also.
        """
        result = []
        dcols = []
        if not self.dataframe is None:
            dcols = list(self.dataframe.columns)

        for v in val:
            pat = re.compile(v)
            fcols = [d for d in dcols if pat.search(d)]
            if len(fcols) > 0:
                result.extend(fcols)
            else:
                self._loadRdb()
                key = v
                if key.endswith('$'):
                    key = key[:-1]
                if key.startswith('^'):
                    key = key[1:]
                fcols = self.rdb.find_keys('^' + key + '$', regex=True)
                if len(fcols) > 0:
                    result.extend(fcols)
                else:
                    warnings.warn('Data not available: ' + v)

        return result
        
    # This works locally and on amazon - keep framework in place for now
    newurl = '10.6.36.96'

    def _loadRdb(self):
        if self.rdb is None:
            if onAmazon():
                self.rdb = datastack.dbs.rdb.RosettaDBMongo(host=self.newurl)
            else:
                self.rdb = datastack.dbs.rdb.RosettaDBMongo(host=self.newurl)

            self.rdb.initialize(version=self.rversion, namespace=self.namespace)

    def _mapGender2 (self):
        """
        In Rosetta encoded in strings - convert to integer.
        We want to leave nan values in place. We only have predicted gender for
        the face data currently, and we want to leave other rows in place.
        """
        glist = list(self.dataframe['dynamic.FACE_P.gender.v1.value'])
        klist = ['unicode' in str(type(g)) for g in glist]
        self.dataframe = self.dataframe[klist]
        genders = ['Male', 'Female']
        self.dataframe['dynamic.FACE_P.gender.v1.value'] = self.dataframe['dynamic.FACE_P.gender.v1.value'].apply(genders.index)

    def _mapGender (self):
        """
        In Rosetta encoded in strings - convert to integer.
        We want to leave nan values in place. We only have predicted gender for
        the face data currently, and we want to leave other rows in place.
        """
        nm = 'dynamic.FACE_P.gender.v1.value'
        self.dataframe[nm] = self.dataframe[nm].map({'Female': 1, 'Male': 0})

    def checkSize (self, df):
        """
        Because of the instability in the underlying data we sometimes use this
        to see what the hell is going on.
        """
        clist = list(df.columns)
        for c in clist:
            df0 = df[c]
            df0 = df0.dropna()
            print 'Col ', c, len(df0)
    
    def fixIntsAndStrings (self, df):
        """
        If we have sample_name in an
        existing data column, it will be an int. Or a string. Or something.
        Force it to be a string - otherwise we get crap when we read it out
        of rosetta and compare.
        """
        if 'ds.index.sample_name' in df.columns:
            df['ds.index.sample_name'] = df['ds.index.sample_name'].apply(str)
        return df

    def _findAllData (self):
        """
        Any values that we need for a target or a covariate, which are not
        already in the data frame, should be in Rosetta.
        The covariate specifications may be regular expressions, but not currently
        the target specifications.
        """
        allvars = list(self.targets.values())
        for _, value in self.covariates.iteritems():
            value = self._expandCovar(value)
            allvars.extend(value)

        # Add the consented and related subject information.
        allvars += self.consented_key + self.related_key

        allvars = list(set(allvars))

        # Keep track of the full list we need
        self.allvars = [self.samp_key] + allvars
        if (not self.dataframe is None) and (settings.ROSETTA_SUBJECT_NAME_KEY in self.dataframe.columns):
            self.allvars.append(settings.ROSETTA_SUBJECT_NAME_KEY)

        if not self.dataframe is None:
            self.dataframe = self.fixIntsAndStrings(self.dataframe)
            allvars = list(set(allvars) - set(self.dataframe.columns))

        if len(allvars) != 0:
            self._loadRdb()
    
            filters = {}
            for t in self.targets.values():
                if t in allvars:
                    filters[t] = (u'Did Not Test', '!=')
            _logger.debug('Getting additional data columns %s with filter %s' % (allvars, filters))
            added = self.rdb.query([self.samp_key] + allvars, filters=filters)
            _logger.debug('Queried Rosetta to get %d records containing %s' % (added.shape[0], added.columns))
    
            if self.dataframe is None:
                self.dataframe = added
            else:
                """
                We're losing lots of data because the database has no related and consented information
                for the samples.
                Take me now, Lord. 
                Force extra rows into added as needed.
                """
                added2 = self.forceRows(added, self.dataframe[self.samp_key])
                self.dataframe = self.dataframe.merge(added2, on=self.samp_key)
    
            # XXX This is a miserable hack: dynamic.FACE.gender uses float
            # encodings for gender, but dynamic.FACE_P.gender uses labels.
            # Bad things happen during regression if dynamic.FACE_P.gender
            # is used as a predictor or a covariate.
            if ('dynamic.FACE_P.gender.v1.value' not in self.targets.values()) and 'dynamic.FACE_P.gender.v1.value' in added.columns:
                self._mapGender()

        # This is another miserable hack: fill blanks in the consented
        # or related data
        for k in self.consented_key:
            if k in self.dataframe.columns:
                self.dataframe[k] = self.dataframe[k].fillna(False)
        for k in self.related_key:
            if k in self.dataframe.columns:
                self.dataframe[k] = self.dataframe[k].fillna('[]')

        return True
        
    def forceRows (self, frame, keys):
        """
        We were losing lots of data because the db had no related and consented
        records.
        Add empty rows as required.
        """
        need = set(keys) - set(frame[self.samp_key])
        if len(need) == 0:
            return frame
        extra = pd.DataFrame(index=need, columns=frame.columns)
        extra[self.samp_key] = extra.index
        newfr = pd.concat([frame, extra])
        return newfr

    def getAggregates (self):
        """
        If the user has aggregates turned on, want AGE (Age, Gender, Ethnicity) and
        AGE with each of the covariates.
        Do this separately for the reported and predicted covariates.
        """
        if self.blcovar == BL_BOTH or self.blcovar == BL_PREDICTED:
            self.getAggregates2('')
        if self.blcovar == BL_BOTH or self.blcovar == BL_REPORTED:
            self.getAggregates2('-R')

    def getAggregates2(self, ext):
        """
        This is either for the predicted aggregates or the reported aggregates.
        Ethnicity has only predicted.
        """
        blist = []
        for b in self.basic:
            if b + ext in self.covariates:
                blist.extend(self.covariates[b + ext])
            elif b in self.covariates:
                blist.extend(self.covariates[b])
        if len(blist) == 0:
            return

        added = [[self.aggregate_covariate_label + ext, blist]]
        for key in self.covariates:
            if not self.isBaseline(key):
                nkey = self.aggregate_covariate_label + ext + ' + ' + key
                nlist = list(self.covariates[key])
                nlist.extend(blist)
                added.append([nkey, nlist])
                if key in self.estimatorsToRun:
                    self.estimatorsToRun[nkey] = self.estimatorsToRun[key]
        for k, l in added:
            self.covariates[k] = l

        _logger.debug('Covariates are now %s' % (self.covariates.keys()))

    def isBaseline (self, key):
        """
        Return true if this is a baseline key, like Gender or AGE-R
        """
        for b in self.basic:
            if b in key:
                return True
        if self.aggregate_covariate_label in key:
            return True
        return False

    def _addCols (self, frlist, tlist, estid = None):
        """
        We have a list of data frames, with standard columns.
        We want to augment the label of the error columns, and append the
        columns into a single frame.
        Add an Id columns which allows us to uniformly access the results later.
        """
        if self.multi:
            if len(tlist) == 1:
                row = frlist[0]
                if not estid is None:
                    row['estid'] = estid
                return row

            admin = ['Model', 'Covariates','SELECT10', 'R2']
            allcol = list(frlist[0].columns)

            # This is less clean, but preserves order
            # ecols = list(set(allcol) - set(admin))
            for a in admin:
                allcol.remove(a)

            result = frlist[0][admin]

            for col in allcol:
                res=col.split(":")
                result[tlist[int(res[0])]+': '+res[1]] = frlist[0].loc[:,col]

            if not estid is None:
                result['estid'] = estid

        else:
            if len(tlist) == 1:
                row = frlist[0]
                if not estid is None:
                    row['estid'] = estid
                return row

            admin = ['Model', 'Covariates']
            allcol = list(frlist[0].columns)

            # This is less clean, but preserves order
            # ecols = list(set(allcol) - set(admin))
            for a in admin:
                allcol.remove(a)

            result = frlist[0][admin]
            for f, t in zip(frlist, tlist):
                for e in allcol:
                    result[t + ': ' + e] = f[e]

            if not estid is None:
                result['estid'] = estid

        return result



    def _runRow (self, row, targ, errfunc):
        """
        The row and target determine a specific cross validation object
        from the regression test.
        Apply the error function to the cv object to get a new error value.
        """
        cov = row['Covariates']
        eid = row['estid']
        key = (cov, targ, eid)
        if not key in self.cv.keys():
            return None
        cv = self.cv[key]
        return errfunc(cv)

    def _getStd (self, cv, targ):
        """
        This is a standard optional column in the tables:
        the standard deviation of the R2 values.
        Given the cross validation object, we need to set up and use the
        shuffle test material.
        We can generalize this if it would be useful.
        """
        est = cv.get_estimator()
        feat = list(cv.X.columns)
        res = shuffleRegress(est, feat, targ, self.dataframe)['r2']
        return res[2]

    def _runExtra (self):
        """
        If we have extra error columns to add, do it now.
        For each target, and each row, we have a new error value.
        We also handle the added column material - this is essentially an
        extra column where the function is known internally.
        """
        for targ in self.targets.keys():
            for key, errfunc in self.extracols.iteritems():
                res = self.metrics_df.apply(lambda row : self._runRow(row, targ, errfunc), 1)
                nkey = key if len(self.targets.keys()) == 1 else targ + ': ' + key
                self.metrics_df[nkey] = res
            for key in self._addedcol:
                errfunc = lambda cv : self._getStd(cv, self.targets[targ])   # Just one now.
                res = self.metrics_df.apply(lambda row : self._runRow(row, targ, errfunc), 1)
                nkey = key if len(self.targets.keys()) == 1 else targ + ': ' + key
                self.metrics_df[nkey] = res

    def _getCV (self):
        """
        Get a cross validator for every evaluator, so we can plot and so on.
        Don't we already have it?
        """
        self.cv = {}
        for key, e in self.evaluators.iteritems():
            # self.cv[key] = e.cross_validator(e.xy_generator, e.kfolds, estimator=e.estimator, nopreproc=self.nopreproc, params=e.params)
            self.cv[key] = e.cv_all

    def _dropColumns (self):
        """
        May have a list of columns to drop.
        We do simple pattern matching - if we drop MSE, for example, we drop
        all columns containing that string - that is, for all targets.
        """
        if len(self._dropcol) == 0:
            return

        cols = list(self.metrics_df.columns)
        drops = []
        for d in self._dropcol:
            drops.extend([c for c in cols if d in c])
        self.metrics_df = self.metrics_df.drop(drops, 1)

    def _unified_metrics(self, baseline):
        '''
        Tabulate the prediction scores for baseline classifiers/regressions
        and fitted classifiers/regressions.
        Based on Ted's code, this handles multiple targets.
        baseline is a dictionary on the target keys; and we have just
        calculated the evaluators for each combination of covariate and
        target keys.
        We are converting dictionaries into lists to insure proper ordering -
        is it necessary?

        If we are using a specific list of rows, some of the things here might
        not have real data.
        '''
        base = {}
        targs = self.targets.keys()
        if self.rows is None:
            if self.multi:
                base = self._addCols([baseline['multi'].metrics_df ], targs) #,['multi']
            else:
                base = self._addCols([baseline[t].metrics_df for t in targs], targs)


        # Place holder so we preserve column order.
        # Estid allows us to later find the evaluators and such attached
        # to each row.
        base['estid'] = 0

        # Get proper ordering. Do predicted and reported.
        rlist = list(self.basic)
        for c in [covariate for covariate in self.covariates_user if covariate not in rlist]:
            rlist.append(c)
        if self.aggregate_covariate_label in self.covariates:
            rlist.append(self.aggregate_covariate_label)
        for b in self.basic:
            if b+'-R' in self.covariates:
                rlist.append(b+'-R')
        for c in [covariate+'-R' for covariate in self.covariates_user if covariate+'-R' not in rlist]:
            rlist.append(c)
        if self.aggregate_covariate_label+'-R' in self.covariates:
            rlist.append(self.aggregate_covariate_label+'-R')
        rest = [r for r in self.covariates.keys() if not r in rlist]
        rest.sort()
        rlist.extend(rest)

        if not self.rows is None:
            rlist = [r for r in rlist if r in self.rows]

        crow = []
        for r in rlist:
            nlist = [key[2] for key in self.evaluators.keys() if key[0] == r]
            nlist = list(set(nlist))
            for n in nlist:
                if self.multi:
                    row = self._addCols([self.evaluators[(r, 'multi', n)].metrics_df ], targs, estid = n)
                else:
                    row = self._addCols([self.evaluators[(r, t, n)].metrics_df for t in targs], targs, estid = n)
                crow.append(row)

        blist = []
        if self.rows is None:
            blist = [base]

        self.metrics_df = pd.concat(blist + crow, ignore_index=True)
        self._runExtra()

        # Add a column to show subjects: 'Models' is required by the
        # bootstrap
        self.metrics_df['Samples'] = self.kfolds.df.shape[0]

        # May have some columns to drop
        self.metrics_df = self.metrics_df.drop('estid', 1)
        self._dropColumns()

    def _tabulate_metrics(self, baseline):
        '''
        Tabulate the prediction scores for baseline classifiers/regressions
        and fitted classifiers/regressions.
        Based on Ted's code, this handles multiple targets.
        baseline is a dictionary on the target keys; and we have just
        calculated the evaluators for each combination of covariate and
        target keys.
        We are converting dictionaries into lists to insure proper ordering -
        is it necessary?
        '''
        self._unified_metrics(baseline)

        # Drop value from mean/median
        if (self.rows is None) and (self._proctype == 'Regress'):
            self.metrics_df.ix[0, 0] = 'Mean'
            self.metrics_df.ix[1, 0] = 'Median'

    def _bootstrap(self, baseline, with_aggregate_covariates=True):
        _logger
        scorers = {}
        scorers['Error'] = lambda obs, pred : 0.0 if (obs == pred) else 1.0
        scorers['MAE'] = lambda obs, pred : abs(pred - obs)
        scorers['MSE'] = lambda obs, pred : (pred - obs) ** 2
        for t in self.targets:
            for metric in ['Error', 'MAE', 'MSE']:
                if metric not in self.metrics_df.columns:
                    _logger.debug('Metric \'%s\' not in results; skipping' % (metric))
                    continue
                _logger.debug('Running bootstrap for target \'%s\' on metric \'%s\'' % (t, metric))
                #
                # Step 1: Identify the best naive estimator for the
                # current metric.
                #
                baseline_metrics_df = baseline[t].metrics_df
                selector_naive = baseline_metrics_df.Model.str.startswith('Mean') | baseline_metrics_df.Model.str.startswith('Median') | baseline_metrics_df.Model.str.startswith('Mode')
                best_naive = baseline_metrics_df.iloc[baseline_metrics_df[selector_naive][metric].idxmin()].Model
                # Get the predictions from the best naive estimator
                if best_naive.startswith('Mean'):
                    best_naive_estimator_y_pred = baseline[t].mean_y_pred
                elif best_naive.startswith('Median'):
                    best_naive_estimator_y_pred = baseline[t].median_y_pred
                elif best_naive.startswith('Mode'):
                    best_naive_estimator_y_pred = baseline[t].mode_y_pred
                else:
                    raise Exception('Unexpected naive estimator: \'%s\'' % (best_naive))
                _logger.debug('Got the best naive baseline: %s [%s]' % (best_naive, best_naive_estimator_y_pred[0]))
                #
                # Step 2: Compare each baseline (A/G/E) and user-
                # specified covariate against the best naive estimator.
                # We may be skipping some covariates
                #
                for covariate in self.basic + self.covariates_user:
                    _logger.debug('Comparing predictions with %s covariate against the naive estimator' % (covariate))
                    key = (covariate, t, 0)
                    if key in self.evaluators:
                        baseline_evaluator = self.evaluators[(covariate, t, 0)]
                        boot = bootstrap.Bootstrap(baseline_evaluator.y, best_naive_estimator_y_pred, baseline_evaluator.y_pred)
                        boot.rescore(scorers[metric])
                        self.metrics_df.loc[self.metrics_df.Covariates == covariate, '%s-signif' % (metric)] = boot.bootstrap()[0]
                #
                # Step 3: Identify the best of the single baseline and
                # user-specified covariates.
                #
                selector_covariate = np.array([False for _ in range(0, self.metrics_df.shape[0])])
                for covariate in self.basic + self.covariates_user:
                    selector_covariate = selector_covariate | (self.metrics_df.Covariates == covariate)
                # self.metrics_df.to_csv('dump.csv', index=False)
                # Why limit the rows?
                # In some cases we are using different kinds of estimators.
                selector_covariate = selector_covariate & \
                    ((self.metrics_df.Model == 'Ridge') | (self.metrics_df.Model == 'LogisticRegression') | (self.metrics_df.Model == 'PreWrapper'))
                intval = self.metrics_df[selector_covariate]
                if len(intval) == 0:
                    continue
                best_covariate = self.metrics_df.iloc[self.metrics_df[selector_covariate][metric].idxmin()].Covariates
                best_covariate_estimator = self.evaluators[(best_covariate, t, 0)]
                _logger.debug('Got the best baseline covariate: %s' % (best_covariate))
                #
                # Step 4: Compare aggregate covariate predictions
                #
                if with_aggregate_covariates == True:
                    #
                    # Step 4a: Compare the aggregate baseline covariate
                    # against the best single covariate
                    #
                    aggregate_covariates = self.aggregate_covariate_label
                    aggregate_evaluator = self.evaluators[(aggregate_covariates, t, 0)]
                    boot = bootstrap.Bootstrap(best_covariate_estimator.y, best_covariate_estimator.y_pred, self.evaluators[(self.aggregate_covariate_label, t, 0)].y_pred, scorer=scorers[metric])
                    # This test fails on missing values, which are handled in preprocessing.
                    y0 = aggregate_evaluator.y.dropna()
                    y1 = best_covariate_estimator.y.dropna()
                    if (y0 == y1).all() != True:
                        raise Exception('Observed y values differ in the single and aggregate evaluators used for bootstrap')
                    self.metrics_df.loc[self.metrics_df.Covariates == aggregate_covariates, '%s-signif' % (metric)] = boot.bootstrap()[0]
                    #
                    # Step 4b: Compare each of the user-specified
                    # covariates combined with the aggregate baseline
                    # covariates against the aggregate baseline
                    # covariates alone
                    for covariate in self.covariates_user:
                        test_aggregate_covariates = ' + '.join([aggregate_covariates, covariate])
                        test_aggregate_evaluator = self.evaluators[(test_aggregate_covariates, t, 0)]

                        if (aggregate_evaluator.y == best_covariate_estimator.y).all() != True:
                            raise Exception('Observed y values differ in the evaluators used for bootstrap')
                        _logger.debug('Bootstrapping and comparing \'%s\' against \'%s\'' % (test_aggregate_covariates, aggregate_covariates))
                        boot = bootstrap.Bootstrap(aggregate_evaluator.y, aggregate_evaluator.y_pred, test_aggregate_evaluator.y_pred, scorer=scorers[metric])
                        self.metrics_df.loc[self.metrics_df.Covariates == test_aggregate_covariates, '%s-signif' % (metric)] = boot.bootstrap()[0]
            self.metrics_df.fillna('', inplace=True)

class BaseRegress (_BaseRegress):
    """
    Provide one line (well, maybe three) evaluation of regression problems.

    The simplest use case is: instantiate this object with one or more
    target variables; call run and then display to show a basic regression
    table.
    """

    def __init__ (self, targets, covariates=None, dataframe=None, nopreproc=False, 
                  rversion=settings.ROSETTA_VERSION, namespace=settings.ROSETTA_NAMESPACE, 
                  use_predicted_baseline_covariates=BL_PREDICTED, multi = False,
                  VERSION=None,TYPE=None,COVARIATES=None, PREDICTED_COVARIATES=None,HOLDOUT=False):
        """
        targets is the required input.
        It can be one value, or a list of values, or a dictionary with
        names for the targets.
        By default use genetically predicted age and gender among
        baseline covariates; set , use_predicted_baseline_covariates to
        False to use subject-reported age and gender.
        """
        self.multi = multi
        self._proctype = 'Regress'
        super(BaseRegress, self).__init__(targets, covariates, dataframe, nopreproc=nopreproc, rversion=rversion, namespace=namespace, use_predicted_baseline_covariates=use_predicted_baseline_covariates, multi = multi,VERSION=VERSION,TYPE=TYPE,COVARIATES=COVARIATES,PREDICTED_COVARIATES=PREDICTED_COVARIATES,HOLDOUT=HOLDOUT)


    def _runSlice (self, key, targ, fit_params=None, covariate_widget=None, kfold_widget=None, *args, **kwargs):
        """
        Run the regression tests for a particular target and covariate set.
        """

        gfit_params = fit_params
        if self.multi:
            tval = list(self.targets.values())
        else:
            tval = self.targets[targ]

        if key in self.estimatorsToRun:
            elist = self.estimatorsToRun[key]
            eind = 0
            for est in elist:
                _logger.debug('Running slice with estimator %s' % (est))
                params = None
                if 'params' in est:
                    params = est['params']
                fit_params = gfit_params
                if 'fit_params' in est:
                    fit_params = est['fit_params']
                if self.multi:
                    e = baselines._evaluate_regression(
                    tval,
                    self.kfolds,
                    covariates=self.covariates[key],
                    nopreproc=self.nopreproc,
                    estimator = est['est'],
                    params = params,
                    fit_params = fit_params,
                    idx_covariates=self._idx_covariates,
                    covariate_widget=covariate_widget,
                    kfold_widget=kfold_widget,
                    VERSION=self.VERSION,TYPE=self.TYPE,COVARIATES=self.COVARIATES,PREDICTED_COVARIATES=self.PREDICTED_COVARIATES,HOLDOUT=self.HOLDOUT,
                    *args,
                    **kwargs);
                else:
                    # XXX check here
                    e = baselines._evaluate_regression(
                    tval,
                    self.kfolds,
                    covariates=self.covariates[key],
                    nopreproc=self.nopreproc,
                    estimator = est['est'],
                    params = params,
                    fit_params = fit_params,
                    idx_covariates=self._idx_covariates,
                    pspec=self._preproc,
                    covariate_widget=covariate_widget,
                    kfold_widget=kfold_widget,
                    VERSION=self.VERSION,TYPE=self.TYPE,COVARIATES=self.COVARIATES,PREDICTED_COVARIATES=self.PREDICTED_COVARIATES,HOLDOUT=self.HOLDOUT,
                    *args,
                    **kwargs);

                # Use the name of the covariates rather than the value
                e.metrics_df.Covariates = e.metrics_df.Covariates.apply(lambda c: key)
                self.evaluators[(key, targ, eind)] = e
                eind = eind + 1
        else:
            _logger.debug('Running slice: Regressing {} on {}'.format(targ, self.covariates[key]))
            e = baselines._evaluate_regression(
                tval,
                self.kfolds,
                covariates=self.covariates[key],
                nopreproc=self.nopreproc,
                fit_params = fit_params,
                idx_covariates=self._idx_covariates,
                pspec=self._preproc,
                covariate_widget=covariate_widget,
                kfold_widget=kfold_widget,
                VERSION=self.VERSION,TYPE=self.TYPE,COVARIATES=self.COVARIATES,PREDICTED_COVARIATES=self.PREDICTED_COVARIATES,HOLDOUT=self.HOLDOUT,
                *args,
                **kwargs);

            # Use the name of the covariates rather than the value
            e.metrics_df.Covariates = e.metrics_df.Covariates.apply(lambda c: key)
            self.evaluators[(key, targ, 0)] = e

    def get_estimators (self, cov, targs, index=0, version='unknown'):
        """
        Return the estimators so we can save them and use them later.

        cov is the name of a covariate set in the regression.

        targs is the list of targets for which we want estimators. If you
        provide a single name, then this method returns a single estimator.
        If you provide a list of targets, then this method returns a
        dictionary in which the keys are the targets and the values the
        estimators.

        If you used multiple estimators for this covariate set, you can supply
        index as the index of the estimator you want. There is no way to get
        something like ridge at index 0 for one target, and lasso at index 1
        for another.

        We may want to generalize this to classification at some point.
        """
        retdict = True
        if type(targs) is str:
            targs = [targs]
            retdict = False
            
        nsamp = self.kfolds.df.shape[0]

        evlist = [self.evaluators[(cov, t, index)] for t in targs]
        cvlist = [self.cv[(cov, t, index)] for t in targs]
        elist = [cv.get_estimator(version=version) for cv in cvlist]
        for e, ev in zip(elist, evlist):
            e.hli_mae = ev.mae
            e.hli_nsamples = nsamp
            e.hli_r2 = ev.r2
            e.hli_mse = ev.mse
        if not retdict:
            return elist[0]
        ret = {t : e for t, e in zip(targs, elist)}
        return ret

    def run (
             self,
             rows=None,
             with_aggregate_covariates=True,
             with_standard_covariates=True,
             with_no_cv_object=False,
             run_full_only=False,
             run_keys=None,
             with_bootstrap=True,
             kfargs=None,
             fit_params=None,
             outlier_filter=None,
             covariate_widget=None,
             kfold_widget=None,
             *args,
             **kwargs):
        """
        Run the regression problems we have specified.

        If you specify a covariate set or list of covariate sets in the run_keys parameter,
        we will run only those rows.

        If with_aggregate_covariates is True, the default, the tools run your covariate sets
        as you specify them and also with the age, gender and ethnicity information added.

        kfargs is an optional dictionary of arguments for the KFoldPredefined method.

        fit_params is an optional dictionary of arguments for the underlying fit method.

        outlier_filter is a filter for removing outliers from the training data

        Because we may run different estimators on different covariates, you can specify
        fit_params as part of the estimatorsToRun if you wish - that will override the
        general setting.


        covariate_widget and kfold_widget are optional UI elements for showing progress.

        Variable arguments are passed down to the estimators.
        """

        """
        We might have a run key like 'AGE + size', so we want to add the 
        standard covariates here even if we have run keys.
        
        The problem we had with this earlier was related to out-of-face
        data - we were dropping samples with no standard covariates, even if
        we wanted to keep it. In this version we upgraded dropCovars so that it
        drops unneeded columns of the data frame as well as unused covariate sets.
        This prevents missing values in unused columns from eliminating samples.
        """
        if (with_standard_covariates == True):
            self._addStandardCovars()

        if not self._findAllData():
            print '*** Missing data'
            return False

        # May want to aggregate our test features with the basic features
        self.aggregate_covariate_label = ''.join([c[0] for c in self.basic])

        #print "before aggregate: {}".format(self.covariates)
        if with_aggregate_covariates:
            self.getAggregates()
            #print "after aggregate: {}".format(self.covariates)

        # Currently drop missing data - should make this more flexible.
        # But, be careful to drop missing data only for the columns we need for this
        # particular test.
        # Does this column selection make a copy even if it is all columns?
        testframe = self.dataframe[self.allvars]
   
        # If we are using run_keys to restrict rows, this is the main
        # point where we implement it.
        testframe = self.dropCovars(run_keys, testframe)

        testframe = testframe.dropna()
        if outlier_filter is not None and not self.multi:
            outlier_filter.check(testframe, testframe, self.targets.values())

        """
        To be on the safe side we will handle missing values within the folds.
        If there are missing values the default preprocessing will handle it,
        but if the user picks different preprocessing - none, or a different
        specification - the user has to get rid of missing values beforehand,
        or specify how to handle them.

        We are setting up the kfolds here, but we might wind up with slightly
        different fold sizes if we later drop some rows with missing values.
        """
        # testframe = testframe.dropna()

        # We might want to access this outside
        self.testframe = testframe

        # Figure out preprocessing for all columns
        # Currently just use old dropna for multi case
        if self.multi:
            self.testframe = self.testframe.dropna()
        else:
            self._settlePreproc()

        if kfargs is None:
            self.kfolds = cross_validation.KFoldPredefined(data=self.testframe)
        else:
            self.kfolds = cross_validation.KFoldPredefined(data=self.testframe, **kfargs)

        # If we have a rows specification, we want to do only those.
        # Assume baseline is not there.
        # No regular expressions.
        rows = run_keys
        if (not rows is None) and (not type(rows) is list):
            rows = [rows]
        self.rows = rows

        baseline = {}
        if self.multi:
            _logger.info('Evaluating baseline models for target %s...' % ('multi'))
            baseline['multi'] = cross_validation.EvaluationRegressionBaseline(list(self.targets.values()), self.kfolds)
            #idx=0
            #last = len(self.covariates)
            _logger.info('Evaluating regression models for target %s...' % ('multi'))

            if  not run_full_only:
                for key in self.covariates:
                    self._runSlice(key, 'multi', fit_params, covariate_widget, kfold_widget, *args, **kwargs)
            else:
                #print self.covariates
                for key in self.covariates:
                    if (key in run_keys):
                        #print "key={}".format(key)
                        self._runSlice(key, 'multi', fit_params, covariate_widget, kfold_widget, *args, **kwargs)
                #idx+=1
        else:
            for t in self.targets:
                # In the case where we are using rows to specify, we don't want the
                # baseline data, but it is easier to fill out the whole frame if we
                # put it in and later take it out.
                _logger.info('Evaluating baseline models for target %s...' % (t))
                baseline[t] = cross_validation.EvaluationRegressionBaseline(self.targets[t], self.kfolds, ppdict=self._ppdict)
                _logger.info('Evaluating regression models for target %s...' % (t))
                for key in self.covariates:
                    if (run_keys is None) or (key in run_keys):
                        self._runSlice(key, t, fit_params, covariate_widget, kfold_widget, *args, **kwargs)
        
        # get CV objects also
        self._getCV()
        self._tabulate_metrics(baseline)

        # If bootstrap requested, run our '**' estimator to identify
        # significant covariates
        if with_bootstrap == True:
            _logger.info('Bootstrap aggregating...')
            self._bootstrap(baseline, with_aggregate_covariates=with_aggregate_covariates)

        return True

def testev (cv):
    """
    """
    return len(cv.kfolds)

# For debugging
if __name__ == "__main__":
    #import face.eyecolor.eyedata as eyedata
    #data = eyedata.EyeData()
    baset3 = BaseRegress('facepheno.hand.strength.right.m1')
    #baset3.covariates['split'] = ['rs12913832A', 'rs12913832B']
    #baset3.covariates['nosplit'] = ['rs12913832']
    #baset3.addData(data.ndata)
    #baset3.addCol('STD(R2)')
    baset3.run(with_aggregate_covariates=False)
    print baset3.metrics_df
