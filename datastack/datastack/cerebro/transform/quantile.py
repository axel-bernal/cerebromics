'''
Created on Nov 2, 2016

@author: twong
'''

import numpy as np
import pandas as pd
import scipy.stats

from sklearn.base import TransformerMixin, RegressorMixin, BaseEstimator, ClassifierMixin


class Quantile(TransformerMixin, RegressorMixin, BaseEstimator, ClassifierMixin):

    def __init__(self, bins=None, eps=1e-6, distribution='normal'):
        self.bins = bins
        self.distribution = distribution
        self.X = None
        self.ranks = None
        self.quant = None
        self.target = None
        self.eps = eps

    def fit_dim(self, x):
        assert isinstance(x, np.ndarray)

        observed = (x == x)
        x = x[observed]
        if self.eps > 0:
            x = x + (np.random.uniform(size=x.shape) - 0.5) * self.eps
        x.sort(0)
        if self.bins is not None:
            steps = max(1, x.shape[0] / self.bins)
            x = x[::steps]

        ranks = scipy.stats.mstats.rankdata(x, axis=0)
        quant = (ranks - 0.5) / x.shape[0]
        if self.distribution == "normal":
            target = -scipy.stats.norm.isf(quant)
        elif self.distribution == "uniform":
            target = -scipy.stats.uniform.isf(quant)
        else:
            raise NotImplementedError(self.distribution)
        return x, ranks, quant, target

    def fit(self, X, y=None):

        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.values

        X = X.copy()
        if len(X.shape) == 2:
            self.X = []
            self.ranks = []
            self.quant = []
            self.target = []
            for i in xrange(X.shape[1]):
                x, ranks, quant, target = self.fit_dim(X[:, i])
                self.X.append(x)
                self.ranks.append(ranks)
                self.quant.append(quant)
                self.target.append(target)
        elif len(X.shape) == 1:
            self.X, self.ranks, self.quant, self.target = self.fit_dim(X)
        else:
            raise NotImplementedError(str(X.shape))

    def fit_transform(self, X, y=None, impute_na='zero'):
        self.fit(X, y=y)
        return self.transform(X=X, y=y, impute_na=impute_na)

    def transform(self, X, y=None, impute_na='zero'):
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.values

        i_obs = X == X

        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.values
        if len(X.shape) == 2:
            target = np.empty(X.shape)
            target[:] = np.nan
            for i in xrange(X.shape[1]):
                ranks = np.searchsorted(self.X[i], X[i_obs[:, i], i], side='left', sorter=None)
                ranks[ranks >= self.X[i].shape[0]] = self.X[i].shape[0] - 1
                target[i_obs[:, i], i] = self.target[i][ranks]
        elif len(X.shape) == 1:
            target = np.empty(X.shape)
            target[:] = np.nan
            ranks = np.searchsorted(self.X, X[i_obs], side='left', sorter=None)
            ranks[ranks >= self.X.shape[0]] = self.X.shape[0] - 1
            target[i_obs] = self.target[ranks]
        else:
            raise NotImplementedError(str(X.shape))
        if impute_na == 'zero':
            target[~i_obs] = 0.0
        return target
