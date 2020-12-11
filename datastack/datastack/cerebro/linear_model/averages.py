'''
Created on Mar 22, 2016

@author: twong
'''

import numpy as np

from datastack.cerebro.linear_model.abstract import Classifier, Regressor
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import check_is_fitted


class Mean(Regressor):

    def fit(self, X, y):
        """Fit model.
        """
        self._mean = y.mean()

    def predict(self, X):
        check_is_fitted(self, '_mean')
        return np.array([self._mean] * X.shape[0])


class Median(Regressor):

    def fit(self, X, y):
        """Fit model.
        """
        self._median = y.median()

    def predict(self, X):
        check_is_fitted(self, '_median')
        return np.array([self._median] * X.shape[0])


class Mode(Classifier):

    def fit(self, X, y):
        """Fit model.
        """
        self._mode = y.mode()
        # Fake a prediction probability vector equivalent to the support
        # fractions for each class. We MUST make sure that the entries
        # correspond to the class order that will be used by the log_loss
        # method (i.e., the order created by `LabelBinarizer`)
        lb = LabelBinarizer()
        lb.fit_transform(y)
        # Support will contain a list of (label, count) tuples, with the most
        # common label first.
        support = sorted([(label, y.values.tolist().count(label)) for label in set(y.values.tolist())], key=(lambda v: v[1]), reverse=True)
        self._mode_prob = [float(dict(support)[l]) / len(y) for l in lb.classes_]

    def predict(self, X):
        check_is_fitted(self, ['_mode', '_mode_prob'])
        return np.array([self._mode] * X.shape[0])

    def predict_proba(self, X):
        check_is_fitted(self, ['_mode', '_mode_prob'])
        return np.array([self._mode_prob] * X.shape[0])