'''
Created on Mar 22, 2016

@author: twong
'''

import six

from abc import ABCMeta
from sklearn.base import RegressorMixin, BaseEstimator, ClassifierMixin
from sklearn.linear_model.base import LinearModel


class Regressor(six.with_metaclass(ABCMeta, LinearModel, RegressorMixin)):

    def get_name(self):
        return self.__class__.__name__


class Classifier(BaseEstimator, ClassifierMixin):

    def get_name(self):
        return self.__class__.__name__
