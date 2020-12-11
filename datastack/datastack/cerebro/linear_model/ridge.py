'''HLI internal ridge regression linear model.

@author: clippert
'''
import logging
import numpy as np
import numpy.linalg
import scipy.linalg
import sklearn.metrics

from datastack.cerebro.linear_model.abstract import Regressor
from datastack.serializer import Serializable
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

_logger = logging.getLogger(__name__)

# twong: I copied some of the infrastructure from sklearn cargo-cult-style
# in order to ensure somewhat uniform API behavior between our linear
# models and those provided by sklearn.


def _get_alpha(alpha, idx_covariates, X):
    '''Transform a regularizer to account for unpenalized covariate
    coefficients.

    Args:
        alpha: The regularizer.
        idx_covariates: A optional array-like of n_features booleans
            indicating which coefficient values should not be
            penalized. If not set, we fit using the array-like set in
            the constructor (which may be `None`) or by a previous
            call to `fit`. If set to None, then alpha remains unchanged.
        X: An array-like of training data feature values.

    Returns:
        Scalar alpha if idx_covariates is `None`, or an array-like
        of alpha values with features specified by idx_covariates left
        unpenalized.
    '''
    _alpha = alpha
    if idx_covariates is not None:
        if issubclass(type(_alpha), np.ndarray) and len(_alpha.shape) == 2:
            _alpha = _alpha.copy()
            _alpha[idx_covariates, :] = 0.0
            _alpha[:, idx_covariates] = 0.0
            _alpha[idx_covariates, idx_covariates] = 1e-10
        else:
            _alpha = np.ones(X.shape[1]) * _alpha
            _alpha[idx_covariates] = 1e-10
    return _alpha


def _beta_large_p(alpha, idx_covariates, X, y):
    _alpha = _get_alpha(alpha, idx_covariates, X)
    if issubclass(type(_alpha), np.ndarray):
        X_rot = X / _alpha[np.newaxis, :]
    elif _alpha > 0:  # scalar
        X_rot = X / _alpha
    else:
        raise ValueError(
            "Expected _alpha > 0.0 for N < D case, got %f" % _alpha)

    XX = X.dot(X_rot.T)
    XX.flat[::XX.shape[0] + 1] += 1.0

    try:
        XXiY = scipy.linalg.solve(XX, y, sym_pos=True)
    except:
        _logger.warning(
            'XX is low rank; reverting to least squares instead')
        XXiY = numpy.linalg.lstsq(XX, y)[0]

    beta = X_rot.T.dot(XXiY)
    return beta


def _beta_small_p(alpha, idx_covariates, X, y):
    _alpha = _get_alpha(alpha, idx_covariates, X)
    XX = X.T.dot(X).astype('float64')
    if issubclass(type(_alpha), np.ndarray) and len(_alpha.shape) == 2:
        XX += _alpha
    else:
        XX.flat[::XX.shape[0] + 1] += _alpha
    XY = X.T.dot(y)
    try:
        beta = scipy.linalg.solve(XX, XY, sym_pos=True)
    except:
        _logger.warning(
            'XX is low rank; reverting to least squares instead')
        beta = numpy.linalg.lstsq(XX, XY)[0]
    return beta


def _beta(alpha, idx_covariates, X, y):
    if X.shape[0] >= X.shape[1]:
        return _beta_small_p(alpha, idx_covariates, X, y)
    else:
        return _beta_large_p(alpha, idx_covariates, X, y)


class Ridge(Regressor, Serializable):

    def __init__(self, alpha=1.0, fit_intercept=True, idx_covariates=None):
        '''Construct a new ridge regression linear model. The constructor
        accepts and stores all of the parameters that can be searched over
        with an sklearn grid search.

        Args:
            alpha: The regularizer (also referenced as the lambda tuning
                parameter) used to penalize linear model coefficient
                values that tend away from zero. Can be a scalar, an
                vector-like of length n_features, or an array-like of
                n_features by n_features
            fit_intercept: A flag to specify whether or not to fit an
                intercept for the linear model.
            idx_covariates: A optional array-like of n_features booleans
                indicating which coefficient values should not be
                penalized. If not set, we fit using the array-like set in
                the constructor (which may be `None`) or by a previous
                call to `fit`.
        '''
        super(Ridge, self).__init__()
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.idx_covariates = idx_covariates

    def _compatible(self, vec, mat):
        return vec.shape[0] == mat.shape[1]

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, v):
        '''Ensure that alpha is a scalar, a vector, or a square matrix.
        '''
        if isinstance(v, list):
            v = np.array(v)
        if issubclass(type(v), np.ndarray):
            if len(v.shape) > 2 or (
                    len(v.shape) == 2 and v.shape[0] != v.shape[1]):
                raise ValueError(
                    'Invalid alpha: Expected scalar, vector, or square matrix')
        self._alpha = v

    def fit(self, X, y, **kwargs):
        '''Fit a ridge regression to a training data set of covariate
        feature values and observed target values.

        Implements `fit` from `sklearn.linear_model.LinearModel`.

        Args:
            X: An array-like data structure of feature values with shape
                n_samples by n_features.
            y: A data structure of observed target values; either a
                vector-like with n_samples or an array-like of n_samples
                by n_targets.
            idx_covariates: A optional array-like of n_features booleans
                indicating which coefficient values should not be
                penalized. If not set, we fit using the array-like set in
                the constructor (which may be `None`) or by a previous
                call to `fit`.
        '''
        if X is None or y is None:
            raise ValueError(
                'Invalid training data: Neither X nor y may be None')
        X, y = check_X_y(
            X, y, multi_output=True, y_numeric=True, estimator=self)
        if 'idx_covariates' in kwargs:
            self.idx_covariates = kwargs['idx_covariates']

        # Check idx_covariates, X, and alpha to make sure all of the dimensions
        # are compatible - some mismatches only seem to result in warnings which
        # produces weird results.
        # Apparently idx_covariates can be an array of indices, or an array of
        # booleans - skip the dimension check, since these will be different.
        if self.idx_covariates is not None:
            """
            if not self._compatible(self.idx_covariates, X):
                raise ValueError(
                    'idx_covariates and X have mismatched n_features: '
                    'Expected %d, got %d'
                    % (self.idx_covariates.shape[0], X.shape[1]))
            """
            if issubclass(type(self.alpha), np.ndarray):
                # Alpha is guaranteed to be square so we only need to check one
                # dimension
                if not self._compatible(self.alpha, X):
                    raise ValueError(
                        'alpha and X have mismatched n_features: '
                        'Expected %d, got %d' % (self.alpha.shape[0], X.shape[1]))

        if y.ndim == 1:
            y = y[:, np.newaxis]

        if self.fit_intercept:
            self._intercept_X = X.mean(0)
            self._intercept_y = y.mean(0)[np.newaxis, :]
            X = X - self._intercept_X
            y = y - self._intercept_y
        else:
            self._intercept_X = None
            self._intercept_y = None

        # The original implementation recomputed beta on every call to
        # predict, which seems wasteful...
        self._beta = _beta(self.alpha, self.idx_covariates, X, y)
        # Return ourselves, just like sklearn.
        return self

    def predict(self, X):
        '''Predict target values given a set of covariate values. Checks
        first that we have trained the model.

        Overrides `predict` from `sklearn.linear_model.LinearModel`.

        Args:
            X: An array-like data structure of feature values with
                shape n_samples by n_features.

        Returns:
            y: A data structure of predicted target values; either a
                vector-like with n_samples or an array-like of n_samples
                by n_targets.
        '''
        check_is_fitted(self, "_beta")
        X = check_array(X)
        if self.fit_intercept:
            X = X - self._intercept_X
            y = safe_sparse_dot(X, self._beta) + self._intercept_y
        else:
            y = safe_sparse_dot(X, self._beta)
        # Optimize the shape of the array: if each row in the array only has
        # one column, flatten the array.
        if len(y.shape) == 2 and y.shape[1] == 1:
            return y.flatten()
        else:
            return y

    def score(self, X, y, sample_weight=None, multioutput='variance_weighted'):
        '''Returns the coefficient of determination R^2 of the
        prediction.

        Implements `score` from `sklearn.base.RegressorMixin`.

        Args:
            _X: An array-like data structure of feature values with
                shape n_samples by n_features.
            y: An data structure of observed target values; either a
                vector-like with n_samples or an array-like of n_samples
                by n_targets.
            sample_weight: A optional array-like of n_samples weights.
            multioutput: An optional string specifying how to aggregate
                multiple output scores or an optional array-like of
                n_targets weights used to average scores.
        '''
        if X is None or y is None:
            raise ValueError(
                'Invalid scoring data: Neither X nor y may be None')
        X, y = check_X_y(
            X, y, multi_output=True, y_numeric=True, estimator=self)
        y_pred = self.predict(X=X)
        score = sklearn.metrics.r2_score(
            y,
            y_pred,
            sample_weight=sample_weight,
            multioutput=multioutput)
        return score

    @property
    def serialized_version(self):
        return 1

    def serialize(self, data):
        data['alpha'] = self._alpha
        data['beta'] = self._beta
        data['fit_intercept'] = self.fit_intercept
        data['intercept_X'] = self._intercept_X
        data['intercept_y'] = self._intercept_y
        return data

    def deserialize(self, data):
        self._alpha = data['alpha']
        self._beta = data['beta']
        self.fit_intercept = data['fit_intercept']
        self._intercept_X = data['intercept_X']
        self._intercept_y = data['intercept_y']
