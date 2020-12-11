""" Weighted Principal Component Analysis
"""

# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Mathieu Blondel <mathieu@mblondel.org>
#         Denis A. Engemann <d.engemann@fz-juelich.de>
#         Michael Eickenberg <michael.eickenberg@inria.fr>
# 
#         
# License: BSD 3 clause

#modified by Christoph Lippert

from math import log, sqrt

import numpy as np
from scipy import linalg
from scipy.special import gammaln

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state, as_float_array
from sklearn.utils import check_array
from sklearn.utils.extmath import fast_dot, fast_logdet, randomized_svd
from sklearn.utils.validation import check_is_fitted


class WeightedPCA(BaseEstimator, TransformerMixin):
    """Principal component analysis (PCA)

    Linear dimensionality reduction using Singular Value Decomposition of the
    data and keeping only the most significant singular vectors to project the
    data to a lower dimensional space.

    This implementation uses the scipy.linalg implementation of the singular
    value decomposition. It only works for dense arrays and is not scalable to
    large dimensional data.

    The time complexity of this implementation is ``O(n ** 3)`` assuming
    n ~ n_samples ~ n_features.

    Read more in the :ref:`User Guide <PCA>`.

    Parameters
    ----------
    n_components : int, None or string
        Number of components to keep.
        if n_components is not set all components are kept::

            n_components == min(n_samples, n_features)

        if n_components == 'mle', Minka\'s MLE is used to guess the dimension
        if ``0 < n_components < 1``, select the number of components such that
        the amount of variance that needs to be explained is greater than the
        percentage specified by n_components

    copy : bool
        If False, data passed to fit are overwritten and running
        fit(X).transform(X) will not yield the expected results,
        use fit_transform(X) instead.

    whiten : bool, optional
        When True (False by default) the `components_` vectors are divided
        by n_samples times singular values to ensure uncorrelated outputs
        with unit component-wise variances.

        Whitening will remove some information from the transformed signal
        (the relative variance scales of the components) but can sometime
        improve the predictive accuracy of the downstream estimators by
        making there data respect some hard-wired assumptions.

    Attributes
    ----------
    components_ : array, [n_components, n_features]
        Principal axes in feature space, representing the directions of
        maximum variance in the data. The components are sorted by
        explained_variance_.

    explained_variance_ : array, [n_components]
        The amount of variance explained by each of the selected components.

    explained_variance_ratio_ : array, [n_components]
        Percentage of variance explained by each of the selected components.

        If ``n_components`` is not set then all components are stored and the
        sum of explained variances is equal to 1.0.

    mean_ : array, [n_features]
        Per-feature empirical mean, estimated from the training set.

        Equal to `X.mean(axis=1)`.

    n_components_ : int
        The estimated number of components. Relevant when `n_components` is set
        to 'mle' or a number between 0 and 1 to select using explained
        variance.

    noise_variance_ : float
        The estimated noise covariance following the Probabilistic PCA model
        from Tipping and Bishop 1999. See "Pattern Recognition and
        Machine Learning" by C. Bishop, 12.2.1 p. 574 or
        http://www.miketipping.com/papers/met-mppca.pdf. It is required to
        computed the estimated data covariance and score samples.

    Notes
    -----
    For n_components='mle', this class uses the method of `Thomas P. Minka:
    Automatic Choice of Dimensionality for PCA. NIPS 2000: 598-604`

    Implements the probabilistic PCA model from:
    M. Tipping and C. Bishop, Probabilistic Principal Component Analysis,
    Journal of the Royal Statistical Society, Series B, 61, Part 3, pp. 611-622
    via the score and score_samples methods.
    See http://www.miketipping.com/papers/met-mppca.pdf

    Due to implementation subtleties of the Singular Value Decomposition (SVD),
    which is used in this implementation, running fit twice on the same matrix
    can lead to principal components with signs flipped (change in direction).
    For this reason, it is important to always use the same estimator object to
    transform data in a consistent fashion.

    Examples
    --------

    >>> import numpy as np
    >>> from sklearn.decomposition import PCA
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> pca = PCA(n_components=2)
    >>> pca.fit(X)
    PCA(copy=True, n_components=2, whiten=False)
    >>> print(pca.explained_variance_) # doctest: +ELLIPSIS
    [ 6.6162...  0.05038...]
    >>> print(pca.explained_variance_ratio_) # doctest: +ELLIPSIS
    [ 0.99244...  0.00755...]

    See also
    --------
    RandomizedPCA
    KernelPCA
    SparsePCA
    TruncatedSVD
    """
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.copy = True

    def fit(self, X, y=None, weights=None):
        """Fit the model with X.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._fit(X, weights)
        return self

    def fit_transform(self, X, y=None, weights=None):
        """Fit the model with X and apply the dimensionality reduction on X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)

        """
        if weights is None:
            weights = np.ones((X.shape[0]))

        U, S, V = self._fit(X, weights)
        return self.transform(X)
        # U = U[:,:self.n_components_]
        # # X_new = X * V = U * S * V^T * V = U * S
        # U *= S[np.newaxis,:self.n_components_]
        #
        # return U

    def _fit(self, X, weights=None):
        """Fit the model on X

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

        Returns
        -------
        U, s, V : ndarrays
            The SVD of the input data, copied and centered when
            requested.
        """
        X = check_array(X)
        n_samples, n_features = X.shape
        X = as_float_array(X, copy=self.copy)

        if weights is None:
            weights = np.ones((X.shape[0]))

        # Center data
        self.mean_ = np.sum(X * weights[:,np.newaxis], axis=0) / weights.sum()

        rank = min(X.shape)
        X = X - self.mean_[np.newaxis,:]
        #K_i = X_i.T.dot( X_i * responsibilities[i,:,np.newaxis])
        #s, v = linalg.eigh(K_i)

        U, S, components_ = linalg.svd(X * np.sqrt(weights[:,np.newaxis]) , full_matrices=False)
        explained_variance_ = (S * S)
        explained_variance_ratio_ = (explained_variance_ /
                                     explained_variance_.sum())
          
        n_components = self.n_components
        if n_components is None:
            n_components = n_features
        elif n_components == 'mle':
            if n_samples < n_features:
                raise ValueError("n_components='mle' is only supported "
                                 "if n_samples >= n_features")

            n_components = _infer_dimension_(explained_variance_,
                                             n_samples, n_features)
        elif not 0 <= n_components <= n_features:
            raise ValueError("n_components=%r invalid for n_features=%d"
                             % (n_components, n_features))

        if 0 < n_components < 1.0:
            # number of components for which the cumulated explained variance
            # percentage is superior to the desired threshold
            ratio_cumsum = explained_variance_ratio_.cumsum()
            n_components = (ratio_cumsum < n_components).sum() + 1

        # Compute noise covariance using Probabilistic PCA model
        # The sigma2 maximum likelihood (cf. eq. 12.46)
        if n_components < rank:
            self.noise_variance_ = explained_variance_[n_components:].mean(1)
        else:
            self.noise_variance_ = 0.

        # store n_samples to revert whitening when getting covariance
        self.n_samples_ = n_samples

        self.components_ = components_[:,:n_components]
        self.explained_variance_ = explained_variance_[:n_components]
        explained_variance_ratio_ = explained_variance_ratio_[:n_components]
        self.explained_variance_ratio_ = explained_variance_ratio_
        self.n_components_ = n_components

        return (U, S, components_)

    def transform(self, X):
        """Apply the dimensionality reduction on X.

        X is projected on the first principal components previous extracted
        from a training set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)

        """
        check_is_fitted(self, 'mean_')

        X = check_array(X)
        if self.mean_ is not None:
            X = X - self.mean_
        X_transformed = fast_dot(X, self.components_.T)
        return X_transformed

class EthnicityPCA(BaseEstimator, TransformerMixin):
    """Principal component analysis (PCA)

    Linear dimensionality reduction using Singular Value Decomposition of the
    data and keeping only the most significant singular vectors to project the
    data to a lower dimensional space.

    This implementation uses the scipy.linalg implementation of the singular
    value decomposition. It only works for dense arrays and is not scalable to
    large dimensional data.

    The time complexity of this implementation is ``O(n ** 3)`` assuming
    n ~ n_samples ~ n_features.

    Read more in the :ref:`User Guide <PCA>`.

    Parameters
    ----------
    n_components : int, None or string
        Number of components to keep.
        if n_components is not set all components are kept::

            n_components == min(n_samples, n_features)

        if n_components == 'mle', Minka\'s MLE is used to guess the dimension
        if ``0 < n_components < 1``, select the number of components such that
        the amount of variance that needs to be explained is greater than the
        percentage specified by n_components

    copy : bool
        If False, data passed to fit are overwritten and running
        fit(X).transform(X) will not yield the expected results,
        use fit_transform(X) instead.

    whiten : bool, optional
        When True (False by default) the `components_` vectors are divided
        by n_samples times singular values to ensure uncorrelated outputs
        with unit component-wise variances.

        Whitening will remove some information from the transformed signal
        (the relative variance scales of the components) but can sometime
        improve the predictive accuracy of the downstream estimators by
        making there data respect some hard-wired assumptions.

    Attributes
    ----------
    components_ : array, [n_components, n_features]
        Principal axes in feature space, representing the directions of
        maximum variance in the data. The components are sorted by
        explained_variance_.

    explained_variance_ : array, [n_components]
        The amount of variance explained by each of the selected components.

    explained_variance_ratio_ : array, [n_components]
        Percentage of variance explained by each of the selected components.

        If ``n_components`` is not set then all components are stored and the
        sum of explained variances is equal to 1.0.

    mean_ : array, [n_features]
        Per-feature empirical mean, estimated from the training set.

        Equal to `X.mean(axis=1)`.

    n_components_ : int
        The estimated number of components. Relevant when `n_components` is set
        to 'mle' or a number between 0 and 1 to select using explained
        variance.

    noise_variance_ : float
        The estimated noise covariance following the Probabilistic PCA model
        from Tipping and Bishop 1999. See "Pattern Recognition and
        Machine Learning" by C. Bishop, 12.2.1 p. 574 or
        http://www.miketipping.com/papers/met-mppca.pdf. It is required to
        computed the estimated data covariance and score samples.

    Notes
    -----
    For n_components='mle', this class uses the method of `Thomas P. Minka:
    Automatic Choice of Dimensionality for PCA. NIPS 2000: 598-604`

    Implements the probabilistic PCA model from:
    M. Tipping and C. Bishop, Probabilistic Principal Component Analysis,
    Journal of the Royal Statistical Society, Series B, 61, Part 3, pp. 611-622
    via the score and score_samples methods.
    See http://www.miketipping.com/papers/met-mppca.pdf

    Due to implementation subtleties of the Singular Value Decomposition (SVD),
    which is used in this implementation, running fit twice on the same matrix
    can lead to principal components with signs flipped (change in direction).
    For this reason, it is important to always use the same estimator object to
    transform data in a consistent fashion.

    Examples
    --------

    >>> import numpy as np
    >>> from sklearn.decomposition import PCA
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> pca = PCA(n_components=2)
    >>> pca.fit(X)
    PCA(copy=True, n_components=2, whiten=False)
    >>> print(pca.explained_variance_) # doctest: +ELLIPSIS
    [ 6.6162...  0.05038...]
    >>> print(pca.explained_variance_ratio_) # doctest: +ELLIPSIS
    [ 0.99244...  0.00755...]

    See also
    --------
    RandomizedPCA
    KernelPCA
    SparsePCA
    TruncatedSVD
    """
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.copy = True

    def fit(self, X, y=None, ethnicity=None):
        """Fit the model with X.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._fit(X, ethnicity)
        return self

    def fit_transform(self, X, y=None, ethnicity=None):
        """Fit the model with X and apply the dimensionality reduction on X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)

        """
        if ethnicity is None:
            ethnicity = np.ones((X.shape[0],1))

        U, S, V = self._fit(X, ethnicity)
        U = U[:,:,:self.n_components_]

        # X_new = X * V = U * S * V^T * V = U * S
        U *= S[:,np.newaxis,:self.n_components_]

        return U

    def _fit(self, X, ethnicity=None):
        """Fit the model on X

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

        Returns
        -------
        U, s, V : ndarrays
            The SVD of the input data, copied and centered when
            requested.
        """
        X = check_array(X)
        n_samples, n_features = X.shape
        X = as_float_array(X, copy=self.copy)

        if ethnicity is None:
            ethnicity = np.ones((1,X.shape[0]))

        ethnicity_denom = ethnicity.sum(0)
        responsibilities = ethnicity / ethnicity_denom[np.newaxis,:]
        pi = responsibilities.mean(1)

        # Center data
        self.mean_ = np.sum(X[np.newaxis,:,:] * responsibilities[:,:,np.newaxis], axis=1) / responsibilities.sum(1)[:,np.newaxis]

        rank = min(X.shape)
        components_ = np.empty((ethnicity.shape[0],rank,X.shape[1]))
        U = np.empty((ethnicity.shape[0],X.shape[0],rank))
        S = np.empty((ethnicity.shape[0],rank))
        explained_variance_ = np.empty((ethnicity.shape[0],rank))
        explained_variance_ratio_ = np.empty((ethnicity.shape[0],rank))
        for i in xrange(ethnicity.shape[0]):
            X_i = X - self.mean_[i,np.newaxis,:]
            #K_i = X_i.T.dot( X_i * responsibilities[i,:,np.newaxis])
            #s, v = linalg.eigh(K_i)

            U[i], S[i], components_[i] = linalg.svd(X_i * np.sqrt(responsibilities[i,:,np.newaxis]) , full_matrices=False)
        explained_variance_ = (S * S)
        explained_variance_ratio_ = (explained_variance_ /
                                     explained_variance_.sum(1)[:,np.newaxis])
          
        n_components = self.n_components
        if n_components is None:
            n_components = n_features
        elif n_components == 'mle':
            if n_samples < n_features:
                raise ValueError("n_components='mle' is only supported "
                                 "if n_samples >= n_features")

            n_components = _infer_dimension_(explained_variance_,
                                             n_samples, n_features)
        elif not 0 <= n_components <= n_features:
            raise ValueError("n_components=%r invalid for n_features=%d"
                             % (n_components, n_features))

        if 0 < n_components < 1.0:
            # number of components for which the cumulated explained variance
            # percentage is superior to the desired threshold
            ratio_cumsum = explained_variance_ratio_.cumsum(1)
            n_components = (ratio_cumsum < n_components).sum(1) + 1

        # Compute noise covariance using Probabilistic PCA model
        # The sigma2 maximum likelihood (cf. eq. 12.46)
        if n_components < rank:
            self.noise_variance_ = explained_variance_[:,n_components:].mean(1)
        else:
            self.noise_variance_ = 0.

        # store n_samples to revert whitening when getting covariance
        self.n_samples_ = n_samples

        self.components_ = components_[:,:,:n_components]
        self.explained_variance_ = explained_variance_[:,:n_components]
        explained_variance_ratio_ = explained_variance_ratio_[:,:n_components]
        self.explained_variance_ratio_ = explained_variance_ratio_
        self.n_components_ = n_components

        return (U, S, components_)
