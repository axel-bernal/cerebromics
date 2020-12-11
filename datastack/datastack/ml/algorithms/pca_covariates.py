""" Principal Component Analysis
"""

# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Mathieu Blondel <mathieu@mblondel.org>
#         Denis A. Engemann <d.engemann@fz-juelich.de>
#         Michael Eickenberg <michael.eickenberg@inria.fr>
#
# License: BSD 3 clause

# modified by Christoph Lippert

from math import log, sqrt

import numpy as np
from scipy import linalg
from scipy.special import gammaln

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import as_float_array
from sklearn.utils import check_array
from sklearn.utils.extmath import fast_dot, fast_logdet, randomized_svd
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import KernelCenterer
from preprocess import KernelCentererCovariates
from linreg import Ridge, OLS

import cPickle



class CovariatesPCA(BaseEstimator, TransformerMixin):
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
    def __init__(self, n_components=None, copy=True, whiten=False, normalize=False, alpha=0.0):
        self.n_components = n_components
        self.copy = copy
        self.whiten = whiten
        self.normalize = normalize
        self.beta_covariates = None
        self.alpha = alpha
        self.mean_ = None
        self.mean_covariates_ = None
        self.noise_variance_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.n_components_ = None
        self.n_samples_ = None

    def fit(self, X, y=None, covariates=None):
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
        self._fit(X, covariates=covariates)
        return self

    def fit_transform(self, X, y=None, covariates=None):
        """Fit the model with X and apply the dimensionality reduction on X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        covariates: array-like, shape (n_samples, n_covariates)
            Training vector, where n_samples in the number of samples and
            n_covariates is the number of covariates to be taken out form the PCA.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)

        """
        U, S, V = self._fit(X, covariates=covariates)
        U = U[:, :self.n_components_]

        if self.whiten:
            # X_new = X * V / S * sqrt(n_samples) = U * sqrt(n_samples)
            U *= sqrt(X.shape[0])
        else:
            # X_new = X * V = U * S * V^T * V = U * S
            U *= S[:self.n_components_]

        if self.normalize:
            U /= np.sqrt(self.explained_variance_.sum())
        return U

    def _fit(self, X, covariates=None):
        """Fit the model on X

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.
        covariates: array-like, shape (n_samples, n_covariates)
            Training vector, where n_samples in the number of samples and
            n_covariates is the number of covariates to be taken out form the PCA.

        Returns
        -------
        U, s, V : ndarrays
            The SVD of the input data, copied and centered when
            requested.
        """
        X = check_array(X, force_all_finite=False)
        n_samples, n_features = X.shape
        X = as_float_array(X, copy=self.copy, force_all_finite=False)
        # Center data
        self.mean_ = np.nanmean(X, axis=0)
        X -= self.mean_
        X[X != X] = 0.0

        if covariates is not None:
            covariates = as_float_array(covariates, copy=self.copy, force_all_finite=False)
            self.mean_covariates_ = np.nanmean(covariates, axis=0)
            covariates -= self.mean_covariates_
            covariates[covariates != covariates] = 0.0
            linreg_covariates = Ridge(X=covariates, Y=X, alpha=self.alpha, idx_covariates=None, fit_intercept=False)
            self.beta_covariates = linreg_covariates.beta()
            X -= covariates.dot(self.beta_covariates)

        U, S, V = linalg.svd(X, full_matrices=False)
        explained_variance_ = (S ** 2) / n_samples
        explained_variance_ratio_ = (explained_variance_ /
                                     explained_variance_.sum())

        components_ = V

        n_components = self.n_components
        if n_components is None:
            n_components = n_features
        elif n_components == 'mle':
            raise ValueError("n_components='mle' is not supported ")
        elif not 0 <= n_components <= n_features:
            raise ValueError("n_components=%r invalid for n_features=%d"
                             % (n_components, n_features))

        if 0 < n_components < 1.0:
            # number of components for which the cumulated explained variance
            # percentage is superior to the desired threshold
            ratio_cumsum = explained_variance_ratio_.cumsum()
            n_components = np.sum(ratio_cumsum < n_components) + 1

        # Compute noise covariance using Probabilistic PCA model
        # The sigma2 maximum likelihood (cf. eq. 12.46)
        if n_components < min(n_features, n_samples):
            self.noise_variance_ = explained_variance_[n_components:].mean()
        else:
            self.noise_variance_ = 0.

        # store n_samples to revert whitening when getting covariance
        self.n_samples_ = n_samples

        self.components_ = components_[:n_components]
        self.explained_variance_ = explained_variance_[:n_components]
        explained_variance_ratio_ = explained_variance_ratio_[:n_components]
        self.explained_variance_ratio_ = explained_variance_ratio_
        self.n_components_ = n_components

        return U, S, V

    def transform(self, X, covariates=None):
        """Apply the dimensionality reduction on X.

        X is projected on the first principal components previous extracted
        from a training set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.
        covariates: array-like, shape (n_samples, n_covariates)
            Training vector, where n_samples in the number of samples and
            n_covariates is the number of covariates to be taken out form the PCA.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)

        """
        check_is_fitted(self, 'mean_')

        X = check_array(X, force_all_finite=False)

        if self.mean_ is not None and self.copy:
            X = X - self.mean_
        else:
            X -= self.mean_
        X[X != X] = 0.0

        if self.beta_covariates is not None:
            covariates = check_array(covariates, force_all_finite=False)
            if self.copy:
                covariates = covariates - self.mean_covariates_
            else:
                covariates -= self.mean_covariates_
            covariates[covariates!=covariates] = 0.0
            X -= covariates.dot(self.beta_covariates)
        else:
            assert covariates is None, "covariates provided, but not included in fit"

        X_transformed = fast_dot(X, self.components_.T)
        if self.whiten:
            X_transformed /= np.sqrt(self.explained_variance_)
        if self.normalize:
            X_transformed /= np.sqrt(self.explained_variance_.sum())
        return X_transformed

    def inverse_transform(self, X, covariates=None):
        """Transform data back to its original space using `n_components_`.

        Returns an input X_original whose transform would be X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_components)
            New data, where n_samples is the number of samples
            and n_components is the number of components. X represents
            data from the projection on to the principal components.
            This can have less than n_components columns, resulting in a low rank reconstruction (1st k eigen vectors).
        covariates: array-like, shape (n_samples, n_covariates) (default None)
            the covariates matrix

        Returns
        -------
        X_original array-like, shape (n_samples, n_features)
        """
        check_is_fitted(self, 'mean_')

        n_components_X = X.shape[1]
        if n_components_X<self.n_components_:
            components = self.components_[:n_components_X]
        else:
            components = self.components_

        if self.whiten:
            result = fast_dot(
                X,
                np.sqrt(self.explained_variance_[:, np.newaxis]) *
                components) + self.mean_
        elif self.normalize:
            result = fast_dot(X, components) * np.sqrt(self.explained_variance_.sum()) + self.mean_
        else:
            result = fast_dot(X, components) + self.mean_

        if self.beta_covariates is not None:
            covariates = check_array(covariates, force_all_finite=False)
            if self.copy:
                covariates = covariates - self.mean_covariates_
            else:
                covariates -= self.mean_covariates_
            covariates[covariates != covariates] = 0.0
            result += covariates.dot(self.beta_covariates)
        else:
            assert covariates is None, "covariates provided, but not included in fit"

        return result

    def to_pickle(self, file_base_name, protocol=-1, allow_pickle=True, fix_imports=True):
        """
        stores pca object to disk

        Parameters:
             file_base_name :  The base file name to write to. Will create three files:
                file_base_name+".pickle", file_base_name+"_components.npy", and file_base_name+"_covariates.npy"
            protocol : Protocol to be used in cPickle (default -1)
            allow_pickle : bool, optional
                Allow saving object arrays using Python pickles. Reasons for disallowing pickles include security
                (loading pickled data can execute arbitrary code) and portability (pickled objects may not be loadable
                on different Python installations, for example if the stored objects require libraries that are not
                available, and not all pickled data is compatible between Python 2 and Python 3). Default: True
            fix_imports : bool, optional
                Only useful in forcing objects in object arrays on Python 3 to be pickled in a Python 2 compatible way.
                If fix_imports is True, pickle will try to map the new Python 3 names to the old module names used in
                Python 2, so that the pickle data stream is readable with Python 2.
        """

        beta_covariates_set = self.beta_covariates is not None
        components_set = self.components_ is not None
        data = {
            'n_components':         self.n_components,
            'copy':                 self.copy,
            'whiten':               self.whiten,
            'normalize':            self.normalize,
            'beta_covariates_set':  beta_covariates_set,
            'components_set':       components_set,
            'alpha':                self.alpha,
            'mean_':                self.mean_,
            'mean_covariates_':     self.mean_covariates_,
            'noise_variance_':      self.noise_variance_,
            'explained_variance_':  self.explained_variance_,
            'explained_variance_ratio_': self.explained_variance_ratio_,
            'n_components_':         self.n_components_,
            'n_samples_':           self.n_samples_,
        }
        pickle_file = file(file_base_name + ".pickle", "wb")
        cPickle.dump(data, pickle_file, protocol=protocol)
        pickle_file.close()
        if data['components_set']:
            np.save(file_base_name + "_components", self.components_, allow_pickle=allow_pickle, fix_imports=fix_imports)
        if data['beta_covariates_set']:
            np.save(file_base_name + "_covariates", self.beta_covariates, allow_pickle=allow_pickle,
                    fix_imports=fix_imports)

    @staticmethod
    def from_pickle(file_base_name, mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII'):
        """
        loads pca object from disk

        Parameters:
            file_base_name :  The base file name to read. Assumes that there are three files:
                file_base_name+".pickle", file_base_name+"_components.npy", and file_base_name+"_covariates.npy"
            mmap_mode : {None, 'r+', 'r', 'w+', 'c'}, optional
                If not None, then memory-map the file, using the given mode (see numpy.memmap for a detailed description
                of the modes). A memory-mapped array is kept on disk. However, it can be accessed and sliced like any
                ndarray. Memory mapping is especially useful for accessing small fragments of large files without
                reading the entire file into memory.
            allow_pickle : bool, optional
                Allow loading pickled object arrays stored in npy files. Reasons for disallowing pickles include
                security, as loading pickled data can execute arbitrary code. If pickles are disallowed, loading object
                arrays will fail. Default: True
            fix_imports : bool, optional
                Only useful when loading Python 2 generated pickled files on Python 3, which includes npy/npz files
                containing object arrays. If fix_imports is True, pickle will try to map the old Python 2 names to the
                new names used in Python 3.
            encoding : str, optional
                What encoding to use when reading Python 2 strings. Only useful when loading Python 2 generated pickled
                files on Python 3, which includes npy/npz files containing object arrays. Values other than 'latin1',
                'ASCII', and 'bytes' are not allowed, as they can corrupt numerical data. Default: 'ASCII'

        Returns:
            CovariatesPCA object
        """
        pickle_file = file(file_base_name + ".pickle", "r")
        data = cPickle.load(pickle_file)
        pickle_file.close()

        pca = CovariatesPCA(n_components=data['n_components'], copy=data['copy'], whiten=data['whiten'], normalize=data['normalize'], alpha=data['alpha'])

        pca.mean_ = data['mean_']
        pca.noise_variance_ = data['noise_variance_']
        pca.explained_variance_ = data['explained_variance_']
        pca.explained_variance_ratio_ = data['explained_variance_ratio_']
        pca.n_components_ = data['n_components_']
        pca.mean_covariates_ = data['mean_covariates_']
        pca.n_samples_ = data['n_samples_']

        if data['beta_covariates_set']:
            pca.beta_covariates = np.load(file_base_name + "_covariates.npy", mmap_mode=mmap_mode, allow_pickle=allow_pickle, fix_imports=fix_imports)
        if data['components_set']:
            pca.components_ = np.load(file_base_name + "_components.npy", mmap_mode=mmap_mode, allow_pickle=allow_pickle, fix_imports=fix_imports)

        return pca


class KernelPCA(BaseEstimator, TransformerMixin):
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
    def __init__(self, n_components=None, copy=True, normalize=False, add_bias=True, threshold=1e-8):
        self.n_components = n_components
        self.copy = copy
        self.normalize = normalize
        self.centerer = KernelCentererCovariates(add_bias=add_bias)
        self.noise_variance_ = None
        self.U = None
        self.s = None
        self.n_components_ = None
        self.n_samples = None
        self.threshold = threshold

    def fit(self, K, y=None, covariates=None):
        """Fit the model with X.

        Parameters
        ----------
        K: array-like, shape (n_samples_train, n_samples_train)
            Training data, where n_samples_train in the number of samples
            used to train the PCA.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._fit(K, covariates=covariates)
        return self

    def fit_transform(self, K, y=None, covariates=None):
        """Fit the model with K and apply the dimensionality reduction to K.

        Parameters
        ----------
        K : array-like, shape (n_samples_train, n_samples_train)
            Training data, where n_samples_train is the number of samples used to train the PCA.
        covariates: array-like, shape (n_samples_train, n_covariates)
            Training vector, where n_samples_train in the number of samples used to train the PCA.
            n_covariates is the number of covariates to be taken out form the PCA.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)

        """
        s, U = self._fit(K, covariates=covariates)

        # X_new = X * V = U * S * V^T * V = U * S
        components = self.U * np.sqrt(self.s[np.newaxis,:])

        if self.normalize:
            components /= np.sqrt(self.s.sum())
        return components

    def _fit(self, K, covariates=None):
        """Fit the model on X

        Parameters
        ----------
        K : array-like, shape (n_samples_train, n_samples_train)
            Training data, where n_samples_train is the number of samples used to train the PCA.
        covariates: array-like, shape (n_samples_train, n_covariates)
            Training vector, where n_samples_train is the number of samples used to train the PCA and
            n_covariates is the number of covariates to be taken out form the PCA.

        Returns
        -------
        s, U : ndarrays
            The eigenvalue decomposition of the input data, copied and centered when
            requested.
        """
        K = check_array(K, force_all_finite=True)
        self.n_samples = K.shape[0]
        K = as_float_array(K, copy=self.copy, force_all_finite=True)
        # Center data
        self.centerer.fit(K=K, covariates=covariates)
        K = self.centerer.transform(K=K, copy=self.copy, covariates=covariates)

        s, U = linalg.eigh(K)

        idx_pos = s >= self.threshold
        self.s = s[idx_pos]
        self.U = U[:, idx_pos]
        if (self.n_components is not None) and (self.n_components < self.s.shape[0]):
            self.noise_variance_ = self.s[:self.n_components].mean()
            idx_last = -(self.n_components + 1)
            self.s = self.s[:idx_last:-1]
            self.U = self.U[:,:idx_last:-1]
            self.n_components_ = self.n_components
        else:
            self.noise_variance_ = 0.0
            self.s = self.s[::-1]
            self.U = self.U[:,::-1]
            self.n_components_ = self.s.shape[0]
        return self.s, self.U

    def transform(self, K, covariates=None, copy=True):
        """Apply the dimensionality reduction on X.

        X is projected on the first principal components previous extracted
        from a training set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.
        covariates: array-like, shape (n_samples, n_covariates)
            Training vector, where n_samples in the number of samples and
            n_covariates is the number of covariates to be taken out form the PCA.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)

        """

        K_transformed = self.centerer.transform(K=K, covariates=covariates, copy=copy)
        K_transformed = K_transformed.dot(self.U) / np.sqrt(self.s[np.newaxis,:])
        if self.normalize:
            K_transformed /= np.sqrt(self.s.sum())
        return K_transformed

    def to_pickle(self, file_base_name, protocol=-1, allow_pickle=True, fix_imports=True):
        """
        stores pca object to disk

        Parameters:
             file_base_name :  The base file name to write to. Will create three files:
                file_base_name+".pickle", file_base_name+"_components.npy", and file_base_name+"_covariates.npy"
            protocol : Protocol to be used in cPickle (default -1)
            allow_pickle : bool, optional
                Allow saving object arrays using Python pickles. Reasons for disallowing pickles include security
                (loading pickled data can execute arbitrary code) and portability (pickled objects may not be loadable
                on different Python installations, for example if the stored objects require libraries that are not
                available, and not all pickled data is compatible between Python 2 and Python 3). Default: True
            fix_imports : bool, optional
                Only useful in forcing objects in object arrays on Python 3 to be pickled in a Python 2 compatible way.
                If fix_imports is True, pickle will try to map the new Python 3 names to the old module names used in
                Python 2, so that the pickle data stream is readable with Python 2.
        """

        data = {
            's':                self.s,
            'n_samples':        self.n_samples,
            'threshold':        self.threshold,
            'n_components':     self.n_components,
            'copy':             self.copy,
            'normalize':        self.normalize,
            'centerer_set':     self.centerer is not None,
            'U_set':            self.U is not None,
            'noise_variance_':  self.noise_variance_,
            'n_components_':    self.n_components_,
        }
        pickle_file = file(file_base_name + ".pickle", "wb")
        cPickle.dump(data, pickle_file, protocol=protocol)
        pickle_file.close()
        if data['centerer_set']:
            self.centerer.to_pickle(file_base_name + "_centerer", protocol=protocol)
        if data['U_set']:
            np.save(file_base_name + "_U", self.U, allow_pickle=allow_pickle,
                    fix_imports=fix_imports)

    @staticmethod
    def from_pickle(file_base_name, mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII'):
        """
        loads pca object from disk

        Parameters:
            file_base_name :  The base file name to read. Assumes that there are three files:
                file_base_name+".pickle", file_base_name+"_components.npy", and file_base_name+"_covariates.npy"
            mmap_mode : {None, 'r+', 'r', 'w+', 'c'}, optional
                If not None, then memory-map the file, using the given mode (see numpy.memmap for a detailed description
                of the modes). A memory-mapped array is kept on disk. However, it can be accessed and sliced like any
                ndarray. Memory mapping is especially useful for accessing small fragments of large files without
                reading the entire file into memory.
            allow_pickle : bool, optional
                Allow loading pickled object arrays stored in npy files. Reasons for disallowing pickles include
                security, as loading pickled data can execute arbitrary code. If pickles are disallowed, loading object
                arrays will fail. Default: True
            fix_imports : bool, optional
                Only useful when loading Python 2 generated pickled files on Python 3, which includes npy/npz files
                containing object arrays. If fix_imports is True, pickle will try to map the old Python 2 names to the
                new names used in Python 3.
            encoding : str, optional
                What encoding to use when reading Python 2 strings. Only useful when loading Python 2 generated pickled
                files on Python 3, which includes npy/npz files containing object arrays. Values other than 'latin1',
                'ASCII', and 'bytes' are not allowed, as they can corrupt numerical data. Default: 'ASCII'

        Returns:
            CovariatesPCA object
        """
        pickle_file = file(file_base_name + ".pickle", "r")
        data = cPickle.load(pickle_file)
        pickle_file.close()
        pca = KernelPCA(n_components=data['n_components'], copy=data['copy'], normalize=data['normalize'], threshold=data['threshold'])
        pca.s = data['s']
        pca.noise_variance_ = data['noise_variance_']
        pca.n_components_ = data['n_components_']
        pca.n_components = data['n_components']
        pca.n_samples = data['n_samples']

        if data['centerer_set']:
            pca.centerer = KernelCentererCovariates.from_pickle(file_base_name=file_base_name+"_centerer")
        if data['U_set']:
            pca.U = np.load(file_base_name + "_U.npy", mmap_mode=mmap_mode, allow_pickle=allow_pickle, fix_imports=fix_imports)
        return pca


if __name__ == '__main__':
    N = 1000
    D = 100
    D_cov = 2
    D_components = 5
    covariates_original = np.array(np.random.uniform(size=(N,D_cov + D_components)) > 0.5, dtype=np.float)
    beta = np.random.normal(size=(D_cov + D_components,D))
    X_original = np.random.normal(size=(N,D)) + covariates_original.dot(beta) + np.random.normal(size=(1,D)) * 5.0

    X = X_original.copy()
    covariates = covariates_original[:,:D_cov].copy()
    X[np.random.uniform(size=(N,D))>1.95] = np.nan
    covariates[np.random.uniform(size=(N,D_cov))>1.95] = np.nan

    #create a PCA model without covariates, without normalizing the principal components
    pca_0 = CovariatesPCA(n_components=5, alpha=0.0)
    pc0 = pca_0.fit_transform(X=X)
    pc0_ = pca_0.transform(X=X)
    pc0__ = pca_0.transform(X=X)
    X_pred0 = pca_0.inverse_transform(X=pc0__)
    res0 = X_original-X_pred0
    print "\n----------\nunnormalized principal components, no covariates\n----------"
    print "norm of the principal components        = %.6f, %.6f" % (np.mean(pc0*pc0), np.mean((pc0__-pc0_)**2.0) )
    print "reconstruction error without covariates = %.6f" % np.mean(res0*res0)

    pca_0.to_pickle("test_nocovariates")
    pca_0_reloaded = CovariatesPCA.from_pickle("test_nocovariates", mmap_mode="r")
    pc0_reload = pca_0_reloaded.transform(X=X)
    print "difference between original PCA and reloaded PCA without covariates = %.6f" % (np.mean((pc0_reload-pc0)**2.0))

    #create a model with PCA covariates, without normalizing the principal components
    pca_1_ = CovariatesPCA(n_components=5, alpha=0.0, normalize=False)
    pc1 = pca_1_.fit_transform(X=X, covariates=covariates)
    pc0 = pca_1_.fit_transform(X=X, covariates=covariates)
    pc0_ = pca_1_.transform(X=X, covariates=covariates)
    pc0__ = pca_1_.transform(X=X, covariates=covariates)
    X_pred1_ = pca_1_.inverse_transform(X=pc1, covariates=covariates)
    res1 = X_original-X_pred1_
    print "\n----------\nunnormalized principal components, with covariates\n----------"
    print "norm of the principal components        = %.6f, %.6f" % (np.mean(pc0*pc0), np.mean((pc0__-pc0_)**2.0) )
    print "reconstruction error with covariates    = %.6f" % np.mean(res1*res1)

    pc1 = pca_1_.transform(X=X, covariates=covariates)
    X_pred1_ = pca_1_.inverse_transform(X=pc1, covariates=covariates)
    res1 = X_original-X_pred1_
    print "\n----------\nunnormalized principal components, with covariates\n----------"
    print "norm of the principal components        = %.6f" % np.mean(pc1*pc1)
    print "reconstruction error with covariates    = %.6f" % np.mean(res1*res1)

    pca_1_.to_pickle(file_base_name="test_unnormalized")

    #create a model with PCA covariates, without normalizing the principal components performing a lower rank reconstruction
    pca_1__ = CovariatesPCA.from_pickle(file_base_name="test_unnormalized", mmap_mode="r")
    pc1 = pca_1_.transform(X=X, covariates=covariates)
    X_pred1_ = pca_1_.inverse_transform(X=pc1, covariates=covariates)
    res1 = X_original-X_pred1_
    print "\n----------\nunnormalized principal components, with covariates\n----------"
    print "norm of the principal components        = %.6f" % np.mean(pc1*pc1)
    print "reconstruction error with covariates    = %.6f" % np.mean(res1*res1)

    #create a PCA model without covariates, normalizing the principal components
    pca_0 = CovariatesPCA(n_components=5, normalize=True, alpha=0.0)
    pc0_n = pca_0.fit_transform(X=X)
    pc0_n_ = pca_0.transform(X=X)
    X_pred0 = pca_0.inverse_transform(X=pc0_n)
    res0 = X_original-X_pred0
    print "\n----------\nnormalized principal components, no covariates\n----------"
    print "norm of the principal components        = %.6f, %.6f" % (np.mean(pc0_n*pc0_n), np.mean((pc0_n-pc0_n_)**2.0) )
    print "reconstruction error without covariates = %.6f" % np.mean(res0*res0)

    #create a model with PCA covariates, normalizing the principal components
    pca_1 = CovariatesPCA(n_components=5, normalize=True, alpha=0.0)
    pc1_n = pca_1.fit_transform(X=X, covariates=covariates)
    pc1_n_ = pca_1.transform(X=X, covariates=covariates)
    X_pred1 = pca_1.inverse_transform(X=pc1_n, covariates=covariates)
    res1 = X_original-X_pred1
    print "\n----------\nnormalized principal components, with covariates\n----------"
    print "norm of the principal components        = %.6f, %.6f" % (np.mean(pc1_n*pc1_n), np.mean((pc1_n-pc1_n_)**2.0) )
    print "reconstruction error with covariates    = %.6f" % np.mean(res1*res1)


    pca_1.to_pickle(file_base_name="test_normalized")


    #create a model with PCA covariates, normalizing the principal components
    pca_1_n_reload = CovariatesPCA.from_pickle(file_base_name="test_normalized")
    pc1_n_ = pca_1_n_reload.transform(X=X, covariates=covariates)
    X_pred1 = pca_1_n_reload.inverse_transform(X=pc1_n_, covariates=covariates)
    res1 = X_original-X_pred1
    print "\n----------\nnormalized principal components, with covariates reloaded without memmap\n----------"
    print "norm of the principal components        = %.6f" % np.mean(pc1_n*pc1_n)
    print "reconstruction error with covariates    = %.6f" % np.mean(res1*res1)

    #create a model with PCA covariates, normalizing the principal components
    pca_1_n_reload_memmap = CovariatesPCA.from_pickle(file_base_name="test_normalized", mmap_mode="r")
    pc1_n_ = pca_1_n_reload_memmap.transform(X=X, covariates=covariates)
    X_pred1 = pca_1_n_reload_memmap.inverse_transform(X=pc1_n_, covariates=covariates)
    res1 = X_original-X_pred1
    print "\n----------\nnormalized principal components, with covariates reloaded with memmap\n----------"
    print "norm of the principal components        = %.6f" % np.mean(pc1_n*pc1_n)
    print "reconstruction error with covariates    = %.6f" % np.mean(res1*res1)


    #create a model with PCA covariates, normalizing the principal components performing a lower rank reconstruction
    pca_1_reload_memmap = CovariatesPCA.from_pickle(file_base_name="test_normalized", mmap_mode="r")
    pc1_ = pca_1_n_reload_memmap.transform(X=X, covariates=covariates)
    X_pred1 = pca_1_n_reload_memmap.inverse_transform(X=pc1_n_[:,:3], covariates=covariates)
    res1 = X_original-X_pred1
    print "\n----------\nnormalized principal components, with covariates reloaded with memmap, rank 3 reconstruction\n----------"
    print "norm of the principal components        = %.6f" % np.mean(pc1_n*pc1_n)
    print "reconstruction error with covariates    = %.6f" % np.mean(res1*res1)

    print "\n\n----------\nKernel PCA\n----------"

    N = 1000
    D = 100
    D_cov = 2
    D_components = 5
    covariates_original = np.array(np.random.uniform(size=(N,D_cov + D_components)) > 0.5, dtype=np.float)
    beta = np.random.normal(size=(D_cov + D_components,D))
    X_original = np.random.normal(size=(N,D)) + covariates_original.dot(beta) + np.random.normal(size=(1,D)) * 5.0

    N_train = 500
    X = X_original.copy()
    covariates = covariates_original[:,:D_cov].copy()
    covariates_train = covariates[:N_train]
    X_train = X[:N_train]

    import linreg
    ridge = linreg.Ridge(X=covariates_train, Y=X_train, alpha=0.0)
    X_ = X - ridge.predict(X=covariates)

    K_ = X_.dot(X_.T)
    K_train_ = K_[:N_train,:N_train]

    # X[np.random.uniform(size=(N,D))>1.95] = np.nan
    # covariates[np.random.uniform(size=(N,D_cov))>1.95] = np.nan

    K = X.dot(X.T)
    preprocess = KernelCentererCovariates(add_bias=True)

    K_train = K[:N_train,:N_train]

    preprocess.fit(K=K_train, covariates=covariates_train)
    K_preprocessed = preprocess.transform(K[:,:N_train], covariates=covariates)

    print "the maximum absolute difference between the two preprocessing methods is %.3e." % (np.absolute(K_[:,:N_train] - K_preprocessed).max())

    n_components = 5

    kpca = KernelPCA(n_components=n_components)
    pc_k_train = kpca.fit_transform(K=K_train, covariates=covariates_train)
    pc_k = kpca.transform(K=K[:,:N_train], covariates=covariates)

    cpca = CovariatesPCA(n_components=n_components)
    pc_c_train = cpca.fit_transform(X=X_train, covariates=covariates_train)
    pc_c = cpca.transform(X=X, covariates=covariates)

    print "max abs difference between pc_k and pc_c = %.3e" % ((np.absolute(pc_c) - np.absolute(pc_k)).max())

    kpca.to_pickle(file_base_name="kpca_test")
    kpca_reload = KernelPCA.from_pickle("kpca_test", mmap_mode="r")
    pc_k_reload = kpca_reload.transform(K=K[:,:N_train], covariates=covariates)
    print "max abs difference between pc_k and pc_k_reload = %.3e" % ((np.absolute(pc_k_reload - pc_k)).max())
