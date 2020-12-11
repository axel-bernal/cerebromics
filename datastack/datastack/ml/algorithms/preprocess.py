from math import log, sqrt
import numpy as np
from scipy import linalg
from scipy.special import gammaln
from sklearn.utils import as_float_array
from sklearn.utils import check_array
from sklearn.base import BaseEstimator, TransformerMixin
import cPickle
from sklearn.utils.validation import FLOAT_DTYPES

class KernelCentererCovariates(BaseEstimator, TransformerMixin):
    """Center a kernel matrix

    Let K(x, z) be a kernel defined by phi(x)^T phi(z), where phi is a
    function mapping x to a Hilbert space. KernelCenterer centers (i.e.,
    normalize to have zero mean) the data without explicitly computing phi(x).
    It is equivalent to centering phi(x) with
    sklearn.preprocessing.StandardScaler(with_std=False).

    Read more in the :ref:`User Guide <kernel_centering>`.
    """
    def __init__(self, add_bias=True):
        self.add_bias = add_bias
        self.X = None
        self.Xpinv = None
        self.XdKXXd = None

    def fit(self, K, y=None, covariates=None):
        """Fit KernelCenterer

        Parameters
        ----------
        K : numpy array of shape [n_samples, n_samples]
            Kernel matrix.

        Returns
        -------
        self : returns an instance of self.
        """
        n_samples = K.shape[0]
        if covariates is None:
            covariates = np.zeros((K.shape[0], 0))
        if self.add_bias:
            self.X = np.concatenate((covariates, np.ones((n_samples, 1))), 1)
        else:
            self.X = covariates
        self.Xpinv = linalg.pinv(self.X)
        XdK = self.Xpinv.dot(K)
        self.XdKXXd = XdK - XdK.dot(self.X).dot(self.Xpinv)
        return self

    def transform(self, K, y=None, copy=True, covariates=None):
        """Center kernel matrix.

        Parameters
        ----------
        K : numpy array of shape [n_samples1, n_samples2]
            Kernel matrix.

        copy : boolean, optional, default True
            Set to False to perform inplace computation.

        Returns
        -------
        K_new : numpy array of shape [n_samples1, n_samples2]
        """

        K = check_array(K, copy=copy, dtype=FLOAT_DTYPES)
        n_samples = K.shape[0]
        if covariates is None:
            covariates = np.zeros((K.shape[0], 0))
        if self.add_bias:
            X = np.concatenate((covariates, np.ones((n_samples, 1))), 1)
        else:
            X = covariates

        K -= K.dot(self.X).dot(self.Xpinv)
        K -= X.dot(self.XdKXXd)
        return K

    def to_pickle(self, file_base_name, protocol=-1):
        """
        stores KernelCentererCovariates object to disk

        Parameters:
             file_base_name :  The base file name to write to. Will create three files:
                file_base_name+".pickle", file_base_name+"_components.npy", and file_base_name+"_covariates.npy"
            protocol : Protocol to be used in cPickle (default -1)
        """
        data = {
            'add_bias':             self.add_bias,
            'X':                    self.X,
            'Xpinv':                self.Xpinv,
            'XdKXXd':               self.XdKXXd,
        }
        pickle_file = file(file_base_name + ".pickle", "wb")
        cPickle.dump(data, pickle_file, protocol=protocol)
        pickle_file.close()

    @staticmethod
    def from_pickle(file_base_name):
        """
        loads pca object from disk

        Parameters:
            file_base_name :  The base file name to read. Assumes that there are three files:
                file_base_name+".pickle", file_base_name+"_components.npy", and file_base_name+"_covariates.npy"

        Returns:
            KernelCentererCovariates object
        """
        pickle_file = file(file_base_name + ".pickle", "r")
        data = cPickle.load(pickle_file)
        pickle_file.close()

        result = KernelCentererCovariates(add_bias=data['add_bias'])
        result.X = data["X"]
        result.Xpinv = data["Xpinv"]
        result.XdKXXd = data["XdKXXd"]
        return result


if __name__ == '__main__':
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

    preprocess.fit(K=K_train,covariates=covariates_train)
    K_preprocessed = preprocess.transform(K[:,:N_train], covariates=covariates)

    print "the maximum absolute difference between the two preprocessing methods is %.3e." % (np.absolute(K_[:,:N_train] - K_preprocessed).max())

