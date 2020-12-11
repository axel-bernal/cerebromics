import scipy.stats as st
import numpy as np
from datastack.ml.algorithms.utils import generate_intervals
from pysnptools.snpreader import Bed
import cPickle


class SnpPreprocessing(object):
    def __init__(self, beta_shape=0.8, dtype=np.float64):
        self.mean = None
        self.N_observed = None
        self.snp_multiplier = None
        self.std = None
        self.N_snps = None
        self.N = None
        self.beta_shape = beta_shape
        self.idx_SNC = None
        self.dtype=dtype

    def init_preprocess(self, snp_reader):
        self.N_snps = snp_reader.sid.shape[0]
        self.N = snp_reader.iid.shape[0]
        self.mean = np.zeros(self.N_snps)
        self.std = np.zeros(self.N_snps)
        self.N_observed = np.zeros(self.N_snps, dtype=np.int)
        self.snp_multiplier = np.zeros(self.N_snps)
        self.idx_SNC = np.zeros(self.N_snps, dtype=np.bool)

    def preprocess_block(self, snp_reader, start=None, stop=None):
        if start is None:
            start = 0
        if stop is None:
            stop = snp_reader.sid.shape[0]

        snps_block = snp_reader[:, start:stop].read(dtype=self.dtype)

        X = snps_block.val
        i_observed = X == X
        # mean
        self.mean[start:stop] = np.nanmean(X, axis=0)
        # std
        self.std[start:stop] = np.nanstd(X, axis=0)
        self.N_observed[start:stop] = (i_observed.sum(0))

        X -= self.mean[np.newaxis, start:stop]
        # fill in NaN
        X[~i_observed] = 0.0

        # multiply by x
        self.idx_SNC[start:stop] = self.std[start:stop] <= 1e-8

        if self.beta_shape is None:
            snp_multiplier = 1.0 / self.std[start:stop]
        else:
            snp_multiplier = st.beta.pdf(self.mean[start:stop] / 2.0, self.beta_shape, self.beta_shape)
        snp_multiplier[self.N_observed[start:stop] == 0] = 0.0
        snp_multiplier[self.idx_SNC[start:stop]] = 0.0
        self.snp_multiplier[start:stop] = snp_multiplier

        X *= self.snp_multiplier[np.newaxis, start:stop]
        # add to kernel
        return X

    def transform_snps_block(self, snp_reader, start=None, stop=None, unsave=False):
        
        if start is None:
            start = 0
        if stop is None:
            stop = self.sid.shape[0]
        
        snps_block_test = snp_reader[:, start:stop].read(dtype=self.dtype)
        # load the SNP values
        X_test = snps_block_test.val
        # mean
        X_test -= self.mean[np.newaxis, start:stop]
        # fill in NaN
        X_test[X_test!=X_test] = 0.0
        # multiply by x
        X_test *= self.snp_multiplier[np.newaxis, start:stop]
        return X_test