import numpy as np
from datastack.ml.algorithms.utils import generate_intervals
from pysnptools.snpreader import Bed
import cPickle
from datastack.ml.algorithms.snp_preprocessing import SnpPreprocessing

class LinearSnpKernel(SnpPreprocessing):
    def __init__(self, beta_shape=0.8):
        SnpPreprocessing.__init__(self, beta_shape=beta_shape)

        self.snp_reader_train = None
        self.K = None
        self.K_multiplier = None
        pass

    def fit(self, snp_reader, blocksize=10000):
        self.init_preprocess(snp_reader=snp_reader)

        self.snp_reader_train = snp_reader
        intervals = generate_intervals(blocksize=blocksize, total_snps=self.N_snps)
        self.K = np.zeros((self.N, self.N))

        for i in xrange(len(intervals) - 1):
            start = intervals[i]
            stop = intervals[i+1]

            X = self.preprocess_block(snp_reader=snp_reader, start=start, stop=stop)

            self.K += X.dot(X.T)
            # pass
        self.K_multiplier = 1.0 / self.K.diagonal().mean()
        self.K *= self.K_multiplier
        return self

    def fit_transform(self, snp_reader, blocksize=10000):
        self.fit(snp_reader=snp_reader, blocksize=blocksize)
        return self.K

    def transform(self, snp_reader, blocksize=10000, unsave=False):
        """
        transform (build a kernel to the training data) of the people present in snp_reader

        Args:
            snp_reader:     pysnptools Bed reader
            blocksize:      number of SNPs loaded at once
            unsave:         if True, skip the check for the same SNP-IDs (cpra strings)
                            useful when combining HG38 and HG19

        Returns 
            K :             SNP kernel matrix
        """

        N = snp_reader.iid.shape[0]

        if unsave:
            assert (snp_reader.sid.shape[0] == self.snp_reader_train.sid.shape[0]), "need the same number of SNPs in unsave mode"
        else:
            assert (snp_reader.sid == self.snp_reader_train.sid).all(), "need the same SNPs in save mode"
        
        intervals = generate_intervals(blocksize=blocksize, total_snps=self.N_snps)
        K = np.zeros((N, self.N))
        for i in xrange(len(intervals) - 1):
            # ipdb.set_trace()
            start = intervals[i]
            stop = intervals[i+1]
            # transform the SNPs using the same parameters as the training set
            X_test = self.transform_snps_block(snp_reader=snp_reader, start=start, stop=stop, unsave=unsave)
            X_train = self.transform_snps_block(snp_reader=self.snp_reader_train, start=start, stop=stop, unsave=unsave)
            # add to kernel
            K += X_test.dot(X_train.T)
            # pass
        K *= self.K_multiplier
        return K

    def to_pickle(self, file_base_name, protocol=-1):
        data = {
        "mean":             self.mean,
        "N_observed":       self.N_observed,
        "snp_multiplier":   self.snp_multiplier,
        "std":              self.std,
        "N_snps":           self.N_snps,
        "K":                self.K,
        "N":                self.N,
        "beta_shape":       self.beta_shape,
        "K_multiplier":     self.K_multiplier,
        "idx_SNC":          self.idx_SNC,
        }

        pickle_file = file(file_base_name + ".pickle", "wb")
        cPickle.dump(data, pickle_file, protocol=protocol)
        pickle_file.close()       


    @staticmethod
    def from_pickle(file_base_name, snp_reader_train=None):
        pickle_file = file(file_base_name + ".pickle", "r")
        data = cPickle.load(pickle_file)
        pickle_file.close()
        kernel = LinearSnpKernel(beta_shape=data['beta_shape'])
        kernel.mean = data['mean']
        kernel.N_observed = data['N_observed']
        kernel.snp_multiplier = data['snp_multiplier']
        kernel.std = data['std']
        kernel.N_snps = data['N_snps']
        kernel.K = data['K']
        kernel.N = data['N']
        kernel.K_multiplier = data["K_multiplier"]
        kernel.idx_SNC = data["idx_SNC"]

        if snp_reader_train is not None:
            kernel.snp_reader_train = snp_reader_train
            assert kernel.N_snps == snp_reader_train.sid.shape[0], "number of SNPs does not match"
            assert kernel.N == snp_reader_train.iid.shape[0], "number of people does not match"
        else:
            assert kernel.N_snps is None, "need snp_reader_train for a trained model"
            assert kernel.N is None, "need snp_reader_train for a trained model"
        return kernel


if __name__ == '__main__':
    basefilename_train = "/Users/clippert/Data/face/HG19_face_MAF_05_gVCF_geno05_hwe001"

    # plink flips the bed bits.
    # basefilename_test = "/Users/clippert/Data/face/HG19_face_MAF_05_gVCF_geno05_hwe001_single_sample"

    snp_reader_train = Bed(basefilename_train)[:,::1000]
    # snp_reader_test = Bed(basefilename_test)#[:,:100]
    snp_reader_test = snp_reader_train[0:1,:]
    kernel = LinearSnpKernel(beta_shape=0.8)

    K_train = kernel.fit_transform(snp_reader_train, blocksize=10000)
    K_test = kernel.transform(snp_reader=snp_reader_test, blocksize=10000)

    print np.absolute(K_train[0:1] - K_test).max()

    # kernel.to_pickle("K_file")

    kernel2 = LinearSnpKernel.from_pickle("K_file", snp_reader_train)
    K_test2 = kernel2.transform(snp_reader=snp_reader_test, blocksize=10000)    

    print np.absolute(K_test - K_test2).max()