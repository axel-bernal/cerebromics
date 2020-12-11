import numpy as np
from datastack.ml.algorithms.utils import generate_intervals
from pysnptools.snpreader import Bed
import cPickle
from datastack.ml.algorithms.kernel_snps_new import LinearSnpKernel
from datastack.ml.algorithms.pca_covariates import CovariatesPCA, KernelPCA
from datastack.ml.algorithms.snp_preprocessing import SnpPreprocessing

class SnpPca(SnpPreprocessing):
    def __init__(self, beta_shape=0.8, n_components=None, dtype=np.float64):
        SnpPreprocessing.__init__(self, beta_shape=beta_shape, dtype=dtype)
        self.n_components = n_components
        self.pcs = None
        self.pca = None
        self.X_multiplier = None
        self.sid = None
        pass

    def fit(self, snp_reader, covariates=None):
        self.init_preprocess(snp_reader=snp_reader)

        self.sid = snp_reader.sid.copy()
        X = self.preprocess_block(snp_reader=snp_reader)

        # pass
        self.X_multiplier = np.sqrt(X.shape[0]) / np.sqrt((X * X).sum())
        X *= self.X_multiplier

        # fit the PCA
        self.pca = CovariatesPCA(n_components=self.n_components)
        self.pcs = self.pca.fit_transform(X=X, covariates=covariates)
        return self

    def transform(self, snp_reader, covariates=None, unsave=False):
        N = snp_reader.iid.shape[0]

        if unsave:
            assert (snp_reader.sid.shape[0] == self.sid.shape[0]), "need the same number of SNPs in unsave mode"
        else:
            assert (snp_reader.sid == self.sid).all(), "need the same SNPs in save mode"
        X_test = self.transform_snps_block(snp_reader=snp_reader, unsave=unsave) 
        X_test *= self.X_multiplier
        return self.pca.transform(X_test, covariates=covariates)

    def to_pickle(self, file_base_name, protocol=-1,  allow_pickle=True, fix_imports=True):
        data = {
            "mean":             self.mean,
            "N_observed":       self.N_observed,
            "snp_multiplier":   self.snp_multiplier,
            "std":              self.std,
            "N_snps":           self.N_snps,
            "N":                self.N,
            "beta_shape":       self.beta_shape,
            "idx_SNC":          self.idx_SNC,
            "X_multiplier":     self.X_multiplier,
            "pcs":              self.pcs,
            "sid":              self.sid,
            "dtype":            self.dtype,
            "n_components":     self.n_components,
        }

        pickle_file = file(file_base_name + ".pickle", "wb")
        cPickle.dump(data, pickle_file, protocol=protocol)
        pickle_file.close() 
        self.pca.to_pickle(file_base_name=file_base_name+"_pca", protocol=protocol, allow_pickle=allow_pickle,
                    fix_imports=fix_imports)

    @staticmethod
    def from_pickle(file_base_name, mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII'):
        pickle_file = file(file_base_name + ".pickle", "r")
        data = cPickle.load(pickle_file)
        pickle_file.close()
        snp_pca = SnpPca(beta_shape=data['beta_shape'], n_components=data['n_components'], dtype=data['dtype'])
        snp_pca.mean = data['mean']
        snp_pca.N_observed = data['N_observed']
        snp_pca.snp_multiplier = data['snp_multiplier']
        snp_pca.std = data['std']
        snp_pca.N_snps = data['N_snps']
        snp_pca.N = data['N']
        snp_pca.X_multiplier = data["X_multiplier"]
        snp_pca.idx_SNC = data["idx_SNC"]
        snp_pca.sid = data["sid"]
        snp_pca.pcs = data["pcs"]
        
        if data["sid"] is not None:
            snp_pca.pca = CovariatesPCA.from_pickle(file_base_name=file_base_name+"_pca", mmap_mode=mmap_mode, allow_pickle=allow_pickle, fix_imports=fix_imports, encoding=encoding)

            assert snp_pca.N_snps == snp_pca.sid.shape[0], "number of SNPs does not match"
        else:
            assert snp_pca.N_snps is None, "need sid for a trained model"
            assert snp_pca.N is None, "need sid for a trained model"
        return snp_pca


if __name__ == '__main__':
    basefilename_train = "/Users/clippert/Data/face/HG19_face_MAF_05_gVCF_geno05_hwe001"

    # plink flips the bed bits.
    # basefilename_test = "/Users/clippert/Data/face/HG19_face_MAF_05_gVCF_geno05_hwe001_single_sample"


    n_components = 5
    covariates_train = None
    covariates_test = None
    beta_shape = 0.8
    snp_reader_train = Bed(basefilename_train)[:,::1000]
    # snp_reader_test = Bed(basefilename_test)#[:,:100]
    snp_reader_test = snp_reader_train[0:1,:]
    kernel = LinearSnpKernel(beta_shape=beta_shape)

    K_train = kernel.fit_transform(snp_reader_train, blocksize=10000)
    kpca = KernelPCA(n_components=n_components)
    pc_k_train = kpca.fit_transform(K=K_train, covariates=covariates_train)

    K_test = kernel.transform(snp_reader=snp_reader_test, blocksize=10000)
    pc_k_test = kpca.transform(K=K_test, covariates=covariates_test)

    print np.absolute(K_train[0:1] - K_test).max()

    kernel.to_pickle("K_file_pca_primal")

    kernel2 = LinearSnpKernel.from_pickle("K_file_pca_primal", snp_reader_train)
    K_test2 = kernel2.transform(snp_reader=snp_reader_test, blocksize=10000)    

    print np.absolute(K_test - K_test2).max()  

    snp_pca = SnpPca(beta_shape=beta_shape, n_components=n_components, dtype=np.float64)
    snp_pca.fit(snp_reader=snp_reader_train, covariates=covariates_train)
    pc_primal = snp_pca.pcs

    print "note that the signs can be different"
    print (np.absolute(pc_k_train) - np.absolute(pc_primal)).max()

    print np.absolute(pc_k_train / pc_primal).max() - 1.0

    pc_primal_test = snp_pca.transform(snp_reader=snp_reader_test, covariates=covariates_test)

    print "note that the signs can be different2"
    print (np.absolute(pc_k_test) - np.absolute(pc_primal_test)).max()

    print np.absolute(pc_k_test / pc_primal_test).max() - 1.0

    snp_pca.to_pickle("snp_pca_primal")
    snp_pca_reload = SnpPca.from_pickle("snp_pca_primal")
    pc_primal_test_reload = snp_pca.transform(snp_reader=snp_reader_test, covariates=covariates_test)    
    print np.absolute(pc_primal_test - pc_primal_test_reload).max()
    