import datastack.cerebro.linear_model.ridge as cerebro_ridge
import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
import scipy.stats as st
import warnings

from scipy import pi
from cached import Cached, cached


class Ridge(cerebro_ridge.Ridge):

    def __init__(self, X=None, Y=None, alpha=1.0, idx_covariates=None, fit_intercept=True):
        """
        implementation of ridge penalized least squares

        See OLS_introduction.tex for additional information and derivations.
    
        Args:
            X               [N x D] numpy.ndarray of inputs (D features for N samples)
            Y               numpy.array of outputs (targets for N samples)
            alpha           the regularizer, can be a scalar, 1-dim ndarray of length D, or  2-dim ndarray of size DxD
            idx_covariates    D dimensional index of the columns in X that are covariates and should not be penalized
        """
        
        """
        Sorry, Christoph, it was too hard to handle per-column preprocessing without dealing
        with frames; and if the user specifies preprocessing as None we need to support
        the same interface here.
        """
        super(Ridge, self).__init__(alpha=alpha, fit_intercept=fit_intercept, idx_covariates=idx_covariates)
        warnings.warn("'datastack.ml.algorithms.linreg.Ridge' is a pass-"
                      "through to 'datastack.cerebro.linear_model.Ridge' "
                      "and will be removed at some point. New code should "
                      "use 'datastack.cerebro.linear_model.Ridge'",
                      UserWarning)
        self.X = X
        self.Y = Y
        if X is not None and Y is not None:
            self.fit(X=X, y=Y)
    
    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, value):
        warnings.warn("Setting X has no computational effect; "
                      "use 'fit' instead",
                      DeprecationWarning)
        self._X = value

    @property
    def Y(self):
        return self._Y

    @Y.setter
    def Y(self, value):
        warnings.warn("Setting Y has no computational effect; "
                      "use 'fit' instead",
                      DeprecationWarning)
        self._Y = value

    def A_div_alpha(self, A):
        if issubclass(type(self.alpha), np.ndarray) and len(self.alpha.shape)==2:    #matrix
            try:
                ret = sla.solve(self.alpha, A.T, sym_pos=True).T
            except:
                print "Ridge low rank"
                ret = la.lstsq(self.alpha, A.T)[0]
        elif issubclass(type(self.alpha), np.ndarray):                                #diagonal matrix in vector form
            return A / self.alpha[np.newaxis,:]
        elif self.alpha>0:                                                                #scalar
                return A / self.alpha
        else:
            raise Exception("alpha has to be > 0.0 for N<D case, found %f" % self.alpha)

    def beta_subset(self, ind_train, ind_test):
        """
        compute beta for a subset of the data
        """
        if self.fit_intercept:
            raise NotImplementedError("beta_subset currently not compatible with fit_intercept")

        X_tr = self.X[ind_train]
        Y_tr = self.Y[ind_train]
        if X_tr.shape[0]>=X_tr.shape[1]:
            return cerebro_ridge._beta_small_p(self.alpha, self.idx_covariates, X_tr, Y_tr)
        else:
            return cerebro_ridge._beta_large_p(self.alpha, self.idx_covariates, X_tr, Y_tr)

    def beta_subset_efficient(self, ind_train, ind_test):
        """
        compute beta for a subset of the data
        """
        if self.fit_intercept:
            raise NotImplementedError("beta_subset_efficient currently not compatible with fit_intercept")
        X_tr = self.X[ind_train]
        Y_tr = self.Y[ind_train]
        if X_tr.shape[0]>=X_tr.shape[1]:
            # twong XXX: There's no way for this to work - beta_small_p()
            # is not defined anywhere.
            return self.beta_small_p(X=X_tr, y=Y_tr)

        X = self.X
        X_rot = self.A_div_alpha(A=X)

        XX = X.dot(X_rot.T)
        XXY = XX.dot(Y)

        XX.flat[::XX.shape[0]+1] += 1.0
        XXi = la.inv(XX)

        XX_te = XXi[ind_test][:,ind_test]
        XX_te_tr = XXi[ind_test][:,ind_train]
        DC = sla.solve(XX_te, XX_te_tr, sym_pos=True)
        BDC = XXi[ind_train][:,ind_train] - XX_te_tr.T.dot(DC)
        KiY = BDC.dot(Y_tr)
        beta_train = X_tr.T.dot(KiY)
        return beta_train

    def beta(self):
        """
        the regression weights
        """
        if self.X.shape[0]>=self.X.shape[1]:
            return cerebro_ridge._beta_small_p(self.alpha, self.idx_covariates, self.X, self.Y)
        else:
            return cerebro_ridge._beta_large_p(self.alpha, self.idx_covariates, self.X, self.Y)

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
        # twong: A horrid hack to deal with legacy pickled models. The
        # home-grown linreg did not define its own fit_intercept member
        # variable.
        if 'fit_intercept' not in vars(self):
            self.fit_intercept = self._fit_intercept
            self.idx_covariates = self._idx_covariates
            self._intercept_y = self._intercept_Y
            self._beta = cerebro_ridge._beta(self.alpha, self.idx_covariates, self._X, self._Y)
        return super(Ridge, self).predict(X)

class OLS(Cached):

    """
    implementation of ordinary least squares

    See OLS_introduction.tex for additional information and derivations.
    """
    def __init__(self, X, Y, add_intercept=True, rank_threshold=1e-10):
        """
        implementation of ordinary least squares

        See OLS_introduction.tex for additional information and derivations.
    
        Args:
            X               [N x D] numpy.ndarray of inputs (D features for N samples)
            Y               numpy.array of outputs (targets for N samples)
            add_intercept    add an intercept? (default: True)
            rank_threshold    where to cut off singular values and eigenvalues for pseudo-inverses?
        """
        Cached.__init__(self)
        self.debug = False
        self.X = X
        self.Y = Y
        self.add_intercept = add_intercept
        self.rank_threshold = rank_threshold

    @property
    def Y(self):
        return self._fixdim(self._Y)

    def _fixdim(self,Y,matrix=True):
        if self.dim_Y == 1:
            if matrix:
                return Y[:,0]
            else:
                return Y[0]
        else:
            return Y

    @cached(["X"])
    def svd_X(self):
        if self.X.shape[1]:
            return la.svd(self.X-self.intercept_X(), full_matrices=False)
        else:
            return (np.zeros((self.N,0)),np.zeros(0),np.zeros((0,0)))

    @property
    def add_intercept(self):
        return self._add_intercept

    @add_intercept.setter
    def add_intercept(self, value):
        self.clear_cache("X")
        self._add_intercept = value

    @property
    def rank_threshold(self):
        return self._rank_threshold

    @rank_threshold.setter
    def rank_threshold(self, value):
        self.clear_cache("X")
        self._rank_threshold = value


    def XXinv(self):
        Xpinv = self.Xpinv()
        return Xpinv.dot(Xpinv.T)

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, value):
        self.clear_cache("X")
        self._X = value

    @Y.setter
    def Y(self, Y):
        self.clear_cache("Y")
        self.dim_Y = len(Y.shape)
        if len(Y.shape) == 1:
            self._Y = Y[:,np.newaxis]
        else:
            self._Y = Y

    def Xpinv(self):
        svd_X = self.svd_X()
        i_pos = svd_X[1]>self.rank_threshold
        if i_pos.any():
            Xpinv =  svd_X[0][:,i_pos] / svd_X[1][i_pos][np.newaxis,:]
            Xpinv = Xpinv.dot((svd_X[2][i_pos,:]))
            Xpinv = Xpinv.T
            if self.debug:
                Xpinv_ = la.pinv(self.X-self.intercept_X())
                diff = np.sum(np.absolute(Xpinv_ - Xpinv))
                print "diff : %.10f"%(diff)
                assert diff<=1e-10, ""
        else:
            Xpinv = np.zeros((self.X.shape[1],self.X.shape[0]))
        return Xpinv

    @cached(["X","Y"])
    def beta(self):
        """
        compute the OLS regression weights.
        """
        Y = self._Y
        return self.Xpinv().dot(Y - self.intercept())

    def intercept_X(self):
        """
        return the intercept for X
        """
        if self.add_intercept:
            return self.X.mean(0)[np.newaxis,:]
        else:
            return 0.0

    def _logl(self, var_Y=None):
        """
        compute the log likelihood under a normal distribution, maximized over the variance and the beta
        """

        if var_Y is None:
            #compute the maximum likelihood estimate of the variance
            var_Y = self._var_Y(unbiased=False)

        const =  self.N * np.log( 2.0 * pi * var_Y)
        dataterm = self._rss()/var_Y
        logl = -0.5 * (const + dataterm)
        return logl

    def intercept(self):
        """
        return the intercept for Y
        """
        if self.add_intercept:
            return self._Y.mean(0)[np.newaxis,:]
        else:
            return 0.0

    def _predict(self, Xstar=None):
        """
        predict Ystar for a new input Xstar
        """
        if Xstar is None:
            Xstar = self.X
        Xstar_zeromean = Xstar - self.intercept_X()
        beta = self.beta()
        Y_mean = self.intercept()
        Y_Xbeta = Xstar_zeromean.dot(beta)
        prediction = Y_mean + Y_Xbeta
        return prediction

    def predict(self, Xstar=None):
        """
        predict Ystar for a new input Xstar
        """
        return self._fixdim(self._predict(Xstar=Xstar))

    @property
    def N(self):
        """
        the sample size
        """
        return self.X.shape[0]

    @property
    def D(self):
        """
        the number of features
        """
        return self.X.shape[1]

    @property
    def P(self):
        return self._Y.shape[1]

    @property
    def dof(self):
        i_pos = self.svd_X()[1]>self.rank_threshold
        rank = i_pos.sum()
        if self.add_intercept:
            return rank + 1
        else:
            return rank

    def covariance_beta(self):
        """
        The covariance of the OLS estimates beta
        """
        if self.dim_Y==1:
            return self._covariance_beta()[:,:,0]
        else:
            return self._covariance_beta()

    @cached(["X", "Y"])
    def variances_beta(self):
        """
        The variances of the OLS estimates beta
        """
        return self.XXinv().diagonal()[:,np.newaxis] * self._var_Y()[np.newaxis,:]

    def standard_errors_beta(self):
        """
        The standard errors of the OLS estimates beta
        """
        return np.sqrt(self.variances_beta())

    def _covariance_beta(self):
        """
        The covariance of the OLS estimates beta
        """
        return self.XXinv()[:,:,np.newaxis] * self._var_Y()[np.newaxis,np.newaxis,:]

    def _residual(self):
        """
        The model residuals Y-X*beta
        """
        return self._Y - self._predict()

    @property
    def residual(self):
        """
        The model residuals Y-X*beta
        """
        return self._fixdim(self._residual())

    @cached(["X", "Y"])
    def _rss(self):
        """
        residual sum of squares
        """
        residual = self._residual()
        return (residual*residual).sum(0)

    @property
    def rss(self):
        """
        residual sum of squares
        """
        return self._fixdim(self._rss(),matrix=False)

    @cached(["X", "Y"])
    def _tss(self):
        """
        total sum of squares (Note that Y is mean centered only if we add an intercept)
        """
        Y_zeromean = self._Y-self.intercept()
        return (Y_zeromean*Y_zeromean).sum(0)


    @property
    def tss(self):
        """
        total sum of squares (Note that Y is mean centered only if we add an intercept)
        """
        return self._fixdim(self._tss(),matrix=False)

    def var_Y(self, unbiased=True):
        """
        OLS estimate of the variance, 
        corrected for the degrees of freedom of the model.
        
        Args:
            unbiased    Boolean, use the unbiased estimate by correcting for the DOF? (default True)
                        The unbiased estimate is the standard OLS estimate or the REML estimate.
                        The biased estimate is the maximum likelihood estimate (under a Gaussian distribution).
        """
        return self._fixdim(self._var_Y(), matrix=False)

    def _var_Y(self, unbiased=True):
        """
        OLS estimate of the variance, 
        corrected for the degrees of freedom of the model.
        
        Args:
            unbiased    Boolean, use the unbiased estimate by correcting for the DOF? (default True)
                        The unbiased estimate is the standard OLS estimate or the REML estimate.
                        The biased estimate is the maximum likelihood estimate (under a Gaussian distribution).
        """
        if unbiased:
            return self._rss()/(self.N-self.dof)
        else:
            return self._rss()/(self.N)

    def _rsquared(self):
        """
        regression r^2 of the model 
        (equivalent to the squared correlations between predictions and Y
         as explained in Section 2 of OLS_introduction.tex)
        """
        rsquared = 1.0-self._rss()/self._tss()
        return rsquared


    @property
    def rsquared(self):
        """
        regression r^2 of the model 
        (equivalent to the squared correlations between predictions and Y
         as explained in Section 2 of OLS_introduction.tex)
        """
        return self._fixdim(self._rsquared(),matrix=False)

    def _lrtest(self, i_F = None, var_Y=None):
        """
        perform a likelihood ratio test in the OLS. The F-test may obtain more than a single parameter.

        Testing involves an indicator array i_F and tests the linear hypothesis
        beta[i_F]=0.

        If both M and i_F are None (default), then the hypothesis beta=0 is tested 
        involving all entries of beta.

        See OLS_introduction.tex Section 2.2 for additional information on the F-test.
        
        Args:
            i_F        indicator numpy.array for testing linear hypothesis beta[i_F]=0 (default: None)
            var_Y    estimate of the variance of Y. If None, it will be estimated for the null and 
                    alternative models. (default: None)

        Returns:
            lrt:      The lrt-statistic:  2 * (logl_alt - logl_0)
            dof:    degrees of freedom of the test
            pvalue  The P-value for the lrt-statistic (using Chi^2_dof distribution)    
        """
        if (i_F is not None):
            ols_0 = OLS(self.X[:,i_F],Y=self.Y, add_intercept=self.add_intercept, rank_threshold=self.rank_threshold)
        else:
            ols_0 = OLS(np.zeros((self.N,0)),Y=self.Y, add_intercept=self.add_intercept, rank_threshold=self.rank_threshold)

        lrt = 2.0 * (self._logl(var_Y=var_Y) - ols_0._logl(var_Y=var_Y))
        dof_lrt =  (self.dof - ols_0.dof)
        pvalue = st.chi2.sf(lrt, dof_lrt)
        return lrt, np.ones_like(lrt) * dof_lrt, pvalue

    def lrtest(self, i_F = None, var_Y=None):
        """
        perform a likelihood ratio test in the OLS. The F-test may obtain more than a single parameter.

        Testing involves an indicator array i_F and tests the linear hypothesis
        beta[i_F]=0.

        If both M and i_F are None (default), then the hypothesis beta=0 is tested 
        involving all entries of beta.

        See OLS_introduction.tex Section 2.2 for additional information on the F-test.
        
        Args:
            i_F        indicator numpy.array for testing linear hypothesis beta[i_F]=0 (default: None)
            var_Y    estimate of the variance of Y. If None, it will be estimated for the null and 
                    alternative models. (default: None)

        Returns:
            lrt:      The lrt-statistic:  2 * (logl_alt - logl_0)
            dof:    degrees of freedom of the test
            pvalue  The P-value for the lrt-statistic (using Chi^2_dof distribution)    
        """
        lrt, dof, pvalue = self._lrtest(i_F = i_F, var_Y=var_Y)
        return self._fixdim(lrt,matrix=False), self._fixdim(dof,matrix=False), self._fixdim(pvalue,matrix=False)

    def ftest(self, M=None, i_F = None):
        """
        perform an F-test in the OLS. The F-test may obtain more than a single parameter.

        There are two modes, one involving an affine transformation of the beta weights using 
        the matrix M, which tests a linear hypothesis in beta, namely M.dot(beta)=0.

        The second mode involves an indicator array i_F and tests the linear hypothesis
        beta[i_F]=0.

        If both M and i_F are None (default), then the hypothesis beta=0 is tested 
        involving all entries of beta.

        See OLS_introduction.tex Section 2.2 for additional information on the F-test.
        
        Args:
            M      [P x D] ndarray for testing the hypothesis M.dot(beta)=0 (default: None)
            i_F        indicator numpy.array for testing linear hypothesis beta[i_F]=0 (default: None)

        Returns:
            fstats:  The F-statistic
            pvalues  The P-value for the F-statistic (using Fisher's F distribution)    
        """
        fstats,pvalues=self._ftest(M=M,i_F=i_F)
        return self._fixdim(fstats,matrix=False), self._fixdim(pvalues,matrix=False)


    def _ftest(self, M=None, i_F = None):
        """
        perform an F-test in the OLS. The F-test may obtain more than a single parameter.

        There are two modes, one involving an affine transformation of the beta weights using 
        the matrix M, which tests a linear hypothesis in beta, namely M.dot(beta)=0.

        The second mode involves an indicator array i_F and tests the linear hypothesis
        beta[i_F]=0.

        If both M and i_F are None (default), then the hypothesis beta=0 is tested 
        involving all entries of beta.

        See OLS_introduction.tex Section 2.2 for additional information on the F-test.
        
        Args:
            M      [P x D] ndarray for testing the hypothesis M.dot(beta)=0 (default: None)
            i_F        indicator numpy.array for testing linear hypothesis beta[i_F]=0 (default: None)

        Returns:
            fstats:  The F-statistic
            pvalues  The P-value for the F-statistic (using Fisher's F distribution)    
        """
        covariance_beta = self._covariance_beta()
        betas = self.beta()

        fstats = np.zeros(self.P)
        pvalues = np.zeros(self.P)

        for p in xrange(self.P):
            covariance_beta_p = covariance_beta[:,:,p]
            beta_p = betas[:,p]
            if (M is not None):
                assert i_F is None, "M and i_F are mutually exclusive."
                cov_params = M.dot(covariance_beta_p.dot(M.T))
                beta = M.dot(beta_p)
                pass
            elif (i_F is not None):
                assert M is None, "M and i_F are mutually exclusive."
                cov_params = covariance_beta_p[i_F,:][:,i_F]
                beta = beta_p[i_F]
                pass
            else:
                cov_params = covariance_beta_p[:,:]
                beta = beta_p
                pass

            if self.D:
                S_,U_ = la.eigh(cov_params)
                S = S_[S_>self.rank_threshold]
                U = U_[:,S_>self.rank_threshold]
            else:
                S = np.zeros((0))
                U = np.zeros((0,0))
            dof_numerator = S.shape[0]
            Ubeta = U.T.dot(beta)
            SiUbeta = Ubeta[S>self.rank_threshold]/S[S>self.rank_threshold]
            fstats[p] = Ubeta.T.dot(SiUbeta)/dof_numerator
            dof_denominator = self.N-self.dof
            pvalues[p] = st.f.sf(fstats[p],dof_numerator,dof_denominator)
        return fstats, pvalues

    def ttest(self, onesided=True):
        """
        perform a one or two-sided t-test in the OLS.
        The t-test is computed for each single parameter.
        For each d in 1...D we test the hypothesis beta_d = 0

        See OLS_introduction.tex Section 1.1 for additional information on the t-test.

        Args:
            twosided     Boolean. perform a One-sided (beta>=0: True) or two-sided (beta><0: False)
                        test. (default False) 
        Returns:
            tstats  numpy. array of length D containing the t-statistics
            pvals   The P-values for the t-statistics (using Students's t-distribution)    
        """

        tstats = self.beta() / self.standard_errors_beta()
        if onesided:
            pvals = st.t.sf(tstats,self.N-self.dof)
        else:
            pvals = 2.0 * st.t.sf(np.absolute(tstats),self.N-self.dof)
        return tstats, pvals


if __name__ == "__main__":
    import sys
    if "debug_linreg" in sys.argv:
        N=10000
        D=10
        P=2
        X = np.random.randn(N,D)
        beta = 0.04*np.random.randn(D,P)
        Y = np.random.randn(N,P) + X.dot(beta)
        ols = OLS(X=X,Y=Y)
        i_F = np.random.rand(D)>0.5
        i_F[0]=True

        print "ftest i_F: " + str(ols.ftest(i_F=i_F))
        print "lrtest i_F: " + str(ols.lrtest(i_F=i_F))
        print "ttest : stat %.6f  pval %.6f" % (ols.ttest()[0][0,0], ols.ttest()[1][0,0])
        i_F=np.array([0])
        print "ftest i_F: " + str(ols.ftest(i_F=i_F))
        print "lrtest first weight: stat %.6f dof %i   pval %.6f" % (ols.lrtest(i_F=i_F)[0][0], ols.lrtest(i_F=i_F)[1][0], ols.lrtest(i_F=i_F)[2][0])


        ols = OLS(X=X,Y=Y[:,0])
        i_F = np.random.rand(D)>0.5

        print "ftest i_F: " + str(ols.ftest(i_F=i_F))
        print "lrtest i_F: " + str(ols.lrtest(i_F=i_F))
        print "ttest : stat %.6f  pval %.6f" % (ols.ttest()[0][0,0], ols.ttest()[1][0,0])
        i_F=np.array([0])
        print "ftest i_F: " + str(ols.ftest(i_F=i_F))
        print "lrtest first weight: stat %.6f dof %i   pval %.6f" % (ols.lrtest(i_F=i_F)[0], ols.lrtest(i_F=i_F)[1], ols.lrtest(i_F=i_F)[2])

    elif "test_ridge" in sys.argv:
        N=100
        D=100000
        P=2
        X = np.random.randn(N,D)
        #X[:,0] = 1.0

        beta = 0.04*np.random.randn(D,P)
        beta[0,:] *= 1000.0
        Y = np.random.randn(N,P) * 50.0 + X.dot(beta)
        Y[:,0] +=4
        Y[:,0] +=40
        idx_covariates = np.zeros(D, dtype = np.bool)
        idx_covariates[0] = True
        ridge = Ridge(X=X, Y=Y, alpha=1.0, idx_covariates=idx_covariates, fit_intercept=True)

        betar = ridge.beta()
        betar_l = cerebro_ridge._beta_large_p(ridge.alpha, ridge.idx_covariates, ridge.X, ridge.Y)

        if 0:
            # twong XXX: beta_small_p() doesn't exist. _beta_small_p()
            # does...
            ols = OLS(X=X,Y=Y, add_intercept=False)
            betar_s = ridge.beta_small_p(X=ridge.X, Y=ridge.Y)

            beta_ols = ridge.beta_small_p(X=ridge.X, Y=ridge.Y)
            beta_ols_ = ols.beta()

        idx_covariates_ = np.zeros(D+1, dtype = np.bool)
        idx_covariates_[0] = True
        idx_covariates_[-1] = True
        X_ = np.concatenate((X,np.ones((X.shape[0],1))),1)
        ridge_nointercept = Ridge(X=X_, Y=Y, alpha=1.0, idx_covariates=idx_covariates_, fit_intercept=False)
        ridge_nointercept.alpha = 1.0
        perm = np.random.permutation(N)
        i_te = perm[:(N/10)]
        i_tr = perm[(N/10):]

        bs1 = ridge_nointercept.beta_subset(ind_train=i_tr, ind_test=i_te)
        bs2 = ridge_nointercept.beta_subset_efficient(ind_train=i_tr, ind_test=i_te)

        import sklearn.grid_search as gs
        gridsearch = gs.GridSearchCV(estimator=ridge, param_grid={"alpha": [0.001, 0.01, 0.1, 1.01, 10.0]})#, scoring=sklearn.metrics.make_scorer(score))
        gridsearch.fit(X=X, y=Y[:,0])

        import sklearn.linear_model
        ridge_sk = sklearn.linear_model.Ridge()
        gridsearch_sk = gs.GridSearchCV(estimator=ridge_sk, param_grid={"alpha": [0.001, 0.01, 0.1, 1.01, 10.0]})#, scoring=sklearn.metrics.make_scorer(score))
        gridsearch_sk.fit(X=X, y=Y[:,0])

        print "best score scikit Ridge : %.4f" % gridsearch_sk.best_score_
        print "best score HLI Ridge : %.4f" % gridsearch.best_score_
        gridsearch_cov = gs.GridSearchCV(fit_params={"idx_covariates": idx_covariates},estimator=ridge, param_grid={"alpha": [0.001, 0.01, 0.1, 1.01, 10.0]})#, scoring=sklearn.metrics.make_scorer(score))
        gridsearch_cov.fit(X=X, y=Y[:,0])
        print "best score HLI Ridge with covariates : %.4f" % gridsearch_cov.best_score_
    if 1:
        import datastack.ml.baseregress as baseregress
        from sklearn import linear_model

        targets = {'right':'facepheno.hand.strength.right.m1', 'left':'facepheno.hand.strength.left.m1'}
        size = ['facepheno.height', 'dynamic.FACE.pheno.v1.bmi']
        eyecolor = ['dynamic.FACE.eyecolor.v1_visit1.*']
        
        base = baseregress.BaseRegress(targets)

        params = {'alpha': [ 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]}
        est0 = {'est' : linear_model.Ridge(), 'params' : params}
        est1 = {'est' : linear_model.Lasso(), 'params' : params}
        est2 = {'est': Ridge(), 'params': params}

        base.estimatorsToRun['size'] = [est0, est2]

        base.covariates['size'] = size
        base.covariates['eyecolor'] = eyecolor
        base.run(with_aggregate_covariates=False)

        print base.metrics_df