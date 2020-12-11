"""
 2015 Copyright by Human Longevity Inc. Written By Eun Yong Kang

"""




import scipy as sp
import scipy.linalg as LA
import scipy.optimize as OPT
import pdb
import matplotlib.pylab as plt
import time
import numpy as np




class LmmLasso:

    def __init__(self, X, y, K, mu=0.5, n_reps=100, f_subset=0.5, numintervals=100,ldeltamin=-5,ldeltamax=5,rho=1,alpha=1):
        self.X = X
        self.y = y
        self.K = K
        self.mu = mu
        self.n_reps=n_reps
        self.f_subset=f_subset
        #self.numintervals=numintervals
        #self.ldeltamin=ldeltamin
        #self.ldeltamax=ldeltamax
        #self.rho=rho
        #self.alpha=alpha



    def stability_selection(self, **kwargs):
        """
        run stability selection

        Input:
        X: SNP matrix: n_s x n_f
        y: phenotype:  n_s x 1
        K: kinship matrix: n_s x n_s
        mu: l1-penalty

        n_reps:   number of repetitions
        f_subset: fraction of datasets that is used for creating one bootstrap

        output:
        selection frequency for all SNPs: n_f x 1
        """
        time_start = time.time()
        [n_s,n_f] = self.X.shape
        n_subsample = sp.ceil(self.f_subset * n_s)
        freq = sp.zeros(n_f)

        for i in range(self.n_reps):
            print 'Iteration %d'%i
            idx = sp.random.permutation(n_s)[:n_subsample]
            res = self.train(self.X[idx], self.K[idx][:,idx], self.y[idx], self.mu, **kwargs)
            snp_idx = (res['weights']!=0).flatten()
            print "Number of non-zero coefficient : " + str(sum(snp_idx))
            freq[snp_idx] += 1.

        freq /= self.n_reps
        time_end = time.time()
        time_diff = time_end - time_start
        print '... finished in %.2fs'%(time_diff)
        return freq



    def train(self, X, K, y, mu, numintervals=100,ldeltamin=-5,ldeltamax=5,rho=1,alpha=1,debug=False,ldelta0=None):
        """
        train linear mixed model lasso

        Input:
        X: SNP matrix: n_s x n_f
        y: phenotype:  n_s x 1
        K: kinship matrix: n_s x n_s
        mu: l1-penalty parameter
        numintervals: number of intervals for delta linesearch
        ldeltamin: minimal delta value (log-space)
        ldeltamax: maximal delta value (log-space)
        rho: augmented Lagrangian parameter for Lasso solver
        alpha: over-relatation parameter (typically ranges between 1.0 and 1.8) for Lasso solver

        Output:
        results
        """
        #print 'train LMM-Lasso'
        #print '...l1-penalty: %.2f'%mu

        time_start = time.time()
        [n_s,n_f] = X.shape
        assert X.shape[0]==y.shape[0], 'dimensions do not match'
        assert K.shape[0]==K.shape[1], 'dimensions do not match'
        assert K.shape[0]==X.shape[0], 'dimensions do not match'
        if y.ndim==1:
            y = sp.reshape(y,(n_s,1))

        # train null model
        if ldelta0 is None:
            S,U,ldelta0,monitor_nm = self.train_nullmodel(y,K,numintervals,ldeltamin,ldeltamax,debug=debug)
        else:
            S,U=LA.eigh(K)
            monitor_nm={}
            monitor_nm['nllgrid'] = None
            monitor_nm['ldeltagrid'] = None
            monitor_nm['ldeltaopt'] = None
            monitor_nm['nllopt'] = None

        # train lasso on residuals
        delta0 = sp.exp(ldelta0)
        Sdi = 1./(S+delta0)
        Sdi_sqrt = sp.sqrt(Sdi)
        SUX = sp.dot(U.T,X)
        SUX = SUX * sp.tile(Sdi_sqrt,(n_f,1)).T
        SUy = sp.dot(U.T,y)
        SUy = SUy * sp.reshape(Sdi_sqrt,(n_s,1))

        w,monitor_lasso = self.train_lasso(SUX,SUy,mu,rho,alpha,debug=debug)

        time_end = time.time()
        time_diff = time_end - time_start
        print '... finished in %.2fs'%(time_diff)

        res = {}
        res['ldelta0'] = ldelta0
        res['weights'] = w
        res['time'] = time_diff
        res['monitor_lasso'] = monitor_lasso
        res['monitor_nm'] = monitor_nm
        return res


    def predict(self, y_t,X_t,X_v,K_tt,K_vt,ldelta,w):
        """
        predict the phenotype

        Input:
        y_t: phenotype: n_train x 1
        X_t: SNP matrix: n_train x n_f
        X_v: SNP matrix: n_val x n_f
        K_tt: kinship matrix: n_train x n_train
        K_vt: kinship matrix: n_val  x n_train
        ldelta: kernel parameter
        w: lasso weights: n_f x 1

        Output:
        y_v: predicted phenotype: n_val x 1
        """
        print 'predict LMM-Lasso'

        assert y_t.shape[0]==X_t.shape[0], 'dimensions do not match'
        assert y_t.shape[0]==K_tt.shape[0], 'dimensions do not match'
        assert y_t.shape[0]==K_tt.shape[1], 'dimensions do not match'
        assert y_t.shape[0]==K_vt.shape[1], 'dimensions do not match'
        assert X_v.shape[0]==K_vt.shape[0], 'dimensions do not match'
        assert X_t.shape[1]==X_v.shape[1], 'dimensions do not match'
        assert X_t.shape[1]==w.shape[0], 'dimensions do not match'

        [n_train,n_f] = X_t.shape
        n_test = X_v.shape[0]

        if y_t.ndim==1:
            y_t = sp.reshape(y_t,(n_train,1))
        if w.ndim==1:
            w = sp.reshape(w,(n_f,1))

        delta = sp.exp(ldelta)
        idx = w.nonzero()[0]

        if idx.shape[0]==0:
            return sp.dot(K_vt,LA.solve(K_tt + delta*sp.eye(n_train),y_t))

        y_v = sp.dot(X_v[:,idx],w[idx]) + sp.dot(K_vt, LA.solve(K_tt + delta*sp.eye(n_train),y_t-sp.dot(X_t[:,idx],w[idx])))
        return y_v




    def train_lasso(self, X, y, mu, rho=1,alpha=1,max_iter=5000,abstol=1E-4,reltol=1E-2,zero_threshold=1E-3,debug=False):
        """
        train lasso via Alternating Direction Method of Multipliers:
        min_w  0.5*sum((y-Xw)**2) + mu*|z|

        Input:
        X: design matrix: n_s x n_f
        y: outcome:  n_s x 1
        mu: l1-penalty parameter
        rho: augmented Lagrangian parameter
        alpha: over-relatation parameter (typically ranges between 1.0 and 1.8)

        the implementation is a python version of Boyd's matlab implementation of ADMM-Lasso, which can be found at:
        http://www.stanford.edu/~boyd/papers/admm/lasso/lasso.html

        more information about ADMM can be found in the paper linked at:
        http://www.stanford.edu/~boyd/papers/distr_opt_stat_learning_admm.html

        In particular, you can use any other Lasso-Solver instead. For the experiments, reported in the paper,
        we used the l1-solver from the package scikits. We didn't apply it here to avoid third-party packages.
        """
        if debug:
            print '... train lasso'

        # init
        [n_s,n_f] = X.shape
        w = sp.zeros((n_f,1))
        z = sp.zeros((n_f,1))
        u = sp.zeros((n_f,1))

        monitor = {}
        monitor['objval'] = []
        monitor['r_norm'] = []
        monitor['s_norm'] = []
        monitor['eps_pri'] = []
        monitor['eps_dual'] = []

        # cache factorization
        U = self.factor(X,rho)

        # save a matrix-vector multiply
        Xy = sp.dot(X.T,y)

        if debug:
            print 'i\tobj\t\tr_norm\t\ts_norm\t\teps_pri\t\teps_dual'

        for i in range(max_iter):
            # w-update
            q = Xy + rho*(z-u)
            w = q/rho - sp.dot(X.T,LA.cho_solve((U,False),sp.dot(X,q)))/rho**2

            # z-update with relaxation
            zold = z
            w_hat = alpha*w + (1-alpha)*zold
            z = self.soft_thresholding(w_hat+u, mu/rho)

            # u-update
            u = u + (w_hat - z)

            monitor['objval'].append(self.lasso_obj(X,y,w,mu,z))
            monitor['r_norm'].append(LA.norm(w-z))
            monitor['s_norm'].append(LA.norm(rho*(z-zold)))
            monitor['eps_pri'].append(sp.sqrt(n_f)*abstol + reltol*max(LA.norm(w),LA.norm(z)))
            monitor['eps_dual'].append(sp.sqrt(n_f)*abstol + reltol*LA.norm(rho*u))

            if debug:
                print '%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f'%(i,monitor['objval'][i],monitor['r_norm'][i],monitor['s_norm'][i],monitor['eps_pri'][i],monitor['eps_dual'][i])

            if monitor['r_norm'][i]<monitor['eps_pri'][i] and monitor['r_norm'][i]<monitor['eps_dual'][i]:
                break


        w[sp.absolute(w)<zero_threshold]=0
        return w,monitor


    def nLLeval(self, ldelta,Uy,S):
        """
        evaluate the negative log likelihood of a random effects model:
        nLL = 1/2(n_s*log(2pi) + logdet(K) + 1/ss * y^T(K + deltaI)^{-1}y,
        where K = USU^T.

        Uy: transformed outcome: n_s x 1
        S:  eigenvectors of K: n_s
        ldelta: log-transformed ratio sigma_gg/sigma_ee
        """
        n_s = Uy.shape[0]
        delta = sp.exp(ldelta)

        # evaluate log determinant
        Sd = S+delta
        ldet = sp.sum(sp.log(Sd))

        # evaluate the variance
        Sdi = 1.0/Sd
        Uy = Uy.flatten()
        ss = 1./n_s * (Uy*Uy*Sdi).sum()

        # evalue the negative log likelihood
        nLL=0.5*(n_s*sp.log(2.0*sp.pi)+ldet+n_s+n_s*sp.log(ss));

        return nLL



    def train_nullmodel(self, y, K, numintervals=100,ldeltamin=-5,ldeltamax=5,debug=False):
        """
        train random effects model:
        min_{delta}  1/2(n_s*log(2pi) + logdet(K) + 1/ss * y^T(K + deltaI)^{-1}y,

        Input:
        X: SNP matrix: n_s x n_f
        y: phenotype:  n_s x 1
        K: kinship matrix: n_s x n_s
        mu: l1-penalty parameter
        numintervals: number of intervals for delta linesearch
        ldeltamin: minimal delta value (log-space)
        ldeltamax: maximal delta value (log-space)
        """
        if debug:
            print '... train null model'

        n_s = y.shape[0]

        # rotate data
        S,U = LA.eigh(K)
        Uy = sp.dot(U.T,y)

        # grid search
        nllgrid=sp.ones(numintervals+1)*sp.inf
        ldeltagrid=sp.arange(numintervals+1)/(numintervals*1.0)*(ldeltamax-ldeltamin)+ldeltamin
        nllmin=sp.inf
        for i in sp.arange(numintervals+1):
            nllgrid[i]=self.nLLeval(ldeltagrid[i],Uy,S);

        # find minimum
        nll_min = nllgrid.min()
        ldeltaopt_glob = ldeltagrid[nllgrid.argmin()]

        # more accurate search around the minimum of the grid search
        for i in sp.arange(numintervals-1)+1:
            if (nllgrid[i]<nllgrid[i-1] and nllgrid[i]<nllgrid[i+1]):
                ldeltaopt,nllopt,iter,funcalls = OPT.brent(self.nLLeval,(Uy,S),(ldeltagrid[i-1],ldeltagrid[i],ldeltagrid[i+1]),full_output=True);
                if nllopt<nllmin:
                    nllmin=nllopt;
                    ldeltaopt_glob=ldeltaopt;

        monitor = {}
        monitor['nllgrid'] = nllgrid
        monitor['ldeltagrid'] = ldeltagrid
        monitor['ldeltaopt'] = ldeltaopt_glob
        monitor['nllopt'] = nllmin

        return S,U,ldeltaopt_glob,monitor


    def factor(self, X,rho):
        """
        computes cholesky factorization of the kernel K = 1/rho*XX^T + I

        Input:
        X design matrix: n_s x n_f (we assume n_s << n_f)
        rho: regularizaer

        Output:
        L  lower triangular matrix
        U  upper triangular matrix
        """
        n_s,n_f = X.shape
        K = 1/rho*sp.dot(X,X.T) + sp.eye(n_s)
        U = LA.cholesky(K)
        return U



    def soft_thresholding(self, w,kappa):
        """
        Performs elementwise soft thresholding for each entry w_i of the vector w:
        s_i= argmin_{s_i}  rho*abs(s_i) + rho/2*(x_i-s_i) **2
        by using subdifferential calculus

        Input:
        w vector nx1
        kappa regularizer

        Output:
        s vector nx1
        """
        n_f = w.shape[0]
        zeros = sp.zeros((n_f,1))
        s = np.max(sp.hstack((w-kappa,zeros)),axis=1) - np.max(sp.hstack((-w-kappa,zeros)),axis=1)
        s = sp.reshape(s,(n_f,1))
        return s


    def lasso_obj(self, X,y,w,mu,z):
        """
        evaluates lasso objective: 0.5*sum((y-Xw)**2) + mu*|z|

        Input:
        X: design matrix: n_s x n_f
        y: outcome:  n_s x 1
        mu: l1-penalty parameter
        w: weights: n_f x 1
        z: slack variables: n_fx1

        Output:
        obj
        """
        return 0.5*((sp.dot(X,w)-y)**2).sum() + mu*sp.absolute(z).sum()
