
import variance_components
import variance_components_lowrank
reload(variance_components)
reload(variance_components_lowrank)
from variance_components import *
from variance_components_lowrank import MixedRegression
import variance_components_opt as opt


if __name__ == '__main__':

    np.random.seed(4)
    N=1000
    N_outliers = N/10
    N_test = 20
    D_K=5
    D=2
    P=2
    K=[]
    G=[]
    K_train = []
    K_test = []
    K_star = []
    G_train = []
    G_star = []
    N_K = 3
    h2 = 0.7

    if 1:
        Y = np.zeros((N,P))
        alphas = 5.0*np.arange(N_K)+1
        alphas/=alphas.sum()
        for i in xrange(N_K):
            X = np.random.randn(N,D_K +i)
            
            XX = X.dot(X.T) #+ 0.0001* np.eye(N)
            var = XX.diagonal().mean()
            K.append(XX / var)
            G.append(X / np.sqrt(var))
            K_test.append(K[-1][:N_test,:][:,:N_test])
            K_train.append(K[-1][N_test:,:][:,N_test:])
            K_star.append(K[-1][:N_test,:][:,N_test:])
            G_star.append(G[-1][:N_test])
            G_train.append(G[-1][N_test:])
            beta = alphas[i] / np.sqrt(D_K+i) * np.random.randn(D_K+i,P)
            Y += (X.dot(beta))
        
        X = np.random.randn(N,D)
        X[:,0] = 1.0

        beta = (1.0) / np.sqrt(D) * np.random.randn(D,P)
        Y += (X.dot(beta))
        
        Y = np.sqrt(h2) * Y + np.sqrt(1.0-h2) * np.random.randn(N,P)
        outliers = np.random.permutation(N)[0:N_outliers]
        Y[outliers] += np.sqrt(1.0-h2)*5.0 * np.random.randn(N_outliers,P)

        Y_test = Y[:N_test,:]
        Y_train = Y[N_test:,:]
        X_test = X[:N_test,:]
        X_train = X[N_test:,:]
    else:
        import cPickle
        datafile = open("variance_components_data.pickle", "r")
        testdata = cPickle.load(datafile)
        
        X_train = testdata['X_train']
        X_test = testdata['X_test']
        Y_train = testdata['Y_train']
        Y_test = testdata['Y_test']
        K_train = testdata['K_train']
        K_test = testdata['K_test']
        K_star = testdata['K_star']
        alphas = testdata['sigma_K']
        datafile.close()        
    from linreg import OLS
    #import variance_components
    #reload(variance_components)
    
    ols = OLS(X=X_train,Y=Y_train)
    lmm = LMM(K=K_train, covariates=X_train, Y=Y_train)
    rr = MixedRegression(G=G_train, covariates=X_train, Y=Y_train)
    lmm.h2 = 0.5
    rr.h2 = lmm.h2
    nll = lmm.nLL()
    nll_rr = rr.nLL()

    pred = lmm.predict(X=X_test, K=K_star, GLS=True, marinalize_h2=False)
    pred_rr = rr.predict(X=X_test, G=G_star, GLS=True, marinalize_h2=False)

    print "mpred"
    mpred = lmm.predict(X=X_test, K=K_star, GLS=True, marinalize_h2=True)
    mpred_rr = rr.predict(X=X_test, G=G_star, GLS=True, marinalize_h2=True)
    print np.absolute(mpred - mpred_rr).sum()

    print "dXKX"
    dXKX = lmm._dXKX()
    dXKX_rr = rr._dXKX()
    for i in xrange(len(dXKX)):
        diff = dXKX[i] - dXKX_rr[i]
        print np.absolute(diff).sum()

    arand = np.random.randn(rr.D().shape[0])
    def D(x,i=None):
        sigma_K = rr.sigma_K.copy()
        rr.sigma_K = x#lmm.sigma_K
        res = rr.D()
        res = (res * arand).sum()
        rr.sigma_K = sigma_K
        return res

    def grad_D(x,i=None):
        sigma_K = rr.sigma_K.copy()
        rr.sigma_K = x
        res = rr._dD()
        res = (res * arand[np.newaxis,:]).sum(1)
        rr.sigma_K = sigma_K
        if i is None:
            return res
        else:
            return res[i]

    def logdetXKX(x,i=None):
        sigma_K = rr.sigma_K.copy()
        #lmm.sigma_K[i] = x[i]
        #lmm.sigma_K = lmm.sigma_K
        #lmm.sigma_K[i] = x[i]
        rr.sigma_K = x#lmm.sigma_K
        #print lmm.sigma_K
        res = rr._logdetXKX()
        lmm.sigma_K = sigma_K
        return res

    def grad_logdetXKX(x,i=None):
        sigma_K = rr.sigma_K.copy()
        #lmm.sigma_K[i] = x[i]
        #lmm.sigma_K = lmm.sigma_K
        rr.sigma_K = x
        #print lmm.sigma_K
        res = rr._derivative_log_det_XKX()
        #res2 = lmm._logdet()[1]
        #print res-res2
        rr.sigma_K = sigma_K
        if i is None:
            return res
        else:
            return res[i]

    x0 = np.array([0.3,-0.5,0.8])
    check_D = mcheck_grad(func=D, grad=grad_D, x0=x0, allerrs=True)
    check_logdetXKX = mcheck_grad(func=logdetXKX, grad=grad_logdetXKX, x0=x0, allerrs=True)
    opts = {
    'gradcheck' : True,
    'max_iter_opt': 200,
    }

    ####################
    # conditional version
    ####################
    #lmm.h2 = 0.0001
    #opth2 = lmm.find_h2()

    optimum = opt.opt_hyper(gpr = lmm, opts=opts)
    optimum_rr = opt.opt_hyper(gpr = rr, opts=opts)
if 1:
    res = lmm.find_h2(nGridH2=10, minH2=0.0, maxH2=0.99999, estimate_Bayes=False)
    
    nll1a = lmm.nLL()


    lmm.dof = 0.00
    lmm.dof = None
    lmm.scale2 = 2.1
    nll1000a_K = lmm.nLL()
    resa_b = lmm.find_h2(nGridH2=10, minH2=0.0, maxH2=0.99999, estimate_Bayes=False)
    resa_ml = lmm.find_h2(nGridH2=10, minH2=0.0, maxH2=0.99999, estimate_Bayes=False)
    post = lmm._posterior_h2(nGridH2=100, minH2=0.0, maxH2=0.99999)

    lmm.h2 = 0.5

    def logdet(x,i=None):
        sigma_K = lmm.sigma_K.copy()
        #lmm.sigma_K[i] = x[i]
        #lmm.sigma_K = lmm.sigma_K
        #lmm.sigma_K[i] = x[i]
        lmm.sigma_K = x#lmm.sigma_K
        #print lmm.sigma_K
        res = lmm._logdetK()
        lmm.sigma_K = sigma_K
        return res

    def grad_logdet(x,i=None):
        sigma_K = lmm.sigma_K.copy()
        #lmm.sigma_K[i] = x[i]
        #lmm.sigma_K = lmm.sigma_K
        lmm.sigma_K = x
        #print lmm.sigma_K
        res = lmm._derivative_log_det()
        #res2 = lmm._logdet()[1]
        #print res-res2
        lmm.sigma_K = sigma_K
        if i is None:
            return res
        else:
            return res[i]

    def logdetXKX(x,i=None):
        sigma_K = lmm.sigma_K.copy()
        #lmm.sigma_K[i] = x[i]
        #lmm.sigma_K = lmm.sigma_K
        #lmm.sigma_K[i] = x[i]
        lmm.sigma_K = x#lmm.sigma_K
        #print lmm.sigma_K
        res = lmm._logdetXKX()
        lmm.sigma_K = sigma_K
        return res

    def grad_logdetXKX(x,i=None):
        sigma_K = lmm.sigma_K.copy()
        #lmm.sigma_K[i] = x[i]
        #lmm.sigma_K = lmm.sigma_K
        lmm.sigma_K = x
        #print lmm.sigma_K
        res = lmm._derivative_log_det_XKX()
        #res2 = lmm._logdet()[1]
        #print res-res2
        lmm.sigma_K = sigma_K
        if i is None:
            return res
        else:
            return res[i]

    def YKY(x,i=None):
        sigma_K = lmm.sigma_K.copy()
        lmm.sigma_K = x
        res = lmm.YKY().sum()
        lmm.sigma_K = sigma_K
        return res

    def dYKY(x,i=None):
        sigma_K = lmm.sigma_K.copy()
        #lmm.sigma_K[i] = x[i]
        #lmm.sigma_K = lmm.sigma_K
        lmm.sigma_K = x
        #print lmm.sigma_K
        res = lmm._derivative_YKY()
        #res2 = lmm._logdet()[1]
        #print res-res2
        lmm.sigma_K = sigma_K
        if i is None:
            return res
        else:
            return res[i]

    def XKY(x,i=None):
        sigma_K = lmm.sigma_K.copy()
        lmm.sigma_K = x
        res = (lmm.XKY()).sum()
        lmm.sigma_K = sigma_K
        return res

    def dXKY(x,i=None):
        sigma_K = lmm.sigma_K.copy()
        #lmm.sigma_K[i] = x[i]
        #lmm.sigma_K = lmm.sigma_K
        lmm.sigma_K = x
        #print lmm.sigma_K
        res = lmm._dXKY()
        for ii in xrange(len(res)):
            mysum = res[ii].sum()
            res[ii] = mysum
        #res2 = lmm._logdet()[1]
        #print res-res2
        lmm.sigma_K = sigma_K
        if i is None:
            return res
        else:
            return res[i]

    def beta(x,i=None):
        sigma_K = lmm.sigma_K.copy()
        lmm.sigma_K = x
        res = (lmm.beta()).sum()
        lmm.sigma_K = sigma_K
        return res

    def dbeta(x,i=None):
        sigma_K = lmm.sigma_K.copy()
        #lmm.sigma_K[i] = x[i]
        #lmm.sigma_K = lmm.sigma_K
        lmm.sigma_K = x
        #print lmm.sigma_K
        res = lmm.derivative_beta()
        for ii in xrange(len(res)):
            mysum = res[ii].sum()
            res[ii] = mysum
        #res2 = lmm._logdet()[1]
        #print res-res2
        lmm.sigma_K = sigma_K
        if i is None:
            return res
        else:
            return res[i]

    def YKXXKXXKY(x,i=None):
        sigma_K = lmm.sigma_K.copy()
        lmm.sigma_K = x
        res = (lmm.XKY() * lmm.beta()).sum()
        lmm.sigma_K = sigma_K
        return res

    def dYKXXKXXKY(x,i=None):
        sigma_K = lmm.sigma_K.copy()
        #lmm.sigma_K[i] = x[i]
        #lmm.sigma_K = lmm.sigma_K
        lmm.sigma_K = x
        #print lmm.sigma_K
        res = lmm._derivative_YKXXKXXKY()
        #res2 = lmm._logdet()[1]
        #print res-res2
        lmm.sigma_K = sigma_K
        if i is None:
            return res
        else:
            return res[i]

    def r2(x,i=None):
        sigma_K = lmm.sigma_K.copy()
        lmm.sigma_K = x
        res = lmm._r2()
        lmm.sigma_K = sigma_K
        return res

    def d_r2(x,i=None):
        sigma_K = lmm.sigma_K.copy()
        #lmm.sigma_K[i] = x[i]
        #lmm.sigma_K = lmm.sigma_K
        lmm.sigma_K = x
        #print lmm.sigma_K
        res = lmm._derivative_r2()
        #res2 = lmm._logdet()[1]
        #print res-res2
        lmm.sigma_K = sigma_K
        if i is None:
            return res
        else:
            return res[i]


    def nLL(x,i=None):
        sigma_K = lmm.sigma_K.copy()
        lmm.sigma_K = x
        res = lmm.nLL()
        lmm.sigma_K = sigma_K
        return res[0]

    def d_nLL(x,i=None):
        sigma_K = lmm.sigma_K.copy()
        #lmm.sigma_K[i] = x[i]
        #lmm.sigma_K = lmm.sigma_K
        lmm.sigma_K = x
        #print lmm.sigma_K
        res = lmm.derivative_nLL()
        #res2 = lmm._logdet()[1]
        #print res-res2
        lmm.sigma_K = sigma_K
        if i is None:
            return res
        else:
            return res[i]

    def avg_nLL(x,i=None):
        h2 = lmm.h2
        sigma_K = lmm.sigma_K.copy()
        lmm.sigma_K = x
        posterior = lmm.posterior_h2()
        #logp = posterior[2]

        #log sum exp trick:
        res = -posterior[-1]# -(np.log(np.exp(logp-logp.max()).mean()) + logp.max())
        
        lmm.sigma_K = sigma_K
        lmm.h2 = h2
        return res

    def d_avg_nLL(x,i=None):
        sigma_K = lmm.sigma_K.copy()
        h2 = lmm.h2
        #lmm.sigma_K[i] = x[i]
        #lmm.sigma_K = lmm.sigma_K
        lmm.sigma_K = x
        posterior = lmm.posterior_h2()
        
        #print lmm.sigma_K
        res = np.zeros_like(np.array(lmm.derivative_nLL()))
        for ii in xrange(len(posterior[1])):
            lmm.h2 = posterior[1][ii]
            res += np.array(lmm.derivative_nLL()) * posterior[3][ii,0] / posterior[1].shape[0]
        #res2 = lmm._logdet()[1]
        #print res-res2
        lmm.sigma_K = sigma_K
        lmm.h2 = h2
        if i is None:
            return res
        else:
            return res[i]        
    if 0:
        x0 = np.array([0.3,-0.5,0.8])
        check_logdet = mcheck_grad(func=logdet, grad=grad_logdet, x0=x0, allerrs=True)
        check_logdetXKX = mcheck_grad(func=logdetXKX, grad=grad_logdetXKX, x0=x0, allerrs=True)
        check_YKY = mcheck_grad(func=YKY, grad=dYKY, x0=x0, allerrs=True)
        check_XKY = mcheck_grad(func=XKY, grad=dXKY, x0=x0, allerrs=True)
        check_beta = mcheck_grad(func=beta, grad=dbeta, x0=x0, allerrs=True)
        check_YKXXKXXKY = mcheck_grad(func=YKXXKXXKY, grad=dYKXXKXXKY, x0=x0, allerrs=True)
        check_r2 = mcheck_grad(func=r2, grad=d_r2, x0=x0, allerrs=True)
        check_nLL = mcheck_grad(func=nLL, grad=d_nLL, x0=x0, allerrs=True)
        check_avg_nLL = mcheck_grad(func=avg_nLL, grad=d_avg_nLL, x0=x0, allerrs=True)


    opts = {
    'gradcheck' : True,
    'max_iter_opt': 200,
    }

    ####################
    # conditional version
    ####################
    #lmm.h2 = 0.0001
    #opth2 = lmm.find_h2()

    optimum = opt.opt_hyper(gpr = lmm, opts=opts)
    opth2_ = lmm.find_h2()
    print lmm.h2
    print opth2_
    lmm.h2 = opth2_[0]
    optimum_ = opt.opt_hyper(gpr = lmm, opts=opts)

    opth2__ = lmm.find_h2()

    optimum__ = opt.opt_hyper(gpr = lmm, opts=opts)

    #lmm.h2 = 0.5
    optimum___ = opt.opt_hyper(gpr = lmm, opts=opts)

    from sklearn.metrics import r2_score

    Y_pred_test_OLS = lmm.predict(X=X_test, K=K_star, GLS=False, marinalize_h2=False)
    residual = Y_test - Y_pred_test_OLS
    MSE_test = (residual * residual).mean()
    MAE_test = np.absolute(residual).mean()
    R_test_OLS = np.corrcoef(Y_pred_test_OLS[:,0],Y_test[:,0])[1,0]
    R2_test_OLS = r2_score(Y_test,Y_pred_test_OLS)

    Y_pred_train_OLS = lmm.predict(X=X_train, K=K_train, GLS=False, marinalize_h2=False)
    residual = Y_train - Y_pred_train_OLS
    MSE_train = (residual * residual).mean()
    MAE_train = np.absolute(residual).mean()
    R_train_OLS = np.corrcoef(Y_pred_train_OLS[:,0],Y_train[:,0])[1,0]
    R2_train_OLS = r2_score(Y_train,Y_pred_train_OLS)

    print "train MSE OLS: %.4f, test MSE OLS: %.4f" % (MSE_train, MSE_test)
    print "train MAE OLS: %.4f, test MAE OLS: %.4f" % (MAE_train, MAE_test)
    print "train R   OLS: %.4f, test R   OLS: %.4f" % (R_train_OLS, R_test_OLS)
    print "train R2  OLS: %.4f, test R2  OLS: %.4f" % (R2_train_OLS, R2_test_OLS)
    
    Y_pred_test_GLS = lmm.predict(X=X_test, K=K_star, GLS=True, marinalize_h2=False)
    residual = Y_test - Y_pred_test_GLS
    MSE_test = (residual * residual).mean()
    MAE_test = np.absolute(residual).mean()
    R_test_GLS = np.corrcoef(Y_pred_test_GLS[:,0],Y_test[:,0])[1,0]
    R2_test_GLS = r2_score(Y_test,Y_pred_test_GLS)

    Y_pred_train_GLS = lmm.predict(X=X_train, K=K_train, GLS=True, marinalize_h2=False)
    residual = Y_train - Y_pred_train_GLS
    MSE_train = (residual * residual).mean()
    MAE_train = np.absolute(residual).mean()
    R_train_GLS = np.corrcoef(Y_pred_train_GLS[:,0],Y_train[:,0])[1,0]
    R2_train_GLS = r2_score(Y_train,Y_pred_train_GLS)

    print "train MSE GLS: %.4f, test MSE GLS: %.4f" % (MSE_train, MSE_test)
    print "train MAE GLS: %.4f, test MAE GLS: %.4f" % (MAE_train, MAE_test)
    print "train R   GLS: %.4f, test R   GLS: %.4f" % (R_train_GLS, R_test_GLS)
    print "train R2  GLS: %.4f, test R2  GLS: %.4f" % (R2_train_GLS, R2_test_GLS)

    ####################
    # marginal version
    ####################
    if 1:
        optimum_marginal = opt.opt_hyper_marginal(gpr = lmm, opts=opts)
        opth2_maringal = lmm.find_h2()

        Y_pred_test_OLS_avg = lmm.predict_avg(Xstar=X_test, Kstar=K_star, GLS=False)
        residual = Y_test - Y_pred_test_OLS_avg
        MSE_test = (residual * residual).mean()
        MAE_test = np.absolute(residual).mean()
        R_test_OLS = np.corrcoef(Y_pred_test_OLS_avg[:,0],Y_test[:,0])[1,0]
        R2_test_OLS = r2_score(Y_test,Y_pred_test_OLS_avg)

        Y_pred_train_OLS_avg = lmm.predict_avg(Xstar=X_train, Kstar=K_train, GLS=False)
        residual = Y_train - Y_pred_train_OLS_avg
        MSE_train = (residual * residual).mean()
        MAE_train = np.absolute(residual).mean()
        R_train_OLS = np.corrcoef(Y_pred_train_OLS_avg[:,0],Y_train[:,0])[1,0]
        R2_train_OLS = r2_score(Y_train,Y_pred_train_OLS_avg)

        print "train MSE OLS_avg: %.4f, test MSE OLS_avg: %.4f" % (MSE_train, MSE_test)
        print "train MAE OLS_avg: %.4f, test MAE OLS_avg: %.4f" % (MAE_train, MAE_test)
        print "train R   OLS_avg: %.4f, test R   OLS_avg: %.4f" % (R_train_OLS, R_test_OLS)
        print "train R2  OLS_avg: %.4f, test R2  OLS_avg: %.4f" % (R2_train_OLS, R2_test_OLS)

        Y_pred_test_GLS_avg = lmm.predict_avg(Xstar=X_test, Kstar=K_star, GLS=True)
        residual = Y_test - Y_pred_test_GLS_avg
        MSE_test = (residual * residual).mean()
        MAE_test = np.absolute(residual).mean()
        R_test_GLS = np.corrcoef(Y_pred_test_GLS_avg[:,0],Y_test[:,0])[1,0]
        R2_test_GLS = r2_score(Y_test,Y_pred_test_GLS_avg)

        Y_pred_train_GLS_avg = lmm.predict_avg(Xstar=X_train, Kstar=K_train, GLS=True)
        residual = Y_train - Y_pred_train_GLS_avg
        MSE_train = (residual * residual).mean()
        MAE_train = np.absolute(residual).mean()
        R_train_GLS = np.corrcoef(Y_pred_train_GLS_avg[:,0],Y_train[:,0])[1,0]
        R2_train_GLS = r2_score(Y_train,Y_pred_train_GLS_avg)

        print "train MSE GLS_avg: %.4f, test MSE GLS_avg: %.4f" % (MSE_train, MSE_test)
        print "train MAE GLS_avg: %.4f, test MAE GLS_avg: %.4f" % (MAE_train, MAE_test)
        print "train R   GLS_avg: %.4f, test R   GLS_avg: %.4f" % (R_train_GLS, R_test_GLS)
        print "train R2  GLS_avg: %.4f, test R2  GLS_avg: %.4f" % (R2_train_GLS, R2_test_GLS)

    if 0:
        import cPickle
        testdata = {
            'X_train':X_train,
            'X_test':X_test,
            'Y_train':Y_train,
            'Y_test':Y_test,
            'K_train':K_train,
            'K_test':K_test,
            'K_star':K_star,
            'sigma_K':lmm.sigma_K,
            'h2':lmm.h2,
            'R_train':R_train,
            'R_test':R_test,
            'result_evalgrid_h2':result_evalgrid_h2,
            'result_evalgrid_simplex':result_evalgrid_simplex,
            'Y_pred_train':Y_pred_train,
            'Y_pred_test':Y_pred_test,
        }
        datafile = open("variance_components_data.pickle", "wb")
        cPickle.dump(testdata, datafile)
        datafile.close()


