#from variance_components import LMM
import scipy.optimize as OPT
import numpy as np
import time
import ipdb
import logging as LG

def param_list_to_dict(list,param_struct,skeys):
    """convert from param dictionary to list
    param_struct: structure of parameter array
    """
    RV = []
    i0= 0
    for key in skeys:
        val = param_struct[key]
        shape = np.array(val)
        np = shape.prod()
        i1 = i0+np
        params = list[i0:i1].reshape(shape)
        RV.append((key,params))
        i0 = i1
    return dict(RV)

def checkgrad(f, fprime, x, *args,**kw_args):
    """
    Analytical gradient calculation using a 3-point method

    """
    LG.debug("Checking gradient ...")
    import numpy as np

    # using machine precision to choose h
    eps = np.finfo(float).eps
    step = np.sqrt(eps)*(x.min())

    # shake things up a bit by taking random steps for each x dimension
    h = step*np.sign(np.random.uniform(-1, 1, x.size))

    f_ph = f(x+h, *args, **kw_args)
    f_mh = f(x-h, *args, **kw_args)
    numerical_gradient = (f_ph - f_mh)/(2*h)
    analytical_gradient = fprime(x, *args, **kw_args)
    ratio = (f_ph - f_mh)/(2*np.dot(h, analytical_gradient))

    h = np.zeros_like(x)
    for i in range(len(x)):
        ipdb.set_trace()
    h[i] = step
    f_ph = f(x+h, *args, **kw_args)
    f_mh = f(x-h, *args, **kw_args)
    numerical_gradient = (f_ph - f_mh)/(2*step)
    analytical_gradient = fprime(x, *args, **kw_args)[i]
    ratio = (f_ph - f_mh)/(2*step*analytical_gradient)
    h[i] = 0
    LG.debug("[%d] numerical: %f, analytical: %f, ratio: %f" % (i, numerical_gradient,analytical_gradient,ratio))


def opt_hyper(gpr,bounds=None,opts={},*args,**kw_args):
    """
    optimize hyperparams

    Input:
    gpr: GP regression class
    hyperparams: dictionary filled with starting hyperparameters
    opts: options for optimizer
    """
    if 'gradcheck' in opts:
        gradcheck = opts['gradcheck']
    else:
        gradcheck = False
    if 'max_iter_opt' in opts:
        max_iter = opts['max_iter_opt']
    else:
        max_iter = 5000
    if 'pgtol' in opts:
        pgtol = opts['pgtol']
    else:
        pgtol = 1e-10

    def f(x):
        sig = gpr.sigma_K.copy()
        sig[:-1] = x
        gpr.sigma_K = sig
        rv = gpr.nLL()[0]
        if np.isnan(rv):
            return 1E6
        return rv

    def df(x):
        sig = gpr.sigma_K.copy()
        sig[:-1] = x
        gpr.sigma_K = sig
        rv = gpr.derivative_nLL()[:-1]
        if (~np.isfinite(rv)).any():
            idx = (~np.isfinite(rv))
            rv[idx] = 1E6
        return rv

    LG.info('Starting optimization ...')
    t = time.time()

    x = gpr.sigma_K.copy()[:-1]
    RVopt = OPT.fmin_tnc(f,x,fprime=df,messages=True,maxfun=int(max_iter),pgtol=pgtol,bounds=bounds)
    LG.info('%s'%OPT.tnc.RCSTRINGS[RVopt[2]])
    LG.info('Optimization is converged at iteration %d'%RVopt[1])
    LG.info('Total time: %.2fs'%(time.time()-t))

    xopt = RVopt[0]
    sig = gpr.sigma_K.copy()
    sig[:-1] = xopt
    gpr.sigma_K = sig
    lml_opt = gpr.nLL()[0]

    if gradcheck:
        err = OPT.check_grad(f,df,xopt)
        LG.info("check_grad (post): %.2f"%err)


    return [xopt,lml_opt]

def opt_hyper_marginal(gpr,bounds=None,threshold_posterior=1.0e-8,opts={},*args,**kw_args):
    """
    optimize hyperparams

    Input:
    gpr: GP regression class
    hyperparams: dictionary filled with starting hyperparameters
    opts: options for optimizer
    """
    if 'gradcheck' in opts:
        gradcheck = opts['gradcheck']
    else:
        gradcheck = False
    if 'max_iter_opt' in opts:
        max_iter = opts['max_iter_opt']
    else:
        max_iter = 5000
    if 'pgtol' in opts:
        pgtol = opts['pgtol']
    else:
        pgtol = 1e-10

    def f(x):
        sig = gpr.sigma_K.copy()
        sig[:-1] = x
        gpr.sigma_K = sig
        rv = -gpr.posterior_h2()[-1]
        if np.isnan(rv):
            return 1E6
        return rv

    def df(x):
        h2_sav = gpr.h2
        sig = gpr.sigma_K.copy()
        sig[:-1] = x
        gpr.sigma_K = sig
        posterior = gpr.posterior_h2()
        rv = np.zeros_like(np.array(gpr.derivative_nLL())[:-1])
        for ii in xrange(len(posterior[1])):
            if posterior[3][ii,0]>threshold_posterior:
                gpr.h2 = posterior[1][ii]
                rv += np.array(gpr.derivative_nLL()[:-1]) * posterior[3][ii,0] / posterior[1].shape[0]

        if (~np.isfinite(rv)).any():
            idx = (~np.isfinite(rv))
            rv[idx] = 1E6
        gpr.h2 = h2_sav
        return rv

    LG.info('Starting optimization ...')
    t = time.time()

    x = gpr.sigma_K.copy()[:-1]
    RVopt = OPT.fmin_tnc(f,x,fprime=df,messages=True,maxfun=int(max_iter),pgtol=pgtol,bounds=bounds)
    LG.info('%s'%OPT.tnc.RCSTRINGS[RVopt[2]])
    LG.info('Optimization is converged at iteration %d'%RVopt[1])
    LG.info('Total time: %.2fs'%(time.time()-t))

    xopt = RVopt[0]
    sig = gpr.sigma_K.copy()
    sig[:-1] = xopt
    gpr.sigma_K = sig
    lml_opt = -gpr.posterior_h2()[-1]

    if gradcheck:
        err = OPT.check_grad(f,df,xopt)
        LG.info("check_grad (post): %.2f"%err)


    return [xopt,lml_opt]