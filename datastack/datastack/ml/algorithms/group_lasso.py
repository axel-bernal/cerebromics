import numpy as np
import scipy as sp
from check_grad import mcheck_grad
import scipy.optimize as OPT
from cached import Cached, cached
import logging as LG
import time
import sklearn.metrics

class DirtyGroupLasso(object):
    def __init__(self, add_bias=True, alpha=1.0, beta=None, gamma=None, opts={'max_iter_opt': 1000, 'pgtol': 1e-3}):
        self.W = None
        self.X = None
        self.Y = None
        self.bias = None
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.add_bias = add_bias
        self.opts = opts
        pass

    @property
    def l22_W(self):
        W = self.W
        return (W * W).sum(1)

    @property
    def l21_W(self):
        return np.sqrt(self.l22_W)

    @staticmethod
    def _dirty_group_L1(X, slope):
        '''
        The dirty L1L2 function evaluated at X.

        If slope is None: returns squared L2 norm |X|_2^2
        If slope is not None: returns a strict lower bound to slope*|X|_2^2,
                                close to zero the function looks like  slope*|X|_2^2 (same gradient w.r.t. X at X=[0])
                                which in the tails looks like 
                                \sum_i |X[i,:]|_2 (the group lasso L2L1 norm)

        Args:
            X:      2darray at which to evaluate the function
            slope:  the slope of the function
        Returns:
            function value (scalar)
        '''
        if slope is None:
            return 0.5 * X.ravel().dot(X.ravel())
        else:
            x = np.sqrt((X * X).sum(1))
            return( np.log(np.exp(-2.0 * slope * x) + 1.0) + slope * x + np.log(0.5)).sum()
            #duplicate implementation: should yield the same result:
            return np.log( (np.exp(slope * x) + np.exp(- slope * x)) / 2.0).sum()


    @staticmethod
    def _gradient_dirty_group_L1(X, slope):
        '''
        The matrix derivative of the dirty L1L2 function w.r.t. X.

        If slope is None: returns gradient of squared L2 norm |X|_2^2
        If slope is not None: returns X[i,:] * tanh(slope * slope*|X[i,:]|_2) for all i in [0:X.shape[0]] (tanh = 2*logistic - 1)
                                the gradient of a strict lower bound to slope*|X|_2^2,
                                close to zero the function looks like  slope*|X|_2^2 (same gradient w.r.t. X at X=[0])
                                which in the tails looks like 
                                \sum_i |X[i,:]|_2 (the group lasso L2L1 norm)

        Args:
            X:      2darray at which to evaluate the function
            slope:  the slope of the function
        Returns:
            matrix derivative w.r.t. X (2darray of same dimensionality as X)
        '''
        if slope is None:
            return X
        else:
            x = np.sqrt((X * X).sum(1))
            exponent = np.exp(-2.0 * slope * x)
            tanh = slope * (1.0 - exponent) / (1.0 + exponent) / x
            return tanh[:,np.newaxis] * X

    def regularizer(self):
        '''
        The model regularizer.

        If self.beta is None: standard ridge penalty
        If self.beta is not None: smooth dirty group penalty
        '''
        return DirtyGroupLasso._dirty_group_L1(X=self.W, slope=self.beta)

    def gradient_regularizer(self):
        '''
        The gradient of the model regularizer w.r.t. self.W (and self.bias).

        If self.beta is None: standard ridge penalty
        If self.beta is not None: smooth dirty group penalty
        '''
        return DirtyGroupLasso._gradient_dirty_group_L1(X=self.W, slope=self.beta)

    def residual(self):
        '''
        The matrix of model residual (Y - X*W)
        '''
        residual = self.Y - self.X.dot(self.W)
        if self.add_bias:
            residual -= self.bias
        return residual

    def loss(self):
        '''
        returns the model loss on the residuals
        '''
        residual = self.residual()
        return DirtyGroupLasso._dirty_group_L1(X=residual, slope=self.gamma)

    def gradient_loss(self):
        '''
        returns the gradient of the model loss w.r.t. self.W
        '''
        residual = self.residual()
        inner = DirtyGroupLasso._gradient_dirty_group_L1(X=residual, slope=self.gamma)
        if self.add_bias:
            gradient = np.empty((self.W.shape[0]+1, self.W.shape[1]))
            gradient[0] = -inner.sum(0)
            gradient[1:] = -self.X.T.dot(inner)
        else:
            gradient = -self.X.T.dot(inner)
        return gradient

    def objective(self, x, i=None):
        '''
        returns loss(Y-XW) + alpha * regularizer(W)
        '''
        self.set_parameters(x=x)
        return self.loss() + self.alpha * self.regularizer()

    def set_parameters(self, x):
        """
        set the weights and the bias of the model

        Args:
            1d array of model parameters
        """
        params = x.reshape((-1, self.Y.shape[1]))
        if self.add_bias:
            self.bias = params[0:1]
            self.W = params[1:]
        else:
            self.W = params

    def get_parameters(self):
        """
        returns an ndarray of the concatenated model parameters self.bias and self.W
        """
        if self.add_bias:
            params = np.concatenate((self.bias, self.W), 0)
        else:
            params = self.W
        return params

    def gradient(self, x, i=None):
        """
        compute the overall gradient of self.objective w.r.t. the model parameters and returns in vectorized form.
        """
        self.set_parameters(x=x)
        gradient = self.gradient_loss()
        if self.add_bias:
            gradient[1:] += self.alpha * self.gradient_regularizer()
        else:
            gradient += self.alpha * self.gradient_regularizer()
        if i is not None:
            return gradient.ravel()[i]
        return gradient.ravel()

    def init_params(self, scale=1e-4):
        """
        initialize the model parameters
        The bias is set to the mean of self.Y.
        The W vector is set to a zero mean random normal with standard deviation = scale

        Args:
            scale:  standard deviation of the weight initialization
        """
        if self.add_bias:
            self.bias = self.Y.mean(0)[np.newaxis,:]
        self.W = np.random.normal(scale=scale, size=(self.X.shape[1],self.Y.shape[1]))

    def fit(self, X, y, hotstart=False):
        """
        set X and Y and optimize the model

        Args:
            X:          2darray of training inputs
            y:          2darray of training outputs
            hotstart:   start the optimization at self.W? (default: False)
                        Yields faster convergence when running with multiple related parameter settings
        """
        self.X = X
        self.Y = y
        if not hotstart:
            self.init_params()
        xopt, objective =opt_hyper(self, opts=self.opts)
        return self

    def predict(self, X):
        """
        predict the target values (\hat{Y} = X.dot(W) + bias) for the inputs X 

        Args:
            X:  2darray of input features
        """
        prediction = X.dot(self.W)
        if self.add_bias:
            prediction += self.bias
        return prediction

    def set_params(self, **params):
          if 'alpha' in params:
            self.alpha = params['alpha']
          if 'beta' in params:
            self.beta = params['beta']
          if 'gamma' in params:
            self.gamma = params['gamma']
          return self

    def get_params(self, deep=False):
         """
         We need all parameters required to clone this instance.
         """
         return {"alpha": self.alpha, "beta": self.beta, "gamma": self.gamma, "W": self.W, "bias": self.bias, "add_bias": self.add_bias, "opts": self.opts}

    def score(self, X, y, sample_weight=None, multioutput=None):
        """
        score the model using the default r2_score
        """
        y_pred = self.predict(X=X)
        score = sklearn.metrics.r2_score(y, y_pred, sample_weight=sample_weight, multioutput=multioutput)
        return score


def opt_hyper(model, bounds=None, opts={}, *args, **kw_args):
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
        max_iter = 1000
    if 'pgtol' in opts:
        pgtol = opts['pgtol']
    else:
        pgtol = 1e-4

    LG.info('Starting optimization ...')
    t = time.time()

    x = model.get_parameters().ravel()
    RVopt = OPT.fmin_tnc(model.objective, x, fprime=model.gradient, messages=True, maxfun=int(max_iter), pgtol=pgtol, bounds=bounds)
    LG.info('%s'%OPT.tnc.RCSTRINGS[RVopt[2]])
    LG.info('Optimization is converged at iteration %d'%RVopt[1])
    LG.info('Total time: %.2fs'%(time.time()-t))

    xopt = RVopt[0]
    model.set_parameters(xopt)
    objective = model.objective(xopt)

    if gradcheck:
        err = OPT.check_grad(model.objective,model.gradient,xopt)
        LG.info("check_grad (post): %.2f"%err)

    return [xopt,objective]


if __name__ == '__main__':
    N = 1000
    P = 2
    D = 3
    threshold_grad_check = 50

    lasso = DirtyGroupLasso()
    lasso.X = np.random.normal(size=(N,D))
    lasso.Y = np.random.normal(size=(N,P))
    lasso.Y[0,:] += 500.0 * np.random.normal(size=(P))
    lasso.gamma = 1.0
    W = np.random.normal(size=(D+1,P))

    print "L2 loss and L2 regularizer"
    lasso.set_parameters(W.copy().flatten())
    lasso.gamma = None
    lasso.beta = None
    if P*D <= threshold_grad_check:
        mcheck_grad(func=lasso.objective, grad=lasso.gradient, x0=W.flatten(), allerrs=True)
    lasso.fit(lasso.X, lasso.Y)
    print np.absolute(lasso.residual()[0,:]).sum()

    print "L2 loss and group regularizer"
    lasso.set_parameters(W.copy().flatten())
    lasso.gamma = None
    lasso.beta = 1.0
    if P*D <= threshold_grad_check:
        mcheck_grad(func=lasso.objective, grad=lasso.gradient, x0=W.flatten(), allerrs=True)
    lasso.fit(lasso.X, lasso.Y)
    print np.absolute(lasso.residual()[0,:]).sum()

    print "group loss and L2 regularizer"
    lasso.set_parameters(W.copy().flatten())
    lasso.gamma = 1.0
    lasso.beta = None
    if P*D <= threshold_grad_check:
        mcheck_grad(func=lasso.objective, grad=lasso.gradient, x0=W.flatten(), allerrs=True)
    lasso.fit(lasso.X, lasso.Y)
    print np.absolute(lasso.residual()[0,:]).sum()

    print "group loss and group regularizer"
    lasso.set_parameters(W.copy().flatten())
    lasso.gamma = 1.0
    lasso.beta = 1.0
    if P*D <= threshold_grad_check:
        mcheck_grad(func=lasso.objective, grad=lasso.gradient, x0=W.flatten(), allerrs=True)
    lasso.fit(lasso.X, lasso.Y)
    print np.absolute(lasso.residual()[0,:]).sum()

    if 1:
        lasso_ = DirtyGroupLasso(beta=1.0, alpha=5.0)
        lasso_.X = np.random.normal(size=(N,1))
        lasso_.Y = np.random.normal(size=(N,1))
        w = np.arange(-10,10,0.01)
        y1 = np.zeros_like(w)
        y2 = np.zeros_like(w)
        y1_l2 = np.zeros_like(w)
        y2_l2 = np.zeros_like(w)
        for i in xrange(len(w)):
            x = w[i]
            lasso_.W = np.ones((1,1)) * x
            y1[i] = lasso_.regularizer()
            y2[i] = lasso_.gradient_regularizer()
        lasso_.beta = None
        for i in xrange(len(w)):
            x = w[i]
            lasso_.W = np.ones((1,1)) * x
            y1_l2[i] = lasso_.regularizer()
            y2_l2[i] = lasso_.gradient_regularizer()


        import pylab as plt
        plt.ion()
        plt.figure()
        plt.subplot(121)
        plt.plot(w,y1)
        plt.plot(w,y1_l2)
        plt.grid()
        plt.ylabel("regularizer")
        plt.subplot(122)
        plt.plot(w,y2)
        plt.plot(w,y2_l2)
        plt.grid()
        plt.ylabel("gradient")

        plt.legend(["group", "L2"])



