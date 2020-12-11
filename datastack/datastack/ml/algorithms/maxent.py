import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
import scipy.stats as st
import scipy.special as ss
import scipy as sp
from fastlmm.util.mingrid import *
from fastlmm.inference.lmm_cov import Linreg, computeAKA, computeAKB
from cached import Cached, cached
import collections
import sys
from sklearn.metrics import r2_score
from plot_dirichlet import bc2xy
import pylab as plt
from check_grad import mcheck_grad
import logging as LG
import variance_components_opt as opt
import time
import scipy.optimize as OPT
#import pyximport; pyximport.install() 
import datastack.ml.algorithms.maxent_grad as maxent_grad

class MaxEnt(Cached):
	def __init__(self, Y, X=None, alpha = 0.0):
		"""
		Args:
			Y:	N x C matrix of the multinomial training probabilities adding up to one per row
				(typically binary)
			X: 	N x C x D tensor of D input features per sample n and class c
		"""

		Cached.__init__(self)
		self.Y = Y
		self.X = X
		self._beta = np.zeros(X.shape[2])
		self.alpha = alpha

	@property
	def beta(self):
		return self._beta

	@beta.setter
	def beta(self, value):
		if (value != self._beta).all():
			self.clear_cache('beta')
		self._beta = value

	@property
	def X(self):
		return self._X
	@X.setter
	def X(self, value):
		self.clear_cache('X')
		self._X = value

	@property
	def Y(self):
		return self._Y
	@Y.setter
	def Y(self, value):
		self.clear_cache('Y')
		self._Y = value

	@property
	def alpha(self):
		return self._alpha
	@alpha.setter
	def alpha(self, value):
		self.clear_cache('alpha')
		self._alpha = value

	def exp_Xb(self,X):
		Xb = self.Xb(X=X)
		exp_Xb = np.exp(Xb)
		return exp_Xb
	
	#@profile
	def grad_exp_Xb(self,X):
		Xb = self.Xb(X=X)
		exp_Xb = np.exp(Xb)[:,:,np.newaxis] * X
		return exp_Xb

	#@profile
	def Xb(self,X):
		#Is it faster to separate out query and target here, instead of keeping around the big tensor X?
		Xb = np.tensordot(X, self.beta,axes = ([2],[0]))
		#t0 = time.time(); Xb = np.tensordot(X, self.beta,axes = ([2],[0])); print time.time()-t0
		#t0 = time.time(); Xb = np.dot(X, self.beta); print time.time()-t0
		return Xb

	#@profile
	def grad_Xb(self,X):
		return X
	
	#@profile
	def predict_probability(self,X):
		"""
		Args:
			X: N x C x D matrix
		"""
		exp_Xb = self.exp_Xb(X=X)
		probabilities = exp_Xb / (exp_Xb.sum(1)[:,np.newaxis])
		return probabilities

	@cached(['beta','X'])
	def _predict_probability(self):
		"""
		Args:
			X: N x C x D matrix
		"""
		exp_Xb = self.exp_Xb(X=self.X)
		probabilities = exp_Xb / (exp_Xb.sum(1)[:,np.newaxis])
		return probabilities


	#@profile
	def predict(self,X):
		"""
		Args:
			X: N x C x D matrix
		"""
		probabilities = self.predict_probability(X=X)
		predictions = np.argmax(probabilities,axis=1)
		return predictions

	#@profile
	def grad_predictions(self):
		predictions = self._predict_probability()
		X__ = (predictions[:,:,np.newaxis] * self.X)
		return predictions[:,:,np.newaxis] * (self.X - X__.sum(1)[:,np.newaxis,:])

	#@profile
	@cached(['beta','X','Y','alpha'])
	def loss(self):
		log_p_Y = (-np.log(self._predict_probability()) * self.Y).sum() + (self.alpha*self.beta*self.beta).sum()
		return log_p_Y

	#@profile
	@cached(['beta','X','Y','alpha'])
	def grad_loss(self):
		if 0:	#original slower version
			gp = self.grad_predictions()
			probs = self._predict_probability()		
			g = self.Y[:,:,np.newaxis] * gp / probs[:,:,np.newaxis]
			gradient_loss = -g.sum(0).sum(0)
			gradient_penalty = 2.0 * self.alpha * self.beta
			return gradient_loss + gradient_penalty
		else:	#2/3 of the runtime
			probs = self._predict_probability()
			#gradient_loss = aggregate(probs, self.X, self.Y)	#the slow version
			gradient_loss = maxent_grad.aggregate(probs, self.X, self.Y)
			gradient_penalty = 2.0 * self.alpha * self.beta
			return gradient_loss + gradient_penalty

	def fit(self, X, y, opts = None):
		self.X = X
		self.Y = y
		if opts is None:
			opts = {
			'gradcheck' : False,
			'max_iter_opt': 200,
			}
		optimum_marginal = opt_maxent(maxent=self,opts=opts)
		return self
		
#@profile
def aggregate(probs, X, Y):
	#the slow version of the code implemented in grad_mmaxent.pyx (Cython)
	X__ = (probs[:,:,np.newaxis] * X).sum(1)
	gp = X -  X__[:,np.newaxis,:]
	g = Y[:,:,np.newaxis] * gp
	gradient_loss = -g.sum(0).sum(0)
	return gradient_loss

#(self.Y / probs)[:,:,np.newaxis] * (probs[:,:,np.newaxis] * self.X) - (self.Y / probs)[:,:,np.newaxis] * probs[:,:,np.newaxis] * X__.sum(1)[:,np.newaxis,:]

def opt_maxent(maxent,bounds=None,opts={},*args,**kw_args):
    """
    optimize hyperparams

    Input:
    maxent: MaxEnt instance
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
        maxent.beta = x
        rv = maxent.loss()
        if np.isnan(rv):
            raise Exception("the loss function is nan.")
        return rv

    def df(x):
        maxent.beta = x
        rv = maxent.grad_loss()
        if np.isnan(rv.any()):
            raise Exception("the gradient is nan.")
        return rv

    LG.info('Starting optimization ...')
    t = time.time()

    x = maxent.beta.copy()
    RVopt = OPT.fmin_tnc(f,x,fprime=df,messages=True,maxfun=int(max_iter),pgtol=pgtol,bounds=bounds)
    LG.info('%s'%OPT.tnc.RCSTRINGS[RVopt[2]])
    LG.info('Optimization is converged at iteration %d'%RVopt[1])
    LG.info('Total time: %.2fs'%(time.time()-t))

    xopt = RVopt[0]
    maxent.beta = xopt
    loss_opt = maxent.loss()

    if gradcheck:
        err = OPT.check_grad(f,df,xopt)
        LG.info("check_grad (post): %.2f"%err)


    return [xopt,loss_opt]

def check_all_gradients():
	print "checking all gradients"
	model = MaxEnt(Y=Y_train, X=X_train, alpha=0.3)
	model.beta = np.random.normal(size=X_train.shape[2])
	a = np.random.randn(N,1)
	b = np.random.randn(1,N_target)


	def xb(x,i=None):
		beta = model.beta.copy()
		model.beta = x
		res = (a*b*model.Xb(model.X)).sum()
		model.beta = beta
		return res

	def grad_xb(x,i=None):
		beta = model.beta.copy()
		model.beta = x
		pps = a[:,:,np.newaxis] * b[:,:,np.newaxis] * model.grad_Xb(model.X)
		res = pps.sum(0).sum(0)
		model.beta = beta
		if i is None:
			return res
		else:
			return res[i]

	x0 = np.random.normal(size=X_train.shape[2])
	check_logdet = mcheck_grad(func=xb, grad=grad_xb, x0=x0, allerrs=True)

	def exb(x,i=None):
		beta = model.beta.copy()
		model.beta = x
		res = model.exp_Xb(model.X).sum()#a.dot(model.exp_Xb(model.X)).dot(b).sum()
		model.beta = beta
		return res

	def grad_exb(x,i=None):
		beta = model.beta.copy()
		model.beta = x
		pps = model.grad_exp_Xb(model.X)
		res = pps.sum(0).sum(0)#((pps * a[:,:,np.newaxis]).sum(1) * b).sum(0)
		model.beta = beta
		if i is None:
			return res
		else:
			return res[i]

	x0 = np.random.normal(size=X_train.shape[2])
	check_logdet = mcheck_grad(func=exb, grad=grad_exb, x0=x0, allerrs=True)


	def predictions(x,i=None):
		beta = model.beta.copy()
		model.beta = x
		pred = model.predict_probability(model.X)
		#print "pred:"
		#print pred 
		res = (a * b * pred).sum()
		model.beta = beta
		return res

	def grad_predictions(x,i=None):
		beta = model.beta.copy()
		model.beta = x
		pps = model.grad_predictions()
		#print "grad : "
		#print pps
		res = (pps *b[:,:,np.newaxis] * a[:,:,np.newaxis]).sum(0).sum(0)
		model.beta = beta
		if i is None:
			return res
		else:
			return res[i]

	x0 = np.random.normal(size=X_train.shape[2])
	check_logdet = mcheck_grad(func=predictions, grad=grad_predictions, x0=x0, allerrs=True)

	def nLL(x,i=None):
		beta = model.beta.copy()
		model.beta = x
		res = model.loss()
		model.beta = beta
		return res

	def grad_nLL(x,i=None):
		beta = model.beta.copy()
		model.beta = x
		res = model.grad_loss()
		model.beta = beta
		if i is None:
			return res
		else:
			return res[i]

	x0 = np.random.normal(size=X_train.shape[2])
	check_logdet = mcheck_grad(func=nLL, grad=grad_nLL, x0=x0, allerrs=True)

#@profile
def mainf():
	N = 100
	N_target = N
	D = 700
	N_test = 10

	print "generating the data"
	#each column in the reconstruction has a different signal to noise ratio.
	#the algorithm should weigh up the columns with higher signal to noise
	signal_to_noise = np.random.uniform(size = (D))

	#the true vector of inputs:
	X_train_original = np.random.randn(N,D)
	noise_train = np.random.randn(N,D)
	
	#the noisy reconstruction:
	X_train_noisy = signal_to_noise[np.newaxis,:] * X_train_original + (1-signal_to_noise[np.newaxis,:]) * noise_train
	
	#measure similarity between original and reconstruction:
	X_train_diff = X_train_noisy[:,np.newaxis,:] - X_train_original[np.newaxis,:,:]
	X_train_prod = X_train_noisy[:,np.newaxis,:] * X_train_original[np.newaxis,:,:]

	X_train = X_train_prod#np.concatenate((X_train_prod,X_train_diff),2)
	#As each sample is compared with each sample, the indicator for the true one is the identity:
	Y_train = np.eye(N)

	if N_target!=N:
		#X_train[0,0] = np.nan
		X_train = X_train[0:N][:,0:N_target]
		Y_train = np.zeros((N,N_target))
		Y_train[:,0] = 1.0
	#the true vector of inputs:
	X_test_original = np.random.randn(N_test,D)
	noise_test = np.random.randn(N_test,D)
	
	#the noisy reconstruction:
	X_test_noisy = signal_to_noise[np.newaxis,:] * X_test_original + (1-signal_to_noise[np.newaxis,:]) * noise_test
	
	#measure similarity between original and reconstruction:
	X_test_diff = X_test_noisy[:,np.newaxis,:] - X_test_original[np.newaxis,:,:]
	X_test_prod = X_test_noisy[:,np.newaxis,:] * X_test_original[np.newaxis,:,:]

	X_test = X_test_prod#np.concatenate((X_test_prod,X_test_diff),2)
	#As each sample is compared with each sample, the indicator for the true one is the identity:
	Y_test = np.eye(N_test)

	if "checkgrad" in sys.argv:
		check_all_gradients()


	print "setting up the model"
	#ipdb.set_trace()
	model = MaxEnt(Y=Y_train, X=np.array(X_train,order="C"), alpha=0.3)
	model.beta = np.random.normal(size=X_train.shape[2])	
	
	print "find optimal weights:"
	print "The regularization parameter alpha = %.4f" % model.alpha

	opts = {
	'gradcheck' : True,
	'max_iter_opt': 200,
	}

	#optimum_marginal = opt_maxent(maxent=model,opts=opts)
	model.fit(X=model.X,y=model.Y)
	print "The optimal weights are:"
	print model.beta

	pred_train = model.predict_probability(X=X_train)
	pred_test = model.predict_probability(X=X_test)

	print "predicted training probabilities for the true class in 1 out of %i selection:" % N
	print pred_train.diagonal()[:np.min([N,5])]

	logloss_train = -(Y_train * np.log(pred_train)).sum()
	print "logloss_train = %.4f" % logloss_train
	logloss_train_naive = -(Y_train * np.log(np.ones_like(Y_train)/Y_train.shape[0])).sum()
	print "logloss_train_naive = %.4f" % logloss_train_naive

	print "predicted test probabilities for the true class in 1 out of %i selection:" % N_test
	print pred_test.diagonal()[:np.min([N_test,5])]
	logloss_test = -(Y_test * np.log(pred_test)).sum()
	print "logloss_test = %.4f" % logloss_test
	logloss_test_naive = -(Y_test * np.log(np.ones_like(Y_test)/Y_test.shape[0])).sum()
	print "logloss_test_naive = %.4f" % logloss_test_naive

if __name__ == '__main__':
	mainf()

