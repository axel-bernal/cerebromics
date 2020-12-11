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
import ipdb
import logging
import sys
from sklearn.metrics import r2_score
from plot_dirichlet import bc2xy

from check_grad import mcheck_grad

import variance_components_opt as opt

def pdf_prod_unif(x,N):
	return (-np.log(x))**(N-1) / np.math.factorial(N-1)

def cdf_prod_2_unif(x):
	return x - x*np.log(x)

def plot_term(points, vals=None):
		import ternary
		scale2 = 1.0
		figure, tax = ternary.figure(scale2=scale2)
		tax.set_title("Scatter Plot", fontsize=20)
		tax.boundary(color="black", linewidth=2.0)
		tax.gridlines(multiple=5, color="blue")

		if vals is None:
			tax.scatter(points, marker='.', color='green', label="Green Diamonds",alpha=0.05)
		tax.ticks(axis='lbr', color="black", linewidth=1, multiple=0.1)
		tax.show()

class LMM(Cached):
	def __init__(self, Y, covariates=None, K=[], rank_threshold=1E-10, sigma_K=None, h2=0.0):
		Cached.__init__(self)
		self._sigma_K = None
		self._dof = None #Multivariate Gaussian is the default
		self.Y = Y
		self.covariates = covariates
		self.K = K
		self.rank_threshold = rank_threshold
		self.sigma_K = sigma_K
		self.scale2 = 1.0
		self.h2 = h2

	@property
	def rank(self):
		return self.N

	@property
	def N(self):
	    return self.Y.shape[0]
	
	@property
	def covariates(self):
		return self._covariates

	@covariates.setter
	def covariates(self, value):
		self.clear_cache("covariates")
		self._covariates = Linreg(X=value)

	@property
	def Y(self):
		return self._Y

	@Y.setter
	def Y(self, value):
		self.clear_cache("Y")
		self._Y = value

	def _AKB(self, UA, UB):
		AKB = computeAKB(self.Sd(), self.denom, UA=UA[0], UB=UB[0], UUA=UA[1], UUB=UB[1])
		return AKB

	@property
	def scale(self):
		return self._scale
	
	@scale.setter
	def scale(self, value):
		self._scale = value

	@property
	def scale2(self):
	    return self.scale*self.scale
	
	@scale2.setter
	def scale2(self, value):
		assert value >= 0.0
		self.scale = np.sqrt(value)

	@property
	def dof(self):
		return self._dof
	
	@dof.setter
	def dof(self, value):
		self._dof = value

	@property
	def dof2(self):
		if self.dof is None:
			return None
		return self.dof*self.dof + 2.0 + self.rank_threshold

	@property
	def h2(self):
		return self.gamma2 / (1.0 + self.gamma2)
	
	@h2.setter
	def h2(self, value):
		assert value >= 0.0
		assert value < 1.0
		self.gamma = np.sqrt( value / (1.0-value) )

	@property
	def gamma2(self):
		return self._gamma * self._gamma

	@property
	def gamma(self):
		return self._gamma
	
	@gamma.setter
	def gamma(self, value):
		self.clear_cache("h2")
		self._gamma = value


	@property
	def N_K(self):
	    return len(self.K)
	

	@property
	def sigma_K(self):
		if (self._sigma_K is None):
			return np.sqrt(np.ones(self.N_K) / self.N_K)
		return self._sigma_K

	@sigma_K.setter
	def sigma_K(self, value):
		if (value is None) or (self._sigma_K is None) or (value != self._sigma_K).any():
			self.clear_cache("K")
			self._sigma_K = value

	@property
	def a2(self):
		sigma2_K = self.sigma_K * self.sigma_K
		a2 = sigma2_K / sigma2_K.sum()
		return a2

	@a2.setter
	def a2(self, value):
		assert (value>=0).all()
		self.sigma_K = np.sqrt(value)
	
	@property
	def Kstar(self):
		return self._Kstar

	@Kstar.setter
	def Kstar(self, value):
		self.clear_cache("Kstar")
		if isinstance(value, np.ndarray):
			self._Kstar = [value]
		elif isinstance(value, list):
			self._Kstar = value
		elif value is None:
			self._Kstar = []
		else:
			raise Exception("unknown type for K: %s " % str(type(value)))
	
	@property
	def K(self):
		return self._K

	@K.setter
	def K(self, value):
		self.clear_cache("K")
		if isinstance(value, np.ndarray):
			self._K = [value]
		elif isinstance(value, list):
			self._K = value
		elif value is None:
			self._K = []
		else:
			raise Exception("unknown type for K: %s " % str(type(value)))

	def K_sum(self):
		if not (len(self.K)>0):
			return None
		K_sum = self.a2[0] * self.K[0]
		for i in xrange(1,len(self.K)):
			K_sum += self.a2[i] * self.K[i]
		return K_sum

	def _Kstar_sum(self):
		K_sum = self.a2[0] * self.Kstar[0]
		for i in xrange(1,len(self.Kstar)):
			K_sum += self.a2[i] * self.Kstar[i]
		return K_sum

	@cached(["K","Kstar"])
	def _Kstar_rot(self):
		assert len(self.Kstar) == len(self.K)
		if not (len(self.Kstar)>0):
			return None
		K_sum = self._Kstar_sum()
		s,U = self.eig()
		K_sum_rot = K_sum.dot(U)
		return (K_sum_rot, None)

	@cached(["K"])
	def eig(self):
		K = self.K_sum()
		if K is None:
			return (np.ones(self.N), np.ones((self.N, 0)))
		s,U = sla.eigh(K, overwrite_a=True)
		if np.any(s < -0.1):
		    logging.warning("kernel contains a negative Eigenvalue")
		return (s, U)

	@cached(["Y","K"])
	def Y_rot(self):
		if len(self.K)==0:
			return self.Y
		return (self.eig()[1].T.dot(self.Y), None)

	@cached(["K","covariates","h2"])
	def _dXKX(self):
		dK_rot = self.dK_rot()
		if not (len(dK_rot)>0):
			return []
		X_rot = self.X_rot()
		dXKX = []
		for i in xrange(len(dK_rot)):
			XKdK = self._AKB(UA=X_rot, UB=(dK_rot[i], None))
			dXKX.append( -self._AKB(UA=(XKdK.T, None), UB=X_rot) * self.h2)
		return dXKX

	@cached(["K","covariates","h2"])
	def dXKXi(self):
		dXKX = self._dXKX()
		if not (len(dXKX)>0):
			return []
		dXKXi = []
		for i in xrange(len(dXKX)):
			res = self._solve_XKX(dXKX[i])
			dXKXi.append( -self._solve_XKX(res.T))
		return dXKXi

	@cached(["K","covariates","h2"])
	def _dXKY(self):
		dK_rot = self.dK_rot()
		if not (len(dK_rot)>0):
			return []
		X_rot = self.X_rot()
		Y_rot = self.Y_rot()
		dXKY = []
		for i in xrange(len(dK_rot)):
			XKdK = self._AKB(UA=X_rot, UB=(dK_rot[i], None))
			dXKY.append( -self._AKB(UA=(XKdK.T, None), UB=Y_rot) * self.h2)
		return dXKY

	def _dK(self):
		dK = []
		sumsquares = (self.sigma_K*self.sigma_K).sum()
		if not (len(self.K)>0):
			return []
		for i in xrange(len(self.K)):
			dK.append( self.scale2 * 2.0 * self.sigma_K[i]/sumsquares * self.K[i])
		for i in xrange(len(self.K)):
			for j in xrange(0,len(self.K)):
				dK[i] -= (self.scale2 * 2.0 * self.sigma_K[j]*self.sigma_K[j]*self.sigma_K[i]) / (sumsquares * sumsquares) * self.K[j]
		return dK

	def _dK_gamma(self):
		"""
		note that this can readily be diagonalized for the rotated version, so this should not be needed.
		"""
		factor = 2.0 * (self.gamma / (self.gamma2+1.0) - self.gamma / ((self.gamma2+1.0)*(self.gamma2+1.0))) 
		res = factor * self.K_sum()
		res.flat[::self.N+1] -= factor
		return res

	def _dK_scale(self):
		"""
		note that this can readily be diagonalized for the rotated version, so this should not be needed.
		"""
		factor = 2.0 * self.scale 
		res = factor * self.h2 * self.K_sum()
		res.flat[::self.N+1] -= factor * (1.0-self.h2)
		return res

	@cached(["K"])
	def dK_rot(self):
		dK = self._dK()
		if not (len(dK)>0):
			return []
		dK_rot = []
		U_K = self.eig()[1]
		for i in xrange(len(dK)):
			dK_rot.append(U_K.T.dot(dK[i]).dot(U_K))
		return dK_rot

	@cached(["K","covariates"])
	def X_rot(self):
		if len(self.K)==0:
			return self.covariates.X
		return (self.eig()[1].T.dot(self.covariates.X), None)

	def Sd(self):
		s = self.eig()[0]
		Sd = (self.h2 * s + (1.0 - self.h2)) * self.scale2
		return Sd

	@property
	def denom(self):
		denom = (1.0 - self.h2) * self.scale2      # determine normalization factor
		return denom


	def _solve_XKX(self, XKY):
		s,U = self._eigXKX()
		res = U.T.dot(XKY)
		res /= s[:,np.newaxis]
		res = U.dot(res)
		return res

	def _logdetXKX(self):
		return np.log(self._eigXKX()[0]).sum()

	@cached(["covariates"])
	def _logdetXX(self):
		XX = self.covariates.X.T.dot(self.covariates.X)
		if XX.shape[0]==0:
			return 0.0
		sign,logdet = la.slogdet(XX)
		assert sign==1.0, "XX not positive definite!"
		return logdet

	@cached(["K","covariates","h2"])
	def _eigXKX(self):
		Sd = self.Sd()
		X_rot = self.X_rot()
		XKX = computeAKB(Sd, self.denom, UA=X_rot[0], UB=X_rot[0], UUA=X_rot[1], UUB=X_rot[1])
		if XKX.shape[0]==0:
			return np.zeros((0)), np.zeros((0,0))
		s,U = sla.eigh(XKX, overwrite_a=True)
		if np.any(s <= -1E-10):
		    logging.warning("XKX contains a negative Eigenvalue")
		nonz = s>=self.rank_threshold
		s = s[nonz]
		U=U[:,nonz]
		return (s, U)

	@cached(["Y","K","covariates","h2"])
	def XKY(self):
		Y_rot = self.Y_rot()
		X_rot = self.X_rot()
		XKY = self._AKB(UA=X_rot, UB=Y_rot)
		return XKY

	@cached(["Y","K","h2"])
	def YKY(self):
		Sd = self.Sd()
		denom = self.denom
		UY = self.Y_rot()
		YKY = computeAKA(Sd=Sd, denom=denom, UA=UY[0], UUA=UY[1])
		return YKY

	@cached(["Y","K","h2"])
	def _derivative_YKY(self):
		"""
		gradient checks!
		"""
		dK_rot = self.dK_rot()
		if not (len(dK_rot)>0):
			return []
		Y_rot = self.Y_rot()
		dYKY = []
		for i in xrange(len(dK_rot)):
			YKdK = self._AKB(UA=Y_rot, UB=(dK_rot[i], None))
			dYKY.append( -self._AKB(UA=(YKdK.T, None), UB=Y_rot).diagonal().sum() * self.h2)
		return dYKY

	@cached(["Y","K","covariates","h2"])
	def beta(self):
		sXKX, UXKX = self._eigXKX()
		XKY = self.XKY()
		beta = self._solve_XKX(XKY)
		return beta

	@cached(["Y","K","covariates","h2"])
	def derivative_beta(self):
		dXKXi = self.dXKXi()
		XKY = self.XKY()
		dXKY = self._dXKY()
		derivative_beta = []
		for i in xrange(len(dXKXi)):
			dXKXXKY = dXKXi[i].dot(XKY)
			derivative_beta.append(dXKXXKY + self._solve_XKX(dXKY[i]))
		return derivative_beta

	@cached(["Y","K","covariates","h2"])
	def _derivative_YKXXKXXKY(self):
		beta = self.beta()
		XKY = self.XKY()
		derivative_beta = self.derivative_beta()
		derivative_XKY = self._dXKY()
		derivative_YKXXKXXKY = []
		for i in xrange(len(derivative_XKY)):
			res = (derivative_XKY[i] * beta).sum()
			derivative_YKXXKXXKY.append(res + (XKY * derivative_beta[i]).sum())
		return derivative_YKXXKXXKY		

	def _logdetK(self):
		Sd = self.Sd()
		logdet = np.log(Sd).sum()
		logdet += (self.N-self.rank) * np.log(self.denom)
		return logdet

	def logdet(self):
		logdet_K = self._logdetK()
		logdet_XX = self._logdetXX()
		logdet_XKX = self._logdetXKX()
		return logdet_K + logdet_XKX - logdet_XX

	def _derivative_log_det(self):
		"""
		checks gradient!
		"""
		Sd = self.Sd()
		dK_rot = self.dK_rot()
		derivative_trace = []
		for i in xrange(len(dK_rot)):
			derivative_trace.append(self.h2 * (dK_rot[i].diagonal() / Sd).sum()) 
		return derivative_trace

	def _derivative_log_det_XKX(self):
		"""
		checks gradient!
		"""
		dXKX = self._dXKX()
		derivative_trace = []
		for i in xrange(len(dXKX)):
			XKXidXKX = self._solve_XKX(dXKX[i])
			derivative_trace.append(np.trace(XKXidXKX))
		return derivative_trace

	def _r2(self):
		var_total = self.YKY()
		var_explained = (self.XKY() * self.beta()).sum(0)
		r2 = var_total.sum(0) - var_explained.sum(0)
		return r2

	def _derivative_r2(self):
		dYKY = self._derivative_YKY()
		derivative_YKXXKXXKY = self._derivative_YKXXKXXKY()
		d_r2 = []
		for i in xrange(len(dYKY)):
			d_r2.append(dYKY[i]-derivative_YKXXKXXKY[i])
		return d_r2

	def nLL(self):
		"""
		dof and scale2 define the inverse Gamma prior on the variance of the prior on the total variance sigma2.
		E[sigma2] = scale2
		V[sigma2] = 2/dof
		large dof means that we have a weak prior on sigma2, meaning that we may accept the MLE
		"""
		if (self.h2<0) or (self.h2>=1.0):
			return (3E20, np.nan)

		N = (self.Y.shape[0] - self.covariates.X.shape[1]) * self.Y.shape[1]

		r2 = self._r2()

		sigma2 = r2/N
		logdetK = self.logdet()
		
		if self.dof2 is None:#Use the Multivariate Gaussian
			nLL = 0.5 * (logdetK + N * (np.log(2.0 * np.pi * sigma2) + 1.0))
		elif 0:#Use multivariate student-t
			nLL = 0.5 * (logdetK + (self.dof2 + N) * np.log(1.0 + r2 / self.dof2))
			nLL +=  0.5 * N * np.log(self.dof2 * np.pi) + ss.gammaln(0.5 * self.dof2) - ss.gammaln(0.5 * (self.dof2 + N))
		else:
			nLL = 0.5 * (logdetK + (self.dof2 + N) * np.log(1.0 + r2 / (self.dof2-2.0)))
			nLL +=  0.5 * N * np.log((self.dof2-2.0) * np.pi) + ss.gammaln(0.5 * self.dof2) - ss.gammaln(0.5 * (self.dof2 + N))
		return (nLL, sigma2)


	@cached(["Y","K","covariates"])
	def posterior_h2(self):
		return self._posterior_h2(nGridH2=500)

	def _posterior_h2(self, nGridH2=100, minH2=0.0, maxH2=0.99999, **kwargs):
		'''
		Find the optimal h2 for a given K. Note that this is the single kernel case. So there is no sigma_K.
		(default maxH2 value is set to a value smaller than 1 to avoid loss of positive definiteness of the final model covariance)
		Args:
			nGridH2 : number of h2-grid points to evaluate the negative log-likelihood at. Number of columns in design matrix for kernel for normalization (default: 10)
			minH2   : minimum value for h2 optimization (default: 0.0)
			maxH2   : maximum value for h2 optimization (default: 0.99999)
			estimate_Bayes: implement me!   (default: False)

		Returns:
			dictionary containing the model parameters at the optimal h2
		'''
		#f = lambda x : (self.nLLeval(h2=x,**kwargs)['nLL'])
		resmin = [None]
		#logging.info("starting H2 search")
		h2_sav = self.h2
		def f(x):
			self.h2 = x
			res = self.nLL(**kwargs)
			#check all results for local minimum:
			if (resmin[0] is None):
					resmin[0] = {'nLL':res[0],'h2':np.zeros_like(res[0])+x, 'sigma2' : res[1]}
			else:
				
				if (res[0] < resmin[0]['nLL']):
					resmin[0]['nLL'] = res[0]
					resmin[0]['h2'] = x
					resmin[0]['sigma2'] = res[1]
			#logging.info("search\t{0}\t{1}".format(x,res['nLL']))
			return res[0]
		(grid,neg_logp) = evalgrid1D(f, evalgrid = None, nGrid=nGridH2, minval=minH2, maxval = maxH2, dimF=self.Y.shape[1])
		
		logp = -neg_logp
		post_h2 = np.exp(logp-logp.max())/np.exp(logp-logp.max()).mean()#*logp.shape[0]
		h2_mean = (np.exp(logp-logp.max())*grid[:,np.newaxis]).sum()/np.exp(logp-logp.max()).sum()
		h2_var = (np.exp(logp-logp.max())*(grid[:,np.newaxis] - h2_mean)**2.0).sum()/np.exp(logp-logp.max()).sum()
		#log sum exp trick
		log_p_marginal = np.log(np.exp(logp-logp.max()).mean()) + logp.max()
		self.h2 = h2_sav
		return resmin[0], grid, logp, post_h2, h2_mean, h2_var, log_p_marginal

	def find_h2(self, nGridH2=10, minH2=0.0, maxH2=0.99999, estimate_Bayes=False, **kwargs):
		'''
		Find the optimal h2 for a given K. Note that this is the single kernel case. So there is no sigma_K.
		(default maxH2 value is set to a value smaller than 1 to avoid loss of positive definiteness of the final model covariance)
		Args:
		    nGridH2 : number of h2-grid points to evaluate the negative log-likelihood at. Number of columns in design matrix for kernel for normalization (default: 10)
		    minH2   : minimum value for h2 optimization (default: 0.0)
		    maxH2   : maximum value for h2 optimization (default: 0.99999)
		    estimate_Bayes: implement me!   (default: False)

		Returns:
		    dictionary containing the model parameters at the optimal h2
		'''
		resmin = [None]
		assert estimate_Bayes == False, "not implemented"
		h2_sav = self.h2

		def f(x,resmin=resmin):
			self.h2 = x
			res = self.nLL(**kwargs)
			if (resmin[0] is None) or (res[0] < resmin[0][1]):
				resmin[0] = (x, res[0], res[1])
			return res[0]   
		(minimum, xmin, evalgrid, resultgrid) = minimize1D(f=f, nGrid=nGridH2, minval=minH2, maxval=maxH2, return_grid=True)
		res = (resmin[0][0], resmin[0][1], resmin[0][2], evalgrid, resultgrid)
		self.h2 = h2_sav
		return res

	def _predict_K(self):
		if not len(self.K)>0:
			return 0.0
		Y_rot = self.Y_rot()
		X_rot = self.X_rot()
		Kstar_rot = self._Kstar_rot()
		Uresidual = Y_rot[0] - X_rot[0].dot(self.beta()) 
		Y_K = computeAKB(self.Sd(), self.denom, UA=Kstar_rot[0].T, UB=Uresidual, UUA=None, UUB=None)
		return Y_K * self.h2	#note that the H2 comes from h2 * Kstar

	def _predict_OLS(self, Xstar):
		"""
		predict a new sample for a single h^2.
		The mean is predicted using the OLS instead of the GLS predictor that has a smaller variance.
		"""		
		#predict OLS part
		self.covariates.set_beta(self.Y)
		Y_covariates = self.covariates.predict(Xstar=Xstar)
		return Y_covariates

	def predict_avg(self, Xstar, Kstar, GLS=True):
		"""
		predict a new sample for a single h^2.
		"""
		self.Kstar = Kstar

		posterior = self.posterior_h2()

		rv = np.zeros((Xstar.shape[0],self.Y.shape[1]))
		for ii in xrange(len(posterior[1])):
			self.h2 = posterior[1][ii]
			rv_ = self.predict(X=Xstar,K=Kstar,GLS=GLS,update_Kstar=False, marinalize_h2=False) * posterior[3][ii,0] / posterior[1].shape[0]
			rv+=rv_
		return rv

	def derivative_nLL(self):
		"""
		take the derivative of the negative log likelihood with respect to a covariance parameter
		"""
		N = (self.Y.shape[0] - self.covariates.X.shape[1]) * self.Y.shape[1]
		
		dlogdet_XKX = self._derivative_log_det_XKX()
		dlogdet_K = self._derivative_log_det()
		d_r2 = self._derivative_r2()
		r2 = self._r2()
		derivative = []
		if self.dof2 is None:
			#Gaussian case
			for i in xrange(len(d_r2)):
				res = 0.5 * (dlogdet_XKX[i] + dlogdet_K[i] + N* d_r2[i]/r2)
				derivative.append(res)
		else:
			#Student t-case
			for i in xrange(len(d_r2)):
				d_nLL = 0.5 * (dlogdet_K[i] + dlogdet_XKX[i] + (self.dof2 + N)/(self.dof2-2.0) * d_r2[i]/(1.0 + r2 / (self.dof2-2.0)))
				derivative.append(d_nLL)
		return derivative

###################
# sklearn interface
###################

	def predict(self, X, K, GLS=True, update_Kstar=True, marinalize_h2=True):
		"""
		predict a new sample for a single h^2.

		Params:
			X:				array-like of test covariates
			K:				list of test kernels to train data (each of shape Ntest x Ntrain)
			GLS:			boolean if to use the generalized least squares predictor for covariates (=best unbiased predictor)
							or the ordinary least squares predictor (unbiased but larger variance) (default True)
			update_Kstar:	if K has been provided before, then update_Kstar can be set to False to enable some caching (default True)
			marginalize_h2:	predict marginalizing over the noise variance h2 or not? (default True)
		"""
		if marinalize_h2:
			return self.predict_avg(Xstar=X, Kstar=K, GLS=GLS)

		if update_Kstar:
			self.Kstar = K
		#predict kernel part
		Y_K = self._predict_K()

		if GLS:
			Y_covariates = X.dot(self.beta())
			"""
			predict a new sample for a single h^2.
			The mean is predicted using the GLS predictor that has a smaller variance than OLS.
			If the likelihood is correct, then this predictor should have smaller variance.
			"""
		else:
			Y_covariates = self._predict_OLS(Xstar=X)
		#the resulting prediction is the sum of the two terms
		Y_pred = Y_covariates + Y_K
		return Y_pred

	def get_params(self):
		"""
		get the parameters of the model.
		"""
		return {
		'a2':		self.a2,
		'h2':		self.h2,
		'scale2':	self.scale2,
		'dof2':		self.dof2,
		}

	def fit(self, X=None, y=None, K=None, marginalize_h2=True):
		"""
		set the training data and fit the model

		Params:
			X:				training covariates
			y:				training target variable
			K:				list of training kernels
			marinalize_h2:	marginalize over h2 when optimizing? (default True) 
		"""
		if X is not None:
			self.covariates = X
		if y is not None:
			self.Y = y[:,np.newaxis]
		if K is not None:
			self.K = K
		if marginalize_h2:
			opts = {
			'gradcheck' : True,
			'max_iter_opt': 200,
			}
			optimum_marginal = opt.opt_hyper_marginal(gpr = self, opts=opts)
		else:
			raise NotImplementedError("only marginal version implemented")
		pass

	def score(self, X, y, K):
		"""
		compute the R2 score function.
		"""
		pred = self.predict(X=X, K=K)
		return r2_score(y, pred[:,0])

	def set_params(self, **params):
		pass

def evalgrid_h2(dim, evalgrid = None, nGrid=10, nGridH2=50, minval=0.00001, maxval = 0.99999, dimF=0, lmm=None):
	'''
	evaluate a function f(x) on all values of a grid.
	--------------------------------------------------------------------------
	Input:
	f(x)    : callable target function
	evalgrid: 1-D array prespecified grid of x-values
	nGrid   : number of x-grid points to evaluate f(x)
	minval  : minimum x-value for optimization of f(x)
	maxval  : maximum x-value for optimization of f(x)
	--------------------------------------------------------------------------
	Output:
	evalgrid    : x-values
	resultgrid  : f(x)-values
	--------------------------------------------------------------------------
	'''
	sigma_K_sav = lmm.sigma_K
	h2_sav = lmm.h2
	def f(x):
		#print x
		#print lmm.sigma_K
		#ipdb.set_trace()
		#lmm.sigma_K = np.array([x[0] ,x[1]])
		lmm.h2 = x[2]
		return lmm.nLL()
		#return np.array([1.0,1.0])

	alphas = np.array([1.0, 1.0, 1.0])
	assert dim==3

	if evalgrid is None:
		step = (maxval-minval)/(nGrid)
		evalgrid = np.arange(minval,maxval+step,step)

	if True:#evalgridH2 is None:
		step = (maxval-minval)/(nGridH2)
		evalgridH2 = np.arange(minval,maxval+step,step)

	simpl = np.zeros([evalgrid.shape[0]* evalgridH2.shape[0],3])
	sigma_K_h2 = np.zeros([evalgrid.shape[0]* evalgridH2.shape[0],3])
	res = np.zeros((evalgrid.shape[0]* evalgridH2.shape[0],2))
	weight = np.zeros((evalgrid.shape[0]* evalgridH2.shape[0]))

	l_simpl = 0
	for i0 in xrange(evalgrid.shape[0]):
		my_sigma_K = np.array([evalgrid[i0],1.0-evalgrid[i0]])
		lmm.sigma_K = my_sigma_K
		for i1 in xrange(evalgridH2.shape[0]):
			simpl[l_simpl] = np.array([evalgrid[i0] * (evalgridH2[i1]), (1.0-evalgrid[i0])*(evalgridH2[i1]), (1.0-evalgridH2[i1])])
			sigma_K_h2[l_simpl] = np.array([evalgrid[i0], (1.0-evalgrid[i0]), (evalgridH2[i1])])
			#simpl[l_simpl] /= simpl[l_simpl].sum()
			res[l_simpl] = f(sigma_K_h2[l_simpl])
			weight[l_simpl] = 2.0 * (evalgridH2[i1])
			l_simpl += 1
	logp = -res[:,0]

	posterior_unnormalized = np.exp(logp-logp.max())
	Z = (posterior_unnormalized * weight).mean()
	log_Z = np.log(Z)
	posterior_normalized = posterior_unnormalized / Z
	#h2_mean = (np.exp(logp-logp.max())[:,np.newaxis]*simpl).sum(0)/(weight * np.exp(logp-logp.max())).sum()
	#h2_mean_weighted = ((weight * np.exp(logp-logp.max()))[:,np.newaxis]*simpl).sum(0)/(weight * np.exp(logp-logp.max())).sum()
	h2_mean = ((posterior_normalized * weight)[:,np.newaxis]*simpl).mean(0)
	lmm.sigma_K = sigma_K_sav
	lmm.h2 = h2_sav
	return simpl, res, posterior_normalized, h2_mean, weight

def evalgrid_simplex(dim, evalgrid = None, nGrid=10, minval=0.00001, maxval = 0.99999, dimF=0, lmm=None):
	'''
	evaluate a function f(x) on all values of a grid.
	--------------------------------------------------------------------------
	Input:
	f(x)    : callable target function
	evalgrid: 1-D array prespecified grid of x-values
	nGrid   : number of x-grid points to evaluate f(x)
	minval  : minimum x-value for optimization of f(x)
	maxval  : maximum x-value for optimization of f(x)
	--------------------------------------------------------------------------
	Output:
	evalgrid    : x-values
	resultgrid  : f(x)-values
	--------------------------------------------------------------------------
	'''
	sigma_K_sav = lmm.sigma_K
	h2_sav = lmm.h2
	def f(x):
		lmm.sigma_K = np.sqrt(np.array([x[0]/(x[0] + x[1]) ,x[1]/ (x[0] + x[1])]))
		lmm.h2 = x[2]
		return lmm.nLL()

	alphas = np.array([1.0, 1.0, 1.0])
	assert dim==3

	if evalgrid is None:
		step = (maxval-minval)/(nGrid)
		evalgrid = np.arange(minval,maxval+step,step)
	evalgrid_gamma = np.zeros((evalgrid.shape[0],3))
	evalgrid_gamma[:,0] = st.gamma.ppf(evalgrid, alphas[0], 0.0, 1.0)
	evalgrid_gamma[:,1] = st.gamma.ppf(evalgrid, alphas[1], 0.0, 1.0)
	evalgrid_gamma[:,2] = st.gamma.ppf(evalgrid, alphas[2], 0.0, 1.0)
	simpl = np.zeros([evalgrid.shape[0]* evalgrid.shape[0]* evalgrid.shape[0],3])
	res = np.zeros((evalgrid.shape[0]* evalgrid.shape[0]* evalgrid.shape[0],2))

	l_simpl = 0
	for i0 in xrange(evalgrid_gamma.shape[0]):
		for i1 in xrange(evalgrid_gamma.shape[0]):
			for i2 in xrange(evalgrid_gamma.shape[0]):
				simpl[l_simpl] = np.array([evalgrid_gamma[i0,0], evalgrid_gamma[i1,1], evalgrid_gamma[i2,2]])
				simpl[l_simpl] /= simpl[l_simpl].sum()
				res[l_simpl] = f(simpl[l_simpl])
				l_simpl += 1
	logp = -res[:,0]
	grid = simpl
	post_h2 = np.exp(logp-logp.max())/np.exp(logp-logp.max()).sum()*logp.shape[0]
	h2_mean = (np.exp(logp-logp.max())[:,np.newaxis]*grid).sum(0)/np.exp(logp-logp.max()).sum()
	lmm.sigma_K = sigma_K_sav
	lmm.h2 = h2_sav
	return simpl, res, post_h2, h2_mean


if __name__ == '__main__':

	np.random.seed(4)
	N=60
	N_test = 20
	D_K=5
	D=2
	P=1
	K=[]
	K_train = []
	K_test = []
	K_star = []
	N_K = 3
	h2 = 0.5

	if 1:
		Y = np.zeros((N,P))
		alphas = 5.0*np.arange(N_K)+1
		alphas/=alphas.sum()
		for i in xrange(N_K):
			X = np.random.randn(N,D_K)
			
			XX = X.dot(X.T) + np.eye(N)
			var = XX.diagonal().mean()
			K.append(XX / var)
			K_test.append(K[-1][:N_test,:][:,:N_test])
			K_train.append(K[-1][N_test:,:][:,N_test:])
			K_star.append(K[-1][:N_test,:][:,N_test:])
			beta = alphas[i] / np.sqrt(D_K) * np.random.randn(D_K,P)
			Y += (X.dot(beta))
		
		X = np.random.randn(N,D)
		X[:,0] = 1.0

		beta = (1.0) / np.sqrt(D) * np.random.randn(D,P)
		Y += (X.dot(beta))
		
		Y = np.sqrt(h2) * Y + np.sqrt(1.0-h2) * np.random.randn(N,P)


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
	lmm.h2 = 0.5
	nll = lmm.nLL()
	#nll1 = lmm.nLL_P()
	#nll1000 = lmm.nLL_P()
	res = lmm.find_h2(nGridH2=10, minH2=0.0, maxH2=0.99999, estimate_Bayes=False)

	lmm.sigma_K = alphas
	#nlla = lmm.nLL_P()
	
	nll1a = lmm.nLL()

	lmm.dof = 1.0
	lmm.scale2 = 1.0
	nll1000a = lmm.nLL_P()
	nll1000a_K = lmm.nLL()
	resa_b = lmm.find_h2(nGridH2=10, minH2=0.0, maxH2=0.99999, estimate_Bayes=False)
	resa_ml = lmm.find_h2(nGridH2=10, minH2=0.0, maxH2=0.99999, estimate_Bayes=False)
	post = lmm.posterior_h2(nGridH2=100, minH2=0.0, maxH2=0.99999)

	dim = 3

	result_evalgrid_simplex = evalgrid_simplex(dim, evalgrid = None, nGrid=10, minval=0.0000001, maxval = 0.9999999, dimF=0, lmm=lmm)
	x,y = bc2xy(res[0])

	result_evalgrid_h2 = evalgrid_h2(dim, evalgrid = None, nGrid=30, nGridH2=300, minval=0.0000001, maxval = 0.9999999, dimF=0, lmm=lmm)
	x_,y_ = bc2xy(result_evalgrid_h2[0])
	opt_ = result_evalgrid_h2[2].argmax()
	x_mean,y_mean = bc2xy(result_evalgrid_h2[3])
	#def f(x):
	#	lmm.sigma_K = np.array([x[0]/(x[0] + x[1]) ,x[1]/ (x[0] + x[1])])
	#	lmm.h2 = x[2]
	#	return lmm.nLL()


	if 'plot' in sys.argv:
		import pylab as plt
		fig = plt.figure()
		plt.scatter(x_,y_,3,result_evalgrid_h2[2],edgecolors='none',cmap='hot')
		
		plt.scatter(x_[opt_],y_[opt_],3,result_evalgrid_h2[2][opt_],edgecolors='black',cmap='hot')
		plt.scatter(x_mean,y_mean,3,1,edgecolors='green',cmap='hot')

	if 'plot3d' in sys.argv:
		import pylab as plt
		from mpl_toolkits.mplot3d import Axes3D
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(xs=x_,ys=y_,zs=result_evalgrid_h2[2],s=1,c=result_evalgrid_h2[2],edgecolors='none',cmap='hot')
		ax.scatter(x_mean,y_mean,3,1,edgecolors='green',cmap='hot')

	Y_pred_test_OLS = lmm.predict(Xstar=X_test, Kstar=K_star, GLS=False)
	residual = Y_test - Y_pred_test_OLS
	MSE_test = (residual * residual).mean()
	R_test_OLS = np.corrcoef(Y_pred_test_OLS[:,0],Y_test[:,0])[1,0]

	Y_pred_train_OLS = lmm.predict(Xstar=X_train, Kstar=K_train, GLS=False)
	residual = Y_train - Y_pred_train_OLS
	MSE_train = (residual * residual).mean()
	R_train_OLS = np.corrcoef(Y_pred_train_OLS[:,0],Y_train[:,0])[1,0]

	print "train R2 OLS: %.4f, test R2 OLS: %.4f" % (R_train_OLS, R_test_OLS)

	Y_pred_test_GLS = lmm.predict(Xstar=X_test, Kstar=K_star, GLS=True)
	residual = Y_test - Y_pred_test_GLS
	MSE_test = (residual * residual).mean()
	R_test_GLS = np.corrcoef(Y_pred_test_GLS[:,0],Y_test[:,0])[1,0]

	Y_pred_train_GLS = lmm.predict(Xstar=X_train, Kstar=K_train, GLS=True)
	residual = Y_train - Y_pred_train_GLS
	MSE_train = (residual * residual).mean()
	R_train_GLS = np.corrcoef(Y_pred_train_GLS[:,0],Y_train[:,0])[1,0]

	print "train R2 GLS: %.4f, test R2 GLS: %.4f" % (R_train_GLS, R_test_GLS)
	

	def nll(x,i):
		lmm.sigma_K[i] = x[i]
		return lmm.nLL()[0][0]

	def grad_nll(x,i):
		lmm.sigma_K[i] = x[i]
		return lmm.derivative_nLL_dK()

	mcheck_grad(func=nll, grad=grad_nll, x0=np.array([0.5,0.5]),
                     allerrs=False)

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


