from variance_components import *
import logging

class MixedRegression(LMM):
	def __init__(self,  Y, covariates=None, G=[], rank_threshold=1E-10, sigma_K=None, h2=0.0):
		LMM.__init__(self, Y=Y, covariates=covariates, K=[], rank_threshold=rank_threshold, sigma_K=sigma_K, h2=h2)
		self._G = None
		self.G = G

	@property
	def G(self):
		return self._G

	@G.setter
	def G(self, value):
		self.clear_cache("K")
		if isinstance(value, np.ndarray):
			self._G = [value]
		elif isinstance(value, list):
			self._G = value
		elif value is None:
			self._G = []
		else:
			raise Exception("unknown type for G: %s " % str(type(value)))
	
	@property
	def N_K(self):
	    return len(self.G)

	@property
	def rank(self):
		if len(self.K)>0:
			return self.N
		else:
			rank = 0
			for i in xrange(len(self.G)):
				rank+=self.G[i].shape[1]
			return rank

	def G_stacked(self):
		G = np.empty((self.N,self.rank))
		col = 0
		for i in xrange(len(self.G)):
			G[:,col:col+self.G[i].shape[1]] = self.G[i]
			col += self.G[i].shape[1]
		return G

	def G_weighted(self):
		G = self.G_stacked()
		G = G * np.sqrt(self.D())
		return G

	def Gstar_weighted(self):
		Gstar = np.empty((self.Gstar[0].shape[0],self.rank))
		col = 0
		sqrt_a2 = np.sqrt(self.a2)
		for i in xrange(len(self.Gstar)):
			Gstar[:,col:col+self.Gstar[i].shape[1]] = sqrt_a2[i] * self.Gstar[i]
			col += self.Gstar[i].shape[1]
		return Gstar

	@cached(["K"])
	def eig(self):
		#it is faster using the eigen decomposition of G.T*G but this is more accurate
		try:
			U,S,V = la.svd(self.G_weighted(), full_matrices=False)
			U = U
			S = S*S
		except la.LinAlgError:  # revert to Eigenvalue decomposition
			logging.warning("Got SVD exception, trying eigenvalue decomposition of square of G. Note that this is a little bit less accurate")
			[S_,V_] = la.eigh(self.G_weighted().T.dot(self.G_weighted()))
			if np.any(S_ < -0.1):
				logging.warning("kernel contains a negative Eigenvalue")
			S_nonz=(S_>0)
			S = S_[S_nonz]
			S*=(self.N/S.sum())
			U=self.G_weighted().dot(V_[:,S_nonz]/np.sqrt(S))
		return (S,U)

	def _dD(self):
		dD = np.zeros((len(self.G), self.rank))
		sumsquares = (self.sigma_K*self.sigma_K).sum()
		if not (len(self.G)>0):
			return []
		cols = 0
		for i in xrange(len(self.G)):			
			dD[i,cols:cols+self.G[i].shape[1]] = ( self.scale2 * 2.0 * self.sigma_K[i]/sumsquares)
			cols += self.G[i].shape[1]
		
		for i in xrange(len(self.G)):
			cols = 0
			for j in xrange(0,len(self.G)):
				dD[i,cols:cols+self.G[j].shape[1]] -= (self.scale2 * 2.0 * self.sigma_K[j]*self.sigma_K[j]*self.sigma_K[i]) / (sumsquares * sumsquares)
				cols += self.G[j].shape[1]
		return dD

	def D(self):
		D = np.zeros((self.rank))
		cols = 0
		for i in xrange(len(self.G)):			
			D[cols:cols+self.G[i].shape[1]] = self.a2[i]
			cols += self.G[i].shape[1]
		return D

	@cached(["K"])
	def G_rot(self):
		G = self.G_stacked()
		U = self.eig()[1]
		UG = U.T.dot(G)
		return UG

	@cached(["K","covariates"])
	def X_rot(self):
		if len(self.G)==0:
			return (self.covariates.X, None)
		U = self.eig()[1]
		UX = U.T.dot(self.covariates.X)
		UUX = self.covariates.X - U.dot(UX)
		return (UX, UUX)

	@cached(["Y","K"])
	def Y_rot(self):
		if len(self.G)==0:
			return (self.Y, None)
		U = self.eig()[1]
		UY = U.T.dot(self.Y)
		UUY = self.Y - U.dot(UY)
		return (UY, UUY)


	def predict(self, X, G, GLS=True, update_Gstar=True, marinalize_h2=True):
		"""
		predict a new sample for a single h^2.

		Params:
			X:				array-like of test covariates
			G:				list of test kernels to train data (each of shape Ntest x Ntrain)
			GLS:			boolean if to use the generalized least squares predictor for covariates (=best unbiased predictor)
							or the ordinary least squares predictor (unbiased but larger variance) (default True)
			update_Gstar:	if K has been provided before, then update_Kstar can be set to False to enable some caching (default True)
			marginalize_h2:	predict marginalizing over the noise variance h2 or not? (default True)
		"""
		if marinalize_h2:
			return self.predict_avg(Xstar=X, Gstar=G, GLS=GLS)

		if update_Gstar:
			self.Gstar = G
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

	@cached(["Y","K","covariates","h2"])
	def random_effects(self):
		if not len(self.G)>0:
			return 0.0
		Y_rot = self.Y_rot()
		X_rot = self.X_rot()
		G_rot = self.G_rot() * np.sqrt(self.D())[np.newaxis,:]
		Uresidual = Y_rot[0] - X_rot[0].dot(self.beta()) 
		return computeAKB(self.Sd(), self.denom, UA=G_rot, UB=Uresidual, UUA=None, UUB=None)

	def _predict_K(self):
		if not len(self.G)>0:
			return 0.0
		random_effects = self.random_effects()
		Y_K = self.Gstar_weighted().dot(random_effects)
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

	def predict_avg(self, Xstar, Gstar, GLS=True):
		"""
		predict a new sample for a single h^2.
		"""
		self.Gstar = Gstar

		posterior = self.posterior_h2()

		rv = np.zeros((Xstar.shape[0],self.Y.shape[1]))
		for ii in xrange(len(posterior[1])):
			self.h2 = posterior[1][ii]
			rv_ = self.predict(X=Xstar,G=Gstar,GLS=GLS,update_Gstar=False, marinalize_h2=False) * posterior[3][ii,0] / posterior[1].shape[0]
			rv+=rv_
		return rv



	def _dK(self):
		dD = self._dD()

	@cached(["K","covariates","h2"])
	def _dXKX(self):
		X_rot = self.X_rot()
		G_rot = self.G_rot()
		dD = self._dD()
		XKG = self._AKB(UA=(X_rot[0],None), UB=(G_rot, None))
		dXKX = []
		for i in xrange(dD.shape[0]):
			XKGdD = XKG * dD[i,np.newaxis,:]
			dXKX_ = -XKG.dot(XKGdD.T) * self.h2
			dXKX.append(dXKX_)
		return dXKX

	def _derivative_log_det(self):
		"""
		checks gradient!
		"""
		Sd = self.Sd()
		G_rot = self.G_rot()
		dD = self._dD()
		derivative_trace = []
		G_rot_Sd = G_rot / Sd[:,np.newaxis]
		for i in xrange(dD.shape[0]):
			derivative = self.h2 * (G_rot_Sd * dD[i,np.newaxis,:] * G_rot).sum()
			derivative_trace.append(derivative) 
		return derivative_trace

	@cached(["Y","K","h2"])
	def _derivative_YKY(self):
		"""
		gradient checks!
		"""
		Y_rot = self.Y_rot()
		G_rot = self.G_rot()
		dD = self._dD()
		YKG = self._AKB(UA=(Y_rot[0],None), UB=(G_rot, None))
		dYKY = []
		for i in xrange(dD.shape[0]):
			YKGdD = YKG * dD[i,np.newaxis,:]
			dYKY_ = -(YKG * YKGdD).sum() * self.h2
			dYKY.append(dYKY_)
		return dYKY

	@cached(["K","covariates","h2"])
	def _dXKY(self):
		X_rot = self.X_rot()
		Y_rot = self.Y_rot()
		G_rot = self.G_rot()
		dD = self._dD()
		XKG = self._AKB(UA=(X_rot[0],None), UB=(G_rot, None))
		YKG = self._AKB(UA=(Y_rot[0],None), UB=(G_rot, None))
		dXKY = []
		for i in xrange(dD.shape[0]):
			YKGdD = YKG * dD[i,np.newaxis,:]
			dXKY_ = -XKG.dot(YKGdD.T) * self.h2
			dXKY.append(dXKY_)
		return dXKY
