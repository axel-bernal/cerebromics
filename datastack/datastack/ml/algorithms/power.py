import numpy as np
import scipy.stats as st

class gwpower(object):
	"""
	a simple class for estimating power in GWAS in an OLS model.
	It uses maf, the true underlying effect size and the variance of y to compute power
	at a given significance threshold (alpha). 

	The estimate of power is defined and explained in Section 3 of OLS_introduction.tex
	"""
	def __init__(self, maf, effect_size, var_Y=1.0, unit_X = "dosage"):
		"""
		a simple class for estimating power (the probability of calling the SNP significant) in GWAS in an OLS model.
		It uses maf, the true underlying effect size and the variance of Y to compute power
		at a given significance threshold (alpha). 

		The estimate of power is defined and explained in Section 3 of OLS_introduction.tex

		Args:

			maf           minor allele frequency of the causal variant
			effect_size   effect size (beta weight) of the causal variant
			var_Y         the variance of the phenotype (default = 1.0)
			unit_X        The encoding of the SNP. Currently only "dosage" is allowed
			              dosage: 0-1-2 encoding counting the number of (minor) alleles.
		"""
		self.maf = maf
		self.effect_size = effect_size
		self.var_Y = var_Y
		self.unit_X = unit_X
		if self.unit_X == "dosage":
			max_X = 2.0
			self.var_X = max_X * (self.maf * (1.0-self.maf)) # expectation of 1/N (X^T X)
		else:
			raise Exception("not implemented: "+self.unit_X )
		#self.var_beta = self.var_Y / self.var_X / self.N

	
	def gwpower(self, N, alpha=5e-8):
		"""
		computes power for given sample size N and significance threshold alpha


		Args:
			N          sample size (int)
			alpha      significance threshold (default 5e-8)

		Returns:
			power (the probability of calling the SNP significant)
		"""
		#solve two integrals, one above 0 and one below 0

		#Expectation of the H_0 (beta=0) survival function (1-CDF)  over the distribution of the test statistic using the true effect size.
		#expectation of the estimate is equal to the true effect size: self.effect_size
		var_beta = self.var_Y / self.var_X / N
		ste = np.sqrt(var_beta)
		mean_beta = self.effect_size
		acceptance_threshold = st.norm.isf(alpha*0.5, loc=0.0, scale=ste )
		power = st.norm.sf(acceptance_threshold, loc=np.absolute(self.effect_size), scale=ste )
		power += st.norm.cdf(-acceptance_threshold, loc=np.absolute(self.effect_size), scale=ste )
		return power

def gwas_power(effect_size, maf, N, var_Y=1.0, alpha=5e-08, unit_X = "dosage"):
	"""
	a simple function for estimating power (the probability of calling the SNP significant) in GWAS in an OLS model.
	It uses maf, the true underlying effect size and the variance of Y to compute power
	at a given significance threshold (alpha). 

	The estimate of power is defined and explained in Section 3 of OLS_introduction.tex

	Args:

		maf           minor allele frequency of the causal variant
		N             sample size (int)		
		effect_size   effect size (beta weight) of the causal variant
		var_Y         the variance of the phenotype (default = 1.0)
		alpha         significance threshold (default 5e-8)
		unit_X        The encoding of the SNP. Currently only "dosage" is allowed
		              dosage: 0-1-2 encoding counting the number of (minor) alleles.

	Returns:
		power (the probability of calling the SNP significant)
	"""

	p = gwpower(maf=maf, effect_size=effect_size,var_Y=var_Y, unit_X = unit_X)
	return p.gwpower(N=N,alpha=alpha)

def est_pow(maf = 0.252, effect_size=-0.097, frac_var_expl=0.37, N=300):
	"""
	estimate power for the telomere length study in Codd et al.
	We first estimate the total phenotype variance from the estimated variance explained
	given in Codd et al. and then use the function gwas_power() defined above.
	"""
	varX = maf*(1-maf)*2
	var_expl = varX*effect_size*effect_size
	var_Y = var_expl / frac_var_expl

	power_ = gwas_power(effect_size=effect_size, maf=maf,N=400,var_Y=var_Y,alpha=5e-8)
	return power_, var_Y

if __name__ == '__main__':
	print "compute power on the 7 significant SNPs associated with Telomere length in Codd et al. on our data set:"

	N= 300
	power,var_Y = est_pow(maf = 0.252,effect_size=-0.097,frac_var_expl=0.37,N=N)
	print "power %.6f" % power
	print "var_Y %.6f" % var_Y
	power,var_Y = est_pow(maf = 0.514,effect_size=-0.078,frac_var_expl=0.31,N=N)
	print "power %.6f" % power
	print "var_Y %.6f" % var_Y
	power,var_Y = est_pow(maf = 0.217,effect_size=-0.074,frac_var_expl=0.19,N=N)
	print "power %.6f" % power
	print "var_Y %.6f" % var_Y
	power,var_Y = est_pow(maf = 0.865,effect_size=-0.069,frac_var_expl=0.11,N=N)
	print "power %.6f" % power
	print "var_Y %.6f" % var_Y
	power,var_Y = est_pow(maf = 0.709,effect_size=-0.062,frac_var_expl=0.09,N=N)
	print "power %.6f" % power
	print "var_Y %.6f" % var_Y
	power,var_Y = est_pow(maf = 0.869,effect_size=-0.062,frac_var_expl=0.09,N=N)
	print "power %.6f" % power
	print "var_Y %.6f" % var_Y
	power,var_Y = est_pow(maf = 0.858,effect_size=-0.056,frac_var_expl=0.08,N=N)
	print "power %.6f" % power
	print "var_Y %.6f" % var_Y