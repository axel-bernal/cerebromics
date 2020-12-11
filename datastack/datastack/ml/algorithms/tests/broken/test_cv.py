from sklearn import linear_model
from sklearn import cross_validation
import scipy.stats as st
import pandas as pd
import numpy as np
import sys
#sys.path.append("../../..")
from linreg import OLS

class RidgeTestCV(object):

	def __init__(self, Y, covariates=None, alphas=2.0**np.arange(-10,10), normalize=True, shuffle=True, random_state=1, cv_inner=None, add_intercept=True, verbose=False):
		"""
		Implements a cross-validation-based set test. Test for a given input matrix X and the target Y,
		conditioned on the covariates in a Ridge regression model. The ridge is adjusted in a nested cross validation loop.

		Args:
			Y				targets
			covariates		regression covariates to be conditioned on
			alphas			grid of regularization parameters to be searched over in the inner loop
			normalize		normalize the columns of X to have variance 1?
							Set to False if you want to use weighted features. (default True)
			shuffle			boolean indicator if the data matrix are being shuffled or not (default: True)
			random_state	random seed for the shuffling
			cv_inner		int of number inner folds for searching over alphas, or inner cross validation object, 
							if None, use Generalized cross validation (default: None)
			add_intercept	Boolean. Add an intercept to the regression model? (default: True)
			verbose			Boolean. Print our lots of stuff over the cross validation?
		"""

		self.alphas = alphas
		
		self.dim_Y = len(Y.shape)
		if len(Y.shape) == 1:
			self.Y = Y[:,np.newaxis]
		else:
			self.Y = Y
		assert self.Y.shape[1]==1, "only single phenotype supported at this time, found %i" % Y.shape[1]

		if covariates is None:
			covariates = np.zeros((Y.shape[0],0))
		
		self.covariates = covariates
		self.random_state = random_state
		self.shuffle = shuffle
		self.normalize = normalize
		self.verbose = verbose
		self.add_intercept = add_intercept

		self.ols_covariates = OLS(X=self.covariates,Y=self.Y, add_intercept=self.add_intercept)#regress out covariates

		self.model = linear_model.RidgeCV(alphas=alphas, normalize=normalize, cv=cv_inner, fit_intercept=True )		#with the mean it is conservative (on null data P-values are almost always close to 1.0)
		#self.model = linear_model.RidgeCV(alphas=alphas, normalize=normalize, cv=cv_inner, fit_intercept=False )	#without an intercept this overfits the mean across the folds

	def test_permutation(self, X, n_permutations, cv_outer=10, n_jobs=1):
		"""
		Computes the permutation-based cross-validation set test for a given input matrix X and the target Y, conditioned on the covariates.

		Args:
			X 				inputs (These are being tested jointly in a set test)
			n_permutations	int specifying the number of permutations to be conducted. 
							The resulting P-value can not be smaller than 1/n_permutaions
			cv_outer		int or sklearn.cross_validation object for the outer loop cross validation
			n_jobs			int specifying the number of jobs for parallelizing the outer cross validation loop

		Returns:
			pval    		permutation P-value
			cv_scores_mean 	mean of the cross validation scores
			cv_score_perm	permuted cross validation scores
		"""
		#regress out the covariates from X and Y
		Y = self.Y - self.ols_covariates.predict()
		X_ = X - self.ols_covariates.predict(Y=X)
		if type(cv_outer)==int:
			cv_outer = cross_validation.KFold(n=self.Y.shape[0], n_folds=cv_outer, indices=None, shuffle=self.shuffle, random_state=self.random_state)

		if n_permutations > 0:
			cv_scores_mean, cv_score_perm, pval = cross_validation.permutation_test_score(self.model, X_, Y[:,0], cv=cv_outer, verbose=self.verbose, n_permutations=n_permutations, n_jobs=n_jobs) #, scoring='mean_squared_error')
		return pval, cv_scores_mean, cv_score_perm

	def regress_out_covariates(self,X):
		"""
		regress out the covariates from a new variable X
		"""
		ols_covariates = OLS(X=self.covariates,Y=X, add_intercept=self.add_intercept)#regress out covariates
		residual = ols_covariates._residual()
		return residual


	def test(self, X, cv_outer=10, n_jobs=1, repeats=1,include_covariates=False, include_bias=True):
		"""
		Computes the cross-validation set test for a given input matrix X and the target Y, conditioned on the covariates.
		Significance is measured using a t-test for correlation between cross-validated predictions and the target Y.

		Args:
			X 				inputs (These are being tested jointly in a set test)
			n_permutations	int specifying the number of permutations to be conducted. 
							The resulting P-value can not be smaller than 1/n_permutaions
			cv_outer		int number folds for the outer loop cross validation, or cross_validation object
			n_jobs			int specifying the number of jobs for parallelizing the outer cross validation loop
			repeats			int how often to repeat cross validation? reduces the variance in the cross-validated predictions.
							Values larger than 1 require cv_outer to be an int and self.random_state to be int or None.
			include_covariates	Boolean. include covariates in the final model? They should theoretically not be needed. (default: False)
			include_bias	Boolean. include bias in the final model? (default: True) 
		Returns:
			pval    		t-test P-value
			rsquared 		coefficient of determination between cross-validated predictions and the target Y
			tstat			t-statistic
		"""
		#implement nested cross validation loop
		
		#regress out the covariates from X and Y
		Y = self.ols_covariates._residual()
		X_ = self.regress_out_covariates(X=X)
		
		if (repeats>1) and (type(cv_outer)!=int):
			raise Exception("cv_outer has to be integer for repeats > 1")
		cv_pred = np.zeros((Y.shape[0],repeats))
		for repeat in xrange(repeats):
			if (self.random_state is not None) and (repeats>0):
				if type(self.random_state) == int:
					random_state = self.random_state + repeat
				else:
					raise Exception("non-int random state is not supported for repeats > 1")
			else:
				random_state = random_state
			if type(cv_outer) == int:
				cv_outer_ = cross_validation.KFold(n=self.Y.shape[0], n_folds=cv_outer, indices=None, shuffle=self.shuffle, random_state=random_state)
			cv_pred[:,repeat] = cross_validation.cross_val_predict(self.model, X_, Y[:,0], cv=cv_outer_, verbose=self.verbose, n_jobs=n_jobs) #, scoring='mean_squared_error')	
		#cv_pred
		if include_covariates:
			X_final = np.concatenate((cv_pred.mean(1)[:,np.newaxis],self.covariates),1)
		else:
			X_final = cv_pred.mean(1)[:,np.newaxis]
		ols = OLS(X=X_final,Y=Y, add_intercept=True)	#the data has already been mean centered, so we don't want to add an intercept
		tstats, pvalue_t = ols.ttest(onesided=True)		#we only want to check if we can predict well, not if we can predict significantly bad.
		rsquared = ols._rsquared()
		residual = Y - cv_pred.mean(1)[:,np.newaxis]
		
		#import ipdb;ipdb.set_trace()
		return pvalue_t[0,0], rsquared[0], tstats[0,0]

if __name__ == '__main__':
	infile = "../../../cerebro-aging/telomere_length/regression-telomere/tel_lengths_vs_pheno.CCCTAA.U.txt"
	features = pd.read_csv(infile, sep="\t")

	# Read features (one row per sample, with columns for telomere length,
	# age, gender, and SNP allele readings (0,1,2 or 3)

	features.ix[features.Gender=='Male', 'Gender'] = 0
	features.ix[features.Gender=='Female', 'Gender'] = 1


	#parameters
	#alpha = 1.0 #only used for non-cross validated ridge
	alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
	shuffle = True
	random_state = 1
	n_permutations = 10000 #(smallest P-value possible = 1/n_permutaions)
	n_permutations = 0 #For 0 permutations just get the cross validated score
	n_jobs = 8	#parallelize cross validation

	# sometimes there are two Gender columns, Gender_x and Gender_y. Same for age.
	if 'Gender' not in features.columns:
	  if 'Gender_x' in features.columns:
	    features.rename(columns={'Gender_x': 'Gender'}, inplace=True)
	    if (args.debug):
	      print "Gender_x renamed"
	if 'Age' not in features.columns:
	  if 'Age_x' in features.columns:
	    features.rename(columns={'Age_x': 'Age'}, inplace=True)
	    if (args.debug):
	      print "Age_x renamed"

	# eliminate any rows where the feature is NaN
	tel_len_col = "k_17"
	features = features[np.isfinite(features[tel_len_col])]

	cv_Y = features[[tel_len_col]].values.flatten()

	# try cross validation

	#cross validation object for the outer loop (used to produce the R^2)
	cv_outer = cross_validation.KFold(n=cv_Y.shape[0], n_folds=10, indices=None, shuffle=shuffle, random_state=random_state)
	#cross validation object for the inner loop (used to adjust the ridge (alpha) parameter used in the outer loop)
	cv_inner = None #None uses leave-one-out and is the fastest (=generalized cross validation)

	#regr = linear_model.Ridge(alpha=alpha, normalize=True, copy_X=True)


	regr = linear_model.RidgeCV(alphas=alphas, normalize=True, cv=cv_inner )

	cv_X = features[['Age']].values
	cv_gender = np.array(features[['Gender']].values,dtype="float")
	if 1:
		cvtest = RidgeTestCV(Y=cv_Y, covariates=cv_gender, alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0], normalize=True, shuffle=True, random_state=1, cv_inner=None, add_intercept=True)
		testresult = cvtest.test(X=cv_X, cv_outer=10, repeats=1)
		print "age, gender as covariate, no repeats: " + str(testresult)

		testresult_repeated = cvtest.test(X=cv_X, cv_outer=10, repeats=50)
		print "age, gender as covariate, 50 repeats: " + str(testresult_repeated)

		cvtest = RidgeTestCV(Y=cv_Y, covariates=None, alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0], normalize=True, shuffle=True, random_state=1, cv_inner=None, add_intercept=True)
		testresult = cvtest.test(X=cv_X, cv_outer=10, repeats=1)
		print "age, no covariates, no repeats: " + str(testresult)

		testresult_repeated = cvtest.test(X=cv_X, cv_outer=10, repeats=50)
		print "age, no covariates, 50 repeats: " + str(testresult_repeated)

	if 0:	#this is not needed:
		if n_permutations > 0:
			cv_scores_mean, cv_score_perm, pval = cross_validation.permutation_test_score(regr, cv_X, cv_Y, cv=cv_outer, verbose=1, n_permutations=n_permutations, n_jobs=n_jobs) #, scoring='mean_squared_error')
			print("Age: %0.2f, P-value = %.2e, %i permutations yield %0.2f (+/- %0.2f)" % (cv_scores_mean, pval, n_permutations, cv_score_perm.mean(), cv_score_perm.std() * 2))
			
		else:
			cv_scores = cross_validation.cross_val_score(regr, cv_X, cv_Y, cv=cv_outer, verbose=1, n_jobs=n_jobs) #, scoring='mean_squared_error')
			cv_pred = cross_validation.cross_val_predict(regr, cv_X, cv_Y, cv=cv_outer, verbose=1, n_jobs=n_jobs) #, scoring='mean_squared_error')
			pearson_r = st.pearsonr(cv_pred,cv_Y)
			ols = OLS(X=cv_pred[:,np.newaxis],Y=cv_Y)
			fstat, pvalue_F = ols.ftest(M=None, i_F = None)
			tstats, pvalue_t = ols.ttest()		
			print cv_scores
			print("Age: %0.2f (+/- %0.2f), pearson_r = %.2f pvalue_r = %.2e pvalue_F = %.2e pvalue_t = %.2e" % (cv_scores.mean(), cv_scores.std() * 2, pearson_r[0], pearson_r[1], pvalue_F, pvalue_t[0]))

	
	cv_X = np.array(features[['Gender']].values,dtype="float")
	if 0: #this is not needed
		if n_permutations > 0:
			cv_scores_mean, cv_score_perm, pval = cross_validation.permutation_test_score(regr, cv_X, cv_Y, cv=cv_outer, verbose=1, n_permutations=n_permutations, n_jobs=n_jobs) #, scoring='mean_squared_error')
			print("Gender: %0.2f, P-value = %.2e, %i permutations yield %0.2f (+/- %0.2f)" % (cv_scores_mean, pval, n_permutations, cv_score_perm.mean(), cv_score_perm.std() * 2))
		else:
			cv_scores = cross_validation.cross_val_score(regr, cv_X, cv_Y, cv=cv_outer, verbose=1, n_jobs=n_jobs) #, scoring='mean_squared_error')
			cv_pred = cross_validation.cross_val_predict(regr, cv_X, cv_Y, cv=cv_outer, verbose=1, n_jobs=n_jobs) #, scoring='mean_squared_error')
			pearson_r = st.pearsonr(cv_pred,cv_Y)
	 		ols = OLS(X=cv_pred[:,np.newaxis],Y=cv_Y)
			ols2 = OLS(X=np.concatenate((cv_pred[:,np.newaxis],np.ones((cv_Y.shape[0],1))),1),Y=cv_Y,add_intercept=False)
			ols3 = OLS(X=np.concatenate((cv_pred[:,np.newaxis],np.ones((cv_Y.shape[0],1))),1),Y=cv_Y,add_intercept=True)
			fstat, pvalue_F = ols.ftest(M=None, i_F = None)
			tstats, pvalue_t = ols.ttest()
			fstat2, pvalue_F2 = ols2.ftest(M=None, i_F = [0])
			tstats2, pvalue_t2 = ols2.ttest()	
			fstat3, pvalue_F3 = ols3.ftest(M=None, i_F = [0])
			tstats3, pvalue_t3 = ols3.ttest()
			Xscramble = np.random.randn(cv_pred.shape[0],1)
			p1 = ols.predict(Xstar = Xscramble)
			p2 = ols2.predict(Xstar = np.concatenate((Xscramble,np.ones((cv_Y.shape[0],1))),1))
			p3 = ols3.predict(Xstar = np.concatenate((Xscramble,np.ones((cv_Y.shape[0],1))),1))
			print cv_scores
			print("Gender: %0.2f (+/- %0.2f), pearson_r = %.2f pvalue_r = %.2e pvalue_F = %.2e pvalue_t = %.2e" % (cv_scores.mean(), cv_scores.std() * 2, pearson_r[0], pearson_r[1], pvalue_F, pvalue_t[0]))
	
	if 1:
		cvtest = RidgeTestCV(Y=cv_Y, covariates=None, alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0], normalize=True, shuffle=True, random_state=1, cv_inner=None, add_intercept=True)
		testresult = cvtest.test(X=cv_X, cv_outer=10, repeats = 1)
		print "gender, no covariates, no repeats: " + str(testresult)

		cvtest_gender_cond_age = RidgeTestCV(Y=cv_Y, covariates=features[['Age']].values, alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0], normalize=True, shuffle=True, random_state=1, cv_inner=None, add_intercept=True)
		testresult_gender_cond_age = cvtest_gender_cond_age.test(X=cv_X, cv_outer=10, repeats = 1)
		print "gender, age as covariate, no repeats: " + str(testresult_gender_cond_age)

		cvtest_gender_cond_age_repeat = RidgeTestCV(Y=cv_Y, covariates=features[['Age']].values, alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0], normalize=True, shuffle=True, random_state=1, cv_inner=None, add_intercept=True)
		testresult_gender_cond_age_repeat = cvtest_gender_cond_age_repeat.test(X=cv_X, cv_outer=10, repeats = 50)
		print "gender, age as covariate, 50 repeats: " + str(testresult_gender_cond_age_repeat)
	if 1:
		cv_X = np.random.randn(cv_X.shape[0],cv_X.shape[1])
		cvtest = RidgeTestCV(Y=cv_Y, covariates=None, alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0], normalize=True, shuffle=True, random_state=1, cv_inner=None, add_intercept=True)
		testresult = cvtest.test(X=cv_X, cv_outer=10, repeats = 1)
		print "random, no covariates, no repeats: " + str(testresult)

		cvtest_gender_cond_age = RidgeTestCV(Y=cv_Y, covariates=features[['Age']].values, alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0], normalize=True, shuffle=True, random_state=1, cv_inner=None, add_intercept=True)
		testresult_gender_cond_age = cvtest_gender_cond_age.test(X=cv_X, cv_outer=10, repeats = 1)
		print "random, age as covariate, no repeats: " + str(testresult_gender_cond_age)

		cvtest_gender_cond_age_repeat = RidgeTestCV(Y=cv_Y, covariates=features[['Age']].values, alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0], normalize=True, shuffle=True, random_state=1, cv_inner=None, add_intercept=True)
		testresult_gender_cond_age_repeat = cvtest_gender_cond_age_repeat.test(X=cv_X, cv_outer=10, repeats = 50)
		print "random, age as covariate, 50 repeats: " + str(testresult_gender_cond_age_repeat)

