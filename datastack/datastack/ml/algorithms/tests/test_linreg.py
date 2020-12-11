import sys
from datastack.ml.algorithms.linreg import OLS
import unittest
import numpy as np
import cPickle
datafile = "./data/linreg_data.pickle"

def almost_equal(x1,x2,acc=1e-8):
	res = (np.absolute(x1-x2))<acc
	return np.all(res)

def compare_pvalue(pv1,pv2):
	lpv1 = np.log(pv1)
	lpv2 = np.log(pv2)
	return almost_equal(lpv1,lpv2)

def compare_lrt(l1,l2):
	#lrt, dof, pvalue
	res1 = almost_equal(l1[0],l2[0])
	res2 = np.all(l1[1] == l2[1])
	res3 = compare_pvalue(l1[2],l2[2])	
	ret =  res1 and res2 and res3
	return ret

def compare_f(f1,f2):
	#fstats,pvalues
	res1 =  almost_equal(f1[0],f2[0])
	res2 = compare_pvalue(f1[1],f2[1])
	ret = res1 and res2
	return ret

def compare_t(t1,t2):
	#tstats, pvals
	res1 = almost_equal(t1[0],t2[0])
	res2 = compare_pvalue(t1[1],t2[1])	
	ret = res1 and res2
	return ret

class TestLinreg(unittest.TestCase):
	def setUp(self):
		inputfile = open(datafile,'rb')
		self.data = cPickle.load(inputfile)
		inputfile.close()
		self.X = self.data["X"]
		self.Y = self.data["Y"]
	def test_ttest(self):
		ols = OLS(X=self.X,Y=self.Y)
		data = self.data
		self.assertTrue(compare_t(data["ttest1"], ols.ttest()))
		ols = OLS(X=self.X,Y=self.Y[:,0])	
		self.assertTrue(compare_t(data["ttest2"], ols.ttest()))

	def test_ftest(self):
		ols = OLS(X=self.X,Y=self.Y)
		data = self.data
		self.assertTrue(compare_f(data["ftest1"], ols.ftest(i_F=data["i_F1"])))
		self.assertTrue(compare_f(data["ftest2"], ols.ftest(i_F=data["i_F2"])))
		ols = OLS(X=self.X,Y=self.Y[:,0])	
		self.assertTrue(compare_f(data["ftest3"], ols.ftest(i_F=data["i_F3"])))
		self.assertTrue(compare_f(data["ftest4"], ols.ftest(i_F=data["i_F4"])))

	def test_lrtest(self):
		ols = OLS(X=self.X,Y=self.Y)
		data = self.data
		self.assertTrue(compare_lrt(data["lrt1"], ols.lrtest(i_F=data["i_F1"])))
		self.assertTrue(compare_lrt(data["lrt2"], ols.lrtest(i_F=data["i_F2"])))
		ols = OLS(X=self.X,Y=self.Y[:,0])
		self.assertTrue(compare_lrt(data["lrt3"], ols.lrtest(i_F=data["i_F3"])))	
		self.assertTrue(compare_lrt(data["lrt4"], ols.lrtest(i_F=data["i_F4"])))
		


if __name__ == '__main__':
	N=10000
	D=10
	P=2
	
	if "generate_data" in sys.argv:#generate the data
		np.random.seed(1)

		data = {}

		data["X"] = np.random.randn(N,D)
		data["beta"] = 0.04*np.random.randn(D,P)
		data["Y"] = np.random.randn(N,P) + data["X"].dot(data["beta"])
		i_F = np.random.rand(D)>0.5
		i_F[0] = True
		
		#inputfile = open(datafile,'rb')
		#data = cPickle.load(inputfile)
		#inputfile.close()
		X = data["X"]
		Y = data["Y"]
		
		ols = OLS(X=X,Y=Y)

		data["ttest1"] = ols.ttest()

		data["i_F1"] = i_F		
		data["ftest1"] = ols.ftest(i_F=data["i_F1"])
		data["lrt1"] = ols.lrtest(i_F=data["i_F1"])
		print "ftest i_F: " + str(data["ftest1"])
		print "lrtest i_F: " + str(data["lrt1"])
		print "ttest : stat %.6f  pval %.6f" % (data["ttest1"][0][0,0], data["ttest1"][1][0,0])  
		
		data["i_F2"]=np.array([0])
		data["ftest2"]=ols.ftest(i_F=data["i_F2"])
		data["lrt2"] = ols.lrtest(i_F=data["i_F2"])
		print "ftest i_F: " + str(data["ftest2"])
		print "lrtest first weight: stat %.6f dof %i   pval %.6f" % (data["lrt2"] [0][0], data["lrt2"][1][0], data["lrt2"][2][0])

		
		ols = OLS(X=X,Y=Y[:,0])

		data["ttest2"] = ols.ttest()

		data["i_F3"] = np.random.rand(D)>0.5
		data["ftest3"] = ols.ftest(i_F=data["i_F3"])
		data["lrt3"] = ols.lrtest(i_F=data["i_F3"])
		print "ftest i_F: " + str(data["ftest3"])
		print "lrtest i_F: " + str(data["lrt3"])
		print "ttest : stat %.6f  pval %.6f" % (data["ttest2"][0][0,0], data["ttest2"][1][0,0])  
		
		data["i_F4"]=np.array([0])
		data["ftest4"] = ols.ftest(i_F=data["i_F4"])
		data["lrt4"] = ols.lrtest(i_F=data["i_F4"])
		print "ftest i_F: " + str(data["ftest4"])
		print "lrtest first weight: stat %.6f dof %i   pval %.6f" % (data["lrt4"][0], data["lrt4"][1], data["lrt4"][2])

		output = open(datafile, 'wb')
		cPickle.dump(data,output,protocol=0)
		output.close()

	unittest.main()		

