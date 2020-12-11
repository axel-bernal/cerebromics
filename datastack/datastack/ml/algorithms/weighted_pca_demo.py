import sys
import pca_weighted
import matplotlib.pyplot as plt
import seaborn as sns
import datastack
from datastack.dbs import rdb as rosetta
import pandas as pd
use_chi_ethnicity = True	#false would use all 29 ethnicities, which are mostly zero...


import json
class JsonConverter():
    @classmethod
    def split_array(self, s, title="PC"):
        l = json.loads(s)
        t = [title+str(i+1) for i in range(len(l))]
        return dict(zip(t, l))

def face_PC(N):
    return ["dynamic.FACE.face.v1_visit1.DepthPC.{}".format(i+1) for i in range(N)]


rdb = rosetta.RosettaDBMongo(host="rosetta.hli.io")
rdb.initialize(namespace="hg19")
if "push" in sys.argv:
	rdb_hg38 = rosetta.RosettaDBMongo()
	rdb_hg38.initialize(rdb_hg38.get_versions(namespace="hg38_noEBV")[-1], namespace="hg38_noEBV")

face_embedding_version='v3'

if use_chi_ethnicity:
	filters = {"lab.ProjectID": "FACE"}
else:
	filters = None
headers = ['qc.sample_key',"qc.sample_name"]
if use_chi_ethnicity:
	eth_keys = [u'facepheno.ethnic.EUR',
		u'facepheno.ethnic.AFR',
		u'facepheno.ethnic.AMR',
		u'facepheno.ethnic.EAS',
		u'facepheno.ethnic.SAS']
else:
	eth_keys = rdb.find_keys("dynamic.FACE.ethnicity.v1.")
coloremb_keys = rdb.find_keys('dynamic.FACE.face.{}_visit1.ColorPC'.format(face_embedding_version))
pca_keys = ['genomics.kinship.pc.%i' % i for i in xrange(1,1001)]

xy = rdb.query(filters=filters, keys=headers+eth_keys+pca_keys+coloremb_keys, with_nans=False)
X_pc = xy[pca_keys]
X_ethnicity = xy[eth_keys]

#create the PCA model
pcae = pca_weighted.EthnicityPCA()
#fit the model and obtain transformations (note that the ethnicity values have the number of ethnicities as leading dimension)
X_pce = pcae.fit_transform(X=X_pc.values, ethnicity=X_ethnicity.values.T)

pcaw = pca_weighted.WeightedPCA()
X_pcw_eur = pcaw.fit_transform(X=X_pc.values, weights=X_ethnicity['facepheno.ethnic.EUR'].values)

X_pcw_afr = pcaw.fit_transform(X=X_pc.values, weights=X_ethnicity['facepheno.ethnic.AFR'].values)

#color prediction
X_color = xy["dynamic.FACE.face.{}_visit1.ColorPC".format(face_embedding_version)].apply(lambda s: pd.Series(JsonConverter.split_array(s, title="dynamic.FACE.face.{}_visit1.ColorPC.".format(face_embedding_version))))

#fit the model and obtain transformations (note that the ethnicity values have the number of ethnicities as leading dimension)
X_color_pce = pcae.fit_transform(X=X_color.values, ethnicity=X_ethnicity.values.T)  

if 1:
	plt.figure()
	plt.scatter(X_pcw_eur[:,0],X_pcw_eur[:,1],c=X_ethnicity['facepheno.ethnic.EUR'].values, s=5)
	plt.title("weighted EUR")
	plt.xlabel("EUR weighed genome PC1")
	plt.ylabel("EUR weighed genome PC2")
	plt.savefig("weighted_geno_EUR_pce1pce2.png")
	plt.close("all")

if 1:
	plt.figure()
	plt.scatter(X_pcw_afr[:,0],X_pcw_afr[:,1],c=X_ethnicity['facepheno.ethnic.AFR'].values, s=5)
	plt.title("weighted AFR")
	plt.xlabel("AFR weighed genome PC1")
	plt.ylabel("AFR weighed genome PC2")
	plt.savefig("weighted_geno_AFR_pce1pce2.png")
	plt.close("all")


for i, col in enumerate(X_ethnicity.columns):
	if 0:		
		plt.figure()
		plt.scatter(X_pce[i,:,0],X_pce[i,:,1],c=X_ethnicity.values[:,i], s=5)
		plt.title(col)
		plt.xlabel("ethnicity reweighed genome PC1")
		plt.ylabel("ethnicity reweighed genome PC2")
		plt.savefig("ethnicity_geno_"+col+"_pce1pce2.png")
		plt.close("all")

	if 0:
		plt.figure()
		plt.scatter(X_color_pce[i,:,0],X_color_pce[i,:,1],c=X_ethnicity.values[:,i], s=5)
		plt.title(col)
		plt.xlabel("ethnicity reweighed color PC1")
		plt.ylabel("ethnicity reweighed color PC2")
		plt.savefig("ethnicity_color_"+col+"_pce1pce2.png")
		plt.close("all")

	if 0:
		plt.figure()
		plt.scatter(X_color.values[:,0],X_color.values[:,1],c=X_ethnicity.values[:,i], s=5)
		plt.title(col)
		plt.xlabel("original color PC1")
		plt.ylabel("original color PC2")
		plt.savefig("color_"+col+"_pce1pce2.png")
		plt.close("all")


	if 0:
		plt.figure()
		plt.scatter(X_color.values[:,0],X_pc.values[:,0],c=X_ethnicity.values[:,i], s=5)
		plt.title(col)
		plt.xlabel("original color PC1")
		plt.ylabel("original genome PC1")
		plt.savefig("geno_color_"+col+"_pce1pce1.png")
		plt.close("all")

	if 0:
		plt.figure()
		plt.scatter(X_color.values[:,0],X_pce[i,:,0],c=X_ethnicity.values[:,i], s=5)
		plt.title(col)
		plt.xlabel("original color PC1")
		plt.ylabel("ethnicity reweighed genome PC1")
		plt.savefig("ethnicity_geno_color_"+col+"_pce1pce1.png")
		plt.close("all")

	if 0:
		plt.figure()
		plt.scatter(X_color_pce[i,:,0],X_pce[i,:,0],c=X_ethnicity.values[:,i], s=5)
		plt.title(col)
		plt.xlabel("ethnicity reweighed color PC1")
		plt.ylabel("ethnicity reweighed genome PC1")
		plt.savefig("ethnicity_geno_color_"+col+"_pce1pce1.png")
		plt.close("all")
	
	if 0:
		plt.figure()
		plt.scatter(X_pc.values[:,0],X_pce[i,:,0],c=X_ethnicity.values[:,i], s=5)
		plt.title(col)
		plt.xlabel("original genome PC1")
		plt.ylabel("ethnicity reweighed genome PC1")

	if "push" in sys.argv:
		#merge into database
		cnames = ["qc.sample_name"]
		for j in xrange(1,51):
			column_name = "genome.new." + col[-3:] + ".%i" %j
			xy[column_name] = X_pce[i,:,j-1]
			cnames.append(column_name)

		rdb_hg38.join(xy[cnames], index_col='qc.sample_name', join_to="qc.sample_name", prefix="genomepc.new."+ col[-3:])
		rdb.join(xy[cnames], index_col='qc.sample_name', join_to="qc.sample_name", prefix="genomepc.new."+ col[-3:])
