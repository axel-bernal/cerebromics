import numpy as np
import pandas as pd
import sys, os, time
import argparse
from datastack.dbs import rdb as rosetta
import subprocess
from datastack.ml.algorithms.kernel_snps import LinearSnpKernel

from datastack.ml.algorithms.pca_covariates import KernelPCA
from pysnptools.snpreader.bed import Bed
parser = argparse.ArgumentParser(description="input parser")

default_reference = "19"

parser.add_argument('--bfile', type=str, default=("HG%s_IsaacVariantCaller_AllSamples_PASS_MAF_001_gVCF" % default_reference))
parser.add_argument('--bmerge', type=str, default="./Jay_Flatley/HG19_PASS_MAF_001.PG0000100-BLD.genome.vcf.gz_Tr")
parser.add_argument('--reference', type=str, default=default_reference)
parser.add_argument('--bedpath', type=str, default="./bed/")
parser.add_argument('--tmp', type=str, default="./bed/tmp/")
parser.add_argument('--s3', type=str, default="s3://hli-mv-data-science/bhicks/")
parser.add_argument('--out', type=str, default="./bed/out/")
parser.add_argument('--maf', type=float, dest='maf', default=0.05)
parser.add_argument('--hwe', type=float, dest='hwe', default=0.0001)
parser.add_argument('--geno', type=float, dest='geno', default=0.1)
parser.add_argument('--plink_path', type=str, dest='plink_path', default="plink")
parser.add_argument('--fn_keep', type=str, default="./bed/keep_FACE_HG%s.txt" % default_reference)
parser.add_argument('--recompute', action='store_true')

args = parser.parse_args()

t00 = time.time()

print args

basefilename = os.path.join(args.bedpath, args.bfile)
args.bfile_filtered = basefilename + "_maf%f_hwe%f_geno%f" % (args.maf, args.hwe, args.geno)

if False: #args.recompute or not os.path.exists(basefilename + ".bed"):
    print "downloading large bed file from S3"
    t0 = time.time()
    basefilenameS3 = args.s3 + args.bfile
    print "aws s3 cp " + basefilenameS3 + ".fam ./bed/"
    cmd = ["aws", "s3", "cp", basefilenameS3 +".fam", "./bed/"]
    subprocess.check_call(cmd)
    print "aws s3 cp " + basefilenameS3 + ".bim ./bed/"
    cmd = ["aws", "s3", "cp", basefilenameS3+".bim", "./bed/"]
    subprocess.check_call(cmd)
    print "aws s3 cp " + basefilenameS3 + "_Tr.bed ./bed/"
    cmd = ["aws", "s3", "cp", basefilenameS3+"_Tr.bed", "./bed/"]
    subprocess.check_call(cmd)
    print "mv " + basefilename + "_Tr.bed " + basefilename + ".bed"
    cmd = ["mv", basefilename + "_Tr.bed", basefilename + ".bed"]
    subprocess.check_call(cmd)
    print "done after %.2fs.\n"  %(time.time()-t0)


if 0:
    print "retrieving FACE sample list from Rosetta DB."
    t0 = time.time()
    rdb = rosetta.RosettaDBMongo(host="172.31.22.29", port=27017)# "rosetta.hli.io")
if args.reference == "38":
    namespace = u'hg38_noEBV'
elif args.reference == "19":
    namespace = 'hg19'
if 0:
    rdb.initialize(namespace=namespace)

filters = {'ds.index.ProjectID': 'FACE',
           'ds.index.qc_filter_pass': True}

index_keys = [u'ds.index.qc_filter_pass',
              u'ds.index',
              u'ds.index.sample_name',
              u'ds.index.client_subject_id',
              u'ds.index.VCF',
              u'ds.index.sample_key',
              u'ds.index.sample_client_name',
              u'ds.index.ProjectID',
              u'ds.index.BAM',
              u'ds.index.hli_wf']


# df_face_rdb = rdb.query(index_keys, filters=filters)

# keep_list = df_face_rdb[['ds.index.sample_key', 'ds.index.sample_key']].values
# np.savetxt(args.fn_keep, keep_list, fmt="%s", delimiter=" ")
# print "done after %.3fs\n" % (time.time()-t0)
if 0:
    print "running plink to get list of SNPs tha pass filters specified"
    t0 = time.time()
    plink_filter = "%s --bfile %s --keep %s --maf %f --hwe %f --geno %f --make-bed --out %s" % (args.plink_path, basefilename, args.fn_keep, args.maf, args.hwe, args.geno, args.bfile_filtered)
    print plink_filter
    plink_filter_call = [args.plink_path, "--bfile", basefilename, "--keep", args.fn_keep, "--maf", str(args.maf), "--hwe", str(args.hwe), "--geno", str(args.geno), "--make-bed", "--out", args.bfile_filtered]
if 0:#args.recompute or not os.path.exists(args.bfile_filtered+".bim"):
        subprocess.check_call(plink_filter_call)
if 0:
	plink_keep_call = [args.plink_path, "--bfile", args.bfile, "--keep", args.fn_keep, "--make-bed", "--out", args.bfile_filtered+".inter"]
	if not os.path.exists(args.bfile_filtered+".inter"+".bim"):
	        subprocess.check_call(plink_keep_call)
	plink_filter_call = [args.plink_path, "--bfile", args.bfile_filtered+".inter", "--maf", str(args.maf), "--hwe", str(args.hwe), "--geno", str(args.geno), "--make-bed", "--out", args.bfile_filtered]
	if not os.path.exists(args.bfile_filtered+".bim"):
	        subprocess.check_call(plink_filter_call)

if 0:
    plink_merge = "%s --bfile %s --bmerge %s --make-bed --out %s" % (args.plink_path, args.bfile_filtered, args.bmerge, args.out)
    print plink_merge
    plink_merge_call = [args.plink_path, "--bfile", args.bfile_filtered, "--bmerge", args.bmerge, "--make-bed", "--out", args.out]
    subprocess.check_call(plink_filter_call)

# print "done after %.2fs.\n" % (time.time() - t0)
# extract the SNPs

# load the filtered .bim file
if 0:
    print "opening bed files in pysnptools"
    t0 = time.time()

    snp_index = pd.read_csv(args.bfile_filtered+".bim", sep="\t", index_col=False, header=None, names=["chr", "cpra", "zero", "pos", "allele1", "allele2"])

basefilename = "/mnt/Jay_Flatley/PG0000100-BLD_hg19_riccardo.vcf.gz.orig.merged.face.geno0.1.maf0.05.hwe0.0001-merge-filtered"

bedlarge = Bed(basefilename)

if 0:
    bedlarge.df_snps = pd.DataFrame({"cpra":bedlarge.sid, "idx": np.arange(bedlarge.sid.shape[0])})

    idx_snps_to_use = pd.merge(left=snp_index, right=bedlarge.df_snps, how="left", on="cpra") 

    bedlarge.df_people = pd.DataFrame({"ds.index.sample_key": bedlarge.iid[:,0], "idx": np.arange(bedlarge.iid.shape[0])})

    idx_people_to_use = pd.merge(left=df_face_rdb, right=bedlarge.df_people, how="left", on="ds.index.sample_key")

    snp_reader_train = bedlarge[idx_people_to_use["idx"].values, idx_snps_to_use["idx"].values]
    snp_reader_test = bedsmall[:,idx_snps_to_use["idx"].values]
# extract the SNPs from the large file
# print "done after %.2fs.\n" % (time.time() - t0)


# extract the SNPs from the small file
covariates_train = None
covariates_test = None

print "creating SNP kernel and PCA"
t0 = time.time()

pcafile = os.path.join(args.out, "pca_") 
if True or args.recompute or not os.path.exists(pcafile + ".pickle"):
	kernel = LinearSnpKernel(beta_shape=0.8)
	K_train = kernel.fit_transform(bedlarge, blocksize=10000)

	# K_test = kernel.transform(snp_reader=snp_reader_test, blocksize=10000)

	kpca = KernelPCA()
	pc_k_train = kpca.fit_transform(K=K_train, covariates=covariates_train)
	
	kpca.to_pickle(pcafile)
else:
	kpca = KernelPCA.from_pickle(pcafile)
	pc_k_train = kpca.transform(K=K_train, covariates=covariates_train)

# pc_k_test = kpca.transform(K=K_test, covariates=covariates_test)

print "done after %.2fs.\n" % (time.time() - t0)


print "string final PCA as .csv file"
t0 = time.time()

pca_keys = ["pc.%i" % i for i in xrange(1, pc_k_train.shape[1]+1)]

df_pca_train = idx_people_to_use.copy()
for i, key in enumerate(pca_keys):
	df_pca_train[key] = pc_k_train[:,i]

df_pca_test = pd.DataFrame(index = bedlarge.iid[:,0], snpreaderdata=pc_k_train, columns=pca_keys)
df_pca_train.to_csv(pcafile + "_train_.csv")
# df_pca_test.to_csv(pcafile + "_test.csv")

print "done after %.2fs.\n" % (time.time() - t0)
print "---------------------------\ntotal time: %.2fs.\n" % (time.time() - t00)
