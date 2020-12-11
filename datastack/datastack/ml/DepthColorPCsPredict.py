# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 11:00:24 2015

Copyright (C) 2015 Human Longevity Inc. Written by Alena.

Data utilities to support Depth and Color PC prediction.
"""

import sys,os,os.path
#sys.path.append('/home/ubuntu/cerebro/datastack')
import datastack.dbs.rdb
from datastack.dbs import rdb as rosetta
import datastack.ml.baselines as baselines
import datastack.ml.baseregress as baseregress
import datastack.ml.cross_validation as cross_validation
import datastack.utilities.gui as gui
from datastack.tools import vdbwrapper
from datastack.dbs import vdb as vdbclass

from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import operator
import json
from  scipy.spatial.distance import cosine
import collections

vdbclass = reload(vdbclass)
gui = reload(gui)
vdbwrapper = reload(vdbwrapper)
cross_validation = reload(cross_validation)

# utility functions
class JsonConverter():
    '''
    parse json
    '''
    @classmethod
    def split_array(self, s, title="PC"):
        l = json.loads(s)
        t = [title+str(i+1) for i in range(len(l))]
        return dict(zip(t, l))

# selected 10 closed for approximation for the face phenotypes
def identify(_id, t_y, p_y):
    '''
    identify most similar person (cosine distance) in a data set and get the rank
    '''
    mse_dict = {}
    for i in t_y.index:
        mse = cosine(p_y[_id],t_y[i])
        mse_dict[i] = mse

    sorted_x       = sorted(mse_dict.items(), key=operator.itemgetter(1))
    rank           = [ii[0] for ii in sorted_x].index(_id)
    return rank

def select10_rate(t_y, p_y):
    '''
    avg fraction of time for select10
    '''
    select_10 = []
    for i in p_y.index:
        rank = identify(i, t_y, p_y)
        select_10.append(pow((1-float(rank-1)/float(len(p_y))),9))
    return np.mean(select_10)

def cleanMult (rs, snplist):
    """
    Sometimes the rsid column contains multiple numbers separated by
    commas. We take the first which is in snplist.
    """
    rslist = rs.split(',')
    if len(rslist) < 2:
        return rs

    users = next(rsid for rsid in rslist if 'rs'+rsid in snplist)
    if users:
        return users

    # Fallback

def renameColumns (df, ndf, snplist):
    """
    For each column, find the row in the query results ndf. Then rename the
    column according to the rsid in the query.
    """
    rename = {}
    cols = list(df.columns)
    for c in cols:
        row = ndf.loc[ndf['var'] == c]
        rsid = row.iloc[0]['rsid'] # this is without rs prefix :(

        # Could be a comma separated list of numbers :(
        for rs in rsid.split(','):
            if 'rs' + rs in snplist: # only rename colname to rsid if we queried by rsid
                rename[c] = 'rs' + cleanMult(rs, snplist)
    df = df.rename(columns=rename)
    return df


class DepthColorPCs (object):
    """
    Class to encapsulate depth and color PCs predictions.
    """
    age = "dynamic.FACE.pheno.v1.age"
    bmi = "dynamic.FACE.pheno.v1.bmi"
    gender = ["dynamic.FACE.pheno.v1.female","dynamic.FACE.pheno.v1.male"]
    snps = ['rs12913832','rs1545397', 'rs16891982', 'rs1426654', 'rs885479', 'rs6119471', 'rs12203592']

    def __init__ (self, version ="v2",rdb_version='1447895375', rdb_namespace = 'hg19', your_pcs_depth="", your_pcs_color=""):
        """
        Set up one big data frame containing all the features.
        It will have some empty cells, which we will need to deal with later.
        """
        # for ec2 rosetta is at 172.31.22.29
        self.rdb = rosetta.RosettaDBMongo(host="rosetta.hli.io",port=27017)
        self.rdb.initialize(version=rdb_version, namespace=rdb_namespace)

        self.your_pcs_depth = your_pcs_depth
        self.your_pcs_color = your_pcs_color
        self.data = self.load_data(version=version)


    def load_data(self,rdb_version='1447895375', rdb_namespace = 'hg38_noEBV',version="v2"):
        """
        load data from RosettaDB and VariantDB
        """

        #colorpc_keys   = self.your_rdb.find_keys(self.your_pcs_depth)  #'dynamic.FACE.face.{}_visit1.ColorPC'.format(version))
        #depthpc_keys   = self.your_rdb.find_keys(self.your_pcs_color) #'dynamic.FACE.face.{}_visit1.DepthPC'.format(version))

        gen_keys    = self.rdb.find_keys("genomics.kinship.pc.*",regex=True)

        data = self.rdb.query(filters = {"ds.index.ProjectID": "FACE"}, #dynamic.FACE_P.age.v2.value
            keys    = ["ds.index.sample_key","lab.ClientSubjectID"]+
                           [self.bmi] +
                           [self.age] +
                           self.gender+
                           gen_keys +
                           [],
                with_nans=True)


        data["dynamic.FACE.pheno.v1.male"]=data.loc[:,"dynamic.FACE.pheno.v1.male"].replace([1],[-1])
        data["gender"] =data.apply(lambda row: row['dynamic.FACE.pheno.v1.female']+row['dynamic.FACE.pheno.v1.male'], axis=1)

        # currently could only find local PCs here
        rdb = rosetta.RosettaDBMongo(host="rosetta.hli.io",port=27017) #
        rdb.initialize(version=rdb_version, namespace=rdb_namespace)
        depthpc_keys   = rdb.find_keys(self.your_pcs_depth)  #'dynamic.FACE.face.{}_visit1.ColorPC'.format(version))
        colorpc_keys   = rdb.find_keys(self.your_pcs_color) #'dynamic.FACE.face.{}_visit1.DepthPC'.format(version))

        data_pcs = rdb.query(filters = {"lab.ProjectID": "FACE"},
               keys    = ["lab.ClientSubjectID"]+
                           colorpc_keys +
                           depthpc_keys +
                           [],
                with_nans=False, #True
              )
        DepthPCs = data_pcs[self.your_pcs_depth].apply(lambda s: pd.Series(JsonConverter.split_array(s, title="dynamic.FACE.face.{}_visit1.DepthPC.".format(version))))
        ColorPCs = data_pcs[self.your_pcs_color].apply(lambda s: pd.Series(JsonConverter.split_array(s, title="dynamic.FACE.face.{}_visit1.ColorPC.".format(version))))
        data_pcs=data_pcs.join(ColorPCs)
        data_pcs=data_pcs.join(DepthPCs)
        temp=pd.merge(data,data_pcs, on='lab.ClientSubjectID')
        data = temp.dropna(subset =['dynamic.FACE.pheno.v1.bmi','dynamic.FACE.pheno.v1.age','dynamic.FACE.pheno.v1.female'])

        #Remove error samples
        error_samples   = ["15-0153", "15-0041", "15-0047", "15-0275", \
                   "15-1072", "15-0032", "15-0156", "15-0179", \
                   "15-0107", "15-0191", "15-0787",  \
                   "15-0656", "15-0346", "15-0959", "15-0444"]

        idx = [j for j in data.index if str(data.ix[j,'lab.ClientSubjectID']) in error_samples]
        idx_incl = set(data.index)-set(idx)
        # clean the data set, do not include bad samples
        data =data.ix[idx_incl,:]

        # get data from VariantDB for 7 SNPs for prediction of Color PCs
        vdb = vdbclass.HpcVarianceDB(host="variantdb.hli.io", port="8080")
        vdb.initialize(version="HG19_IsaacVariantCaller_AllSamples_PASS_MAF_001_gVCF", separateCPRAs=False)

        filters = {"lab.ProjectID": "FACE"}
        keys    = ["ds.index.sample_key"]
        query      = self.rdb.query(keys, filters=filters)
        sample_list = list(query["ds.index.sample_key"].values)
        rsids = self.snps
        filters ={}
        filters['samples']   = sample_list
        filters['variants']  = rsids
        keys_vdb            = ["variantcall"]

        df                   = vdb.query(filters, keys_vdb)
        #print sample_list
        #print len(sample_list)
        #print len(df.index)
        #wdb wrapper is currently broken - get some of its functionality here
        # use Mode for missing
        df = df.applymap(lambda x : {0: 0, 1: 1, 2: 2, -9: None}[x])
        df = df.fillna(df.mode().iloc[0])

        # Replace column names with snp labels
        filters = {}
        filters['variants'] = list(df.columns)
        ndf = vdb.query(filters, ['variants'])
        ndf['var'] = ndf.index
        df = renameColumns(df, ndf, rsids)

        # sample key is the index of the df frame
        df.index = sample_list
        df["ds.index.sample_key"] = df.index

        #vdba = vdbwrapper.VdbWrapper(self.rdb, False)
        #df = vdba.faceVars(rsids, missing=vdbwrapper.MissingOptions.mode)
        data=pd.merge(data,df, on='ds.index.sample_key')

        #print data.head(5)
        depthPCs = ["dynamic.FACE.face.{}_visit1.DepthPC.".format(version)+str(x) for x in range(1,1001,1)]
        colorPCs = ["dynamic.FACE.face.{}_visit1.ColorPC.".format(version)+str(x) for x in range(1,1001,1)]

        covariates = gen_keys+["dynamic.FACE.pheno.v1.bmi","dynamic.FACE.pheno.v1.age","gender",'lab.ClientSubjectID','ds.index.sample_key']+depthPCs+colorPCs+rsids
        return data.loc[:,covariates]


    def predictDepthPCs(self, n=50, covPCs=None, version ="v2", multi = True, run_full_only = True, with_holdout = True, run_keys=None):
        '''
        predict depth PCs using actual bmi, actuale age, actual gender, covPCs (can be global PCs, local PCs
        or combination of both, n - number of Depth PCs to predict, multi - run prediction through
        baseregression framework as multidimensional target)
        '''

        target_dict=collections.OrderedDict()
        for i in xrange(1,n+1):
            target_dict['pc{}'.format(i)]='dynamic.FACE.face.{}_visit1.DepthPC.{}'.format(version,i)

        base = baseregress.BaseRegress(target_dict,multi=multi)
        base.covariates['BMI'] = [self.bmi]
        base.addData(self.data)
        params = {'alpha': [ 0.0001, 0.001, 0.01,  0.1, 1, 10, 1000]}
        est1 = {'est' : linear_model.Ridge(), 'params' : params}
        est2 = {'est' : linear_model.Lasso(), 'params' : params}

        fmt = {}
        fmt['MAE'] = 3
        fmt['MSE'] = 3
        fmt['R2'] =3
        fmt['SELECT10']=3
        # drop some error columns
        base.dropCol('MSE')
        base.dropCol('MAE')

        base.covariates['Age'] = [self.age]
        base.covariates['Ethnicity'] = covPCs
        base.covariates['Gender']=["gender"]
        base.estimatorsToRun['Ethnicity']=[est1]
        base.estimatorsToRun['AGE'] = [est1]
        base.estimatorsToRun['AGE + BMI'] = [est1]
        if with_holdout:
            base.run(with_aggregate_covariates=True,kfargs={'n_folds_holdout': 0},run_full_only=run_full_only,run_keys=run_keys)
        else:
            base.run(with_aggregate_covariates=True,run_full_only=run_full_only,run_keys=run_keys)
        base.display(fmtdict=fmt)
        self.baseDepthPCs = base
        return base

    def predictColorPCs(self, n=50, covPCs=None, snps = ['rs12913832','rs1545397', 'rs16891982', 'rs1426654', 'rs885479', 'rs6119471', 'rs12203592'], version ="v2", multi = True, run_full_only=True, with_holdout=True, run_keys=None):
        '''
        predict color PCs using actuale age, actual gender, 7 SNPs, covPCs (can be global PCs, local PCs
        or combination of both, n - number of Color PCs to predict, multi - run prediction through
        baseregression framework as multidimensional target)
        '''
        target_dict=collections.OrderedDict()
        for i in xrange(1,n+1):
            target_dict['pc{}'.format(i)]='dynamic.FACE.face.{}_visit1.ColorPC.{}'.format(version,i)

        base = baseregress.BaseRegress(target_dict,multi=multi)
        base.covariates['SNPs'] = snps
        base.addData(self.data)
        params = {'alpha': [ 0.0001, 0.001, 0.01,  0.1, 1, 10, 1000]}
        est1 = {'est' : linear_model.Ridge(), 'params' : params}
        est2 = {'est' : linear_model.Lasso(), 'params' : params}

        fmt = {}
        fmt['MAE'] = 3
        fmt['MSE'] = 3
        fmt['R2'] =3
        fmt['SELECT10']=3
        # drop some error columns
        base.dropCol('MSE')
        base.dropCol('MAE')

        base.covariates['Age'] = [self.age]
        base.covariates['Gender']=["gender"]
        base.covariates['Ethnicity'] = covPCs
        base.estimatorsToRun['Ethnicity']=[est1]
        base.estimatorsToRun['AGE'] = [est1]
        base.estimatorsToRun['AGE + SNPs'] = [est1]
        if with_holdout:
            base.run(with_aggregate_covariates=True,kfargs={'n_folds_holdout': 0},run_full_only=run_full_only,run_keys=run_keys)
        else:
            base.run(with_aggregate_covariates=True,run_full_only=run_full_only, run_keys=run_keys)
        base.display(fmtdict=fmt)
        self.baseColorPCs = base
        return base



if __name__ == "__main__":
    m=1000
    genPCs = ["genomics.kinship.pc."+str(x) for x in xrange(1,m+1,1)]
    modelPCs = DepthColorPCs()
    baseColor = modelPCs.predictColorPCs(covPCs=genPCs)
    baseDepth = modelPCs.predictDepthPCs(covPCs=genPCs)
    modelColorCV=baseColor.cv[('AGE + SNPs', 'multi', 0)]
    modelDepthCV=baseDepth.cv[('AGE + BMI', 'multi', 0)]
