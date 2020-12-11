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

class DepthColorPCs (object):
    """
    Class to encapsulate depth and color PCs predictions.
    """
    age = "dynamic.FACE.pheno.v1.age"
    bmi = "dynamic.FACE.pheno.v1.bmi"
    gender = ["dynamic.FACE.pheno.v1.female","dynamic.FACE.pheno.v1.male"]
    snps = ['rs12913832','rs1545397', 'rs16891982', 'rs1426654', 'rs885479', 'rs6119471', 'rs12203592']

    def __init__ (self, version ="v2",rdb_version='1447895375', data=None):
        """
        Set up one big data frame containing all the features.
        It will have some empty cells, which we will need to deal with later.
        We allow the import of a data frame to support regression testing.
        """
        if data is None:
            self.rdb = self._getRdb()
            self.rdb.initialize(version=rdb_version, namespace='hg19')
    
            self.data = self.load_data(version=version)
        else:
            self.data = data
    
    def _getRdb (self):
        # for ec2 rosetta is at 172.31.22.29
        return rosetta.RosettaDBMongo(host="172.31.22.29",port=27017)
        # return rosetta.RosettaDBMongo()
    
    def _getVdb (self):
        # ec2 version
        vdb = vdbclass.HpcVarianceDB(host="172.31.47.154", port="8080")
        # vdb = vdbclass.HpcVarianceDB()
        return vdb

    def load_data(self,version="v2"):
        """
        load data from RosettaDB and VariantDB
        """

        colorpc_keys   = self.rdb.find_keys('dynamic.FACE.face.{}_visit1.ColorPC'.format(version))
        depthpc_keys   = self.rdb.find_keys('dynamic.FACE.face.{}_visit1.DepthPC'.format(version))

        gen_keys    = self.rdb.find_keys("genomics.kinship.pc.*",regex=True)

        data = self.rdb.query(filters = {"ds.index.ProjectID": "FACE"}, #dynamic.FACE_P.age.v2.value
            keys    = ["ds.index.sample_key","lab.ClientSubjectID"]+
                           [self.bmi] +
                           [self.age] +
                           self.gender+
                           gen_keys +
                           colorpc_keys +
                           depthpc_keys,
                with_nans=True)

        data = data.dropna(subset =['dynamic.FACE.face.{}_visit1.DepthPC'.format(version)])
        DepthPCs = data["dynamic.FACE.face.{}_visit1.DepthPC".format(version)].apply(lambda s: pd.Series(JsonConverter.split_array(s, title="dynamic.FACE.face.{}_visit1.DepthPC.".format(version))))
        ColorPCs = data["dynamic.FACE.face.{}_visit1.ColorPC".format(version)].apply(lambda s: pd.Series(JsonConverter.split_array(s, title="dynamic.FACE.face.{}_visit1.ColorPC.".format(version))))
        data=data.join(ColorPCs)
        data=data.join(DepthPCs)
        data["dynamic.FACE.pheno.v1.male"]=data.loc[:,"dynamic.FACE.pheno.v1.male"].replace([1],[-1])
        data["gender"] =data.apply(lambda row: row['dynamic.FACE.pheno.v1.female']+row['dynamic.FACE.pheno.v1.male'], axis=1)

        # currently could only find local PCs here
        rdb = self._getRdb()
        rdb.initialize(namespace="hg38_noEBV")
        lgen_keys   = rdb.find_keys("dynamic.genomepclocal10000.pc1")
        data_local_pcs = rdb.query(filters = {"lab.ProjectID": "FACE"},
               keys    = ["lab.ClientSubjectID"]+
                           lgen_keys +
                           [],
                with_nans=True
              )
        temp=pd.merge(data,data_local_pcs, on='lab.ClientSubjectID')
        data = temp.dropna(subset =['dynamic.FACE.face.{}_visit1.DepthPC'.format(version),'dynamic.FACE.pheno.v1.bmi','dynamic.FACE.pheno.v1.age','dynamic.FACE.pheno.v1.female'])

        #Remove error samples
        error_samples   = ["15-0153", "15-0041", "15-0047", "15-0275", \
                   "15-1072", "15-0032", "15-0156", "15-0179", \
                   "15-0107", "15-0191", "15-0787", "15-0107", \
                   "15-0656", "15-0346", "15-0959", "15-0444", "15-0444"]

        idx = [j for j in data.index if str(data.ix[j,'lab.ClientSubjectID']) in error_samples]
        idx_incl = set(data.index)-set(idx)
        # clean the data set, do not include bad samples
        data =data.ix[idx_incl,:]

        # get data from VariantDB for 7 SNPs for prediction of Color PCs
        vdb = self._getVdb()
        vdb.initialize(version="HG19_IsaacVariantCaller_AllSamples_PASS_MAF_001_gVCF")
        filters = {"qc.ProjectID": "FACE"}
        keys    = ["ds.index.sample_key"]
        query      = self.rdb.query(keys, filters=filters)
        sample_list = list(query["ds.index.sample_key"].values)
        rsids = self.snps

        vdba = vdbwrapper.VdbWrapper(self.rdb, False)
        df = vdba.faceVars(rsids, missing=vdbwrapper.MissingOptions.mode)
        data=pd.merge(data,df, on='ds.index.sample_key')

        depthPCs = ["dynamic.FACE.face.v2_visit1.DepthPC."+str(x) for x in range(1,1001,1)]
        colorPCs = ["dynamic.FACE.face.v2_visit1.ColorPC."+str(x) for x in range(1,1001,1)]

        covariates = gen_keys+lgen_keys+["dynamic.FACE.pheno.v1.bmi","dynamic.FACE.pheno.v1.age","gender",'lab.ClientSubjectID','ds.index.sample_key']+depthPCs+colorPCs+rsids
        return data.loc[:,covariates]


    def predictDepthPCs(self, n=50, covPCs=None, version ="v2", multi = True):
        '''
        predict depth PCs using actual bmi, actuale age, actual gender, covPCs (can be global PCs, local PCs
        or combination of both, n - number of Depth PCs to predict, multi - run prediction through
        baseregression framework as multidimensional target)
        '''
        target_dict={}
        for i in xrange(1,n+1):
            target_dict['pc{}'.format(i)]='dynamic.FACE.face.{}_visit1.DepthPC.{}'.format(version,i)

        base = baseregress.BaseRegress(target_dict,multi=multi)
        base.covariates['BMI'] = [self.bmi]
        base.addData(self.data)
        params = {'alpha': [ 0.0001, 0.001, 0.01,  0.1, 1, 10, 1000]}
        est1 = {'est' : linear_model.Ridge(), 'params' : params}
        #est2 = {'est' : linear_model.Lasso(), 'params' : params}

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
        base.run(with_aggregate_covariates=True,kfargs={'n_folds_holdout': 0})
        base.display(fmtdict=fmt)
        self.baseDepthPCs = base
        return base

    def predictColorPCs(self, n=50, covPCs=None, snps = ['rs12913832','rs1545397', 'rs16891982', 'rs1426654', 'rs885479', 'rs6119471', 'rs12203592'], version ="v2", multi = True):
        '''
        predict color PCs using actuale age, actual gender, 7 SNPs, covPCs (can be global PCs, local PCs
        or combination of both, n - number of Color PCs to predict, multi - run prediction through
        baseregression framework as multidimensional target)
        '''
        target_dict={}
        for i in xrange(1,n+1):
            target_dict['pc{}'.format(i)]='dynamic.FACE.face.{}_visit1.ColorPC.{}'.format(version,i)

        base = baseregress.BaseRegress(target_dict,multi=multi)
        base.covariates['SNPs'] = snps
        base.addData(self.data)
        params = {'alpha': [ 0.0001, 0.001, 0.01,  0.1, 1, 10, 1000]}
        est1 = {'est' : linear_model.Ridge(), 'params' : params}
        #est2 = {'est' : linear_model.Lasso(), 'params' : params}

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
        base.run(with_aggregate_covariates=True,kfargs={'n_folds_holdout': 0})
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
