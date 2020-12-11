#FACE specific base regression functions

#import datastack.tools.facephenotyper.ComputeEmbedding as embed
from datastack.dbs import rdb as rosetta
import datastack.settings as settings
from datastack.ml.algorithms.pca_covariates import CovariatesPCA

import pandas as pd
import numpy as np
import json
import os

def write_csv_files(sample_keys_df,file_name='./sample_keys.csv',header=None,append=False):
    '''
    write a data frame line by line into csv either append or start a new file based on append flag
    '''
    flag = ''

    if append:
        flag = 'a'
        header = False
    else:
        flag = 'w'
        if header is None:
            header = False
    with open(file_name, flag) as f_:
        sample_keys_df.to_csv(f_,header=header,index=False)


class JsonConverter():
    '''
    parse json
    '''
    @classmethod
    def split_array(self, s, title="PC"):
        l = json.loads(s)
        t = [title+str(i+1) for i in range(len(l))]
        return dict(zip(t, l))


def symmetrizeFace(img,position):
    '''
    Take a face image and make it symmetric by averaging the left and right hand sides
    img: A numpy array of size [3, 256, 256]
    position: True if this image corresponds to the position, False if it is color
    returns the symmetrized image
    '''
    symmetricImage = img.copy()
    fw=256
    fh=256
    for y in range(fh):
        for x in range(fw):
            if position:
                symmetricImage[0,x,y] = (img[0,x,y] - img[0,fw-1-x,y]) * 0.5
            else:
                symmetricImage[0,x,y] = (img[0,x,y] + img[0,fw-1-x,y]) * 0.5
            symmetricImage[1,x,y] = (img[1,x,y] + img[1,fw-1-x,y]) * 0.5
            symmetricImage[2,x,y] = (img[2,x,y] + img[2,fw-1-x,y]) * 0.5
    return symmetricImage

#order of the covariates [float(gender), float(age), float(bmi), float(afr), float(amr), float(csa), float(eas), float(eur)]
def add_covariates(dw,fold,covariates=None,VERSION=None,TYPE=None,COVARIATES=None,HOLDOUT=False,directory='/data/notebooks'):
    '''
    Add back covariate for regressed out covariates models for the face
    '''
    import time
    
    np_list = []
    print "Running Version {} Type {} Covariates {}".format(VERSION,TYPE,COVARIATES)
    for fld in range(1,11,1) :
        foldStartTime     = time.clock()
        ending = ''
        if VERSION is None or VERSION == 'V8NoCov':
            VERSION = 'V8NoCov'
            ending = 'NoCov'
        else:
            if VERSION == 'V8Cov':
                ending = "GAB"
            else:
                if VERSION == 'V11Cov':
                    ending = 'GAB_Ethnicity'
        if TYPE is None:
            TYPE = 'Position'
        if HOLDOUT:
            file_name = os.path.join(str(directory),"FaceEmbeddings_Holdout/Fold_")+str(fld)+"/"+ending+"/"+TYPE+"_Model"
        else:
            file_name = os.path.join(str(directory),"FaceEmbeddings_NoHoldout/Fold_")+str(fld)+"/"+ending+"/"+TYPE+"_Model"

        
        print "Using {}".format(file_name)
        model_v = CovariatesPCA.from_pickle(file_base_name=file_name,mmap_mode='r')
        convStartTime = time.clock()
        
        
        emb = np.array(dw.loc[list(np.array(fold['fold']==fld)),:].values,dtype=np.float64)
        if covariates is not None:
            cov = np.array(covariates.loc[list(np.array(fold['fold']==fld)),:].values,dtype=np.float64)
            #print "shape of emb {} shape of cov {}, type emb {} type cov {}".format(emb.shape,cov.shape,emb.dtype,cov.dtype)
            if np.sum(covariates.loc[list(np.array(fold['fold']==fld)),:].isnull().values)>0:
                print "cov any nan {}".format(np.sum(covariates.loc[list(np.array(fold['fold']==fld)),:].isnull().values))
        else:
            cov = None
        if np.sum(dw.loc[list(np.array(fold['fold']==fld)),:].isnull().values)>0:
            print "emb any nan {}".format(np.sum(dw.loc[list(np.array(fold['fold']==fld)),:].isnull().values))


        if VERSION=='V8NoCov':
            np_list.append(model_v.inverse_transform(emb))
        else:
            np_list.append(model_v.inverse_transform(emb,covariates=cov))
        #print "row = {}, time = {} seconds".format(fld, time.clock() - foldStartTime)
    
    femb = np.vstack(np_list)
    #print "vstack done time = {} seconds".format(time.clock() - foldStartTime)
    return femb

class _Face (object):
    """
    class that initialize specific version of face, and allows us to substitute SVD values per fold

    """

    def __init__ (self,
                  VERSION = None,
                  TYPE = None,
                  COVARIATES = None,
                  PREDICTED_COVARIATES = None,
                  HOLDOUT =False,
                  *args,
                  **kwargs):
        """
        Prepare to handle calls to both cv and eval objects.
        """

        self.VERSION = VERSION
        self.TYPE =TYPE
        self.COVARIATES = COVARIATES
        self.PREDICTED_COVARIATES = PREDICTED_COVARIATES
        self.HOLDOUT = HOLDOUT
        self.args = args
        self.kwargs = kwargs

    def _getSVDValuesPerFoldFace(self,idx_cv,ydim,directory="/data/notebooks"):
        '''
        If doing face prediction over-write y-values for face SVD (position and color) from face embedding files generated per fold
        
        Parameters:
        idx_cv - fold number
        ydim - number of columns in y
        directory - directory where files reside

        Returns:
        data frame with all the necessary info to over-write y's and provide covariates
        model_pcs - names of the columns where y PCs reside
        '''
        # only do this if we are doing face run
        
        res_frame = pd.DataFrame()
        model_pcs=[]
        if self.VERSION is not None and self.TYPE is not None:
            ending=''
            vers=''
            ef=''
            pc_vers = ''

            if self.VERSION=='V8Cov':
                ending = 'GAB'
                vers='v8'
                ef='Cov'
            else:
                if self.VERSION=='V11Cov':
                    ending = 'GAB_Ethnicity'
                    vers = 'v11'
                    ef='Cov'
                else:
                    if self.VERSION=='V8NoCov':
                        ending = 'NoCov'
                        vers = 'v8'
                        ef='NoCov'
            if self.TYPE == 'Position':
                pc_vers='Depth'
            else:
                if self.TYPE == 'Color':
                    pc_vers = 'Color'

            pc_version = "dynamic.FACE.3dface."+vers+"_"+self.TYPE+"_Face_"+ef+".visit1.sessionId"
            if self.HOLDOUT:
                file_name = os.path.join(str(directory),"FaceEmbeddings_Holdout/Fold_")+str(idx_cv)+"/"+ending+"/"+self.TYPE+"_Embedding.csv"
            else:
                file_name = os.path.join(str(directory),"FaceEmbeddings_NoHoldout/Fold_")+str(idx_cv)+"/"+ending+"/"+self.TYPE+"_Embedding.csv"
            
            remap=pd.read_csv(file_name,header=0,sep=',')
            print "reading in {}".format(file_name)
            remap=remap.drop(remap.columns[0], axis=1)
            remap.columns = ['jsonData','sessionId','lab.ClientSubjectID']
            remap['sessionId'] = remap['sessionId'].astype(int).astype(str)
            remap['subject_id'] = remap[["lab.ClientSubjectID", 'sessionId']].apply(lambda x: '_'.join(x), axis=1)
            
            # compute embedding doesn't have ds.index.smple_name need to get this from Rosetta, along with values for the covariates
            rdb_version_hg38=settings.ROSETTA_VERSION 
            rdb_namespace_hg38 = settings.ROSETTA_NAMESPACE
            rdb_hg38 = rosetta.RosettaDBMongo(host=settings.ROSETTA_URL,port=27017)
            rdb_hg38.initialize(version=rdb_version_hg38, namespace=rdb_namespace_hg38)

            cov = []
            if self.COVARIATES is not None:
                cov=list(self.COVARIATES)
            if self.PREDICTED_COVARIATES is not None:
                cov.extend(self.PREDICTED_COVARIATES)
                cov = list(set(cov))

            map_id = rdb_hg38.query(filters = {"lab.ProjectID": "FACE"},
                                            keys    = ["ds.index.sample_key","lab.ClientSubjectID","ds.index.sample_name"]+
                                        cov+
                                        [pc_version]+
                                        [],
                                        with_nans=True,
                    )
            map_id = map_id.dropna(subset=[pc_version])
            map_id[pc_version] = map_id[pc_version].astype(int).astype(str)
            map_id['subject_id'] = map_id[["lab.ClientSubjectID", pc_version]].apply(lambda x: '_'.join(x), axis=1)
            if self.COVARIATES is not None and "pheno.gender" in self.COVARIATES:
                map_id["pheno.gender"] = map_id["pheno.gender"].replace(["Male","Female",np.nan],[0,1,0])
            
            if self.PREDICTED_COVARIATES is not None and "dynamic.FACE_P.gender.v1.value" in self.PREDICTED_COVARIATES:
                map_id["dynamic.FACE_P.gender.v1.value"]=map_id["dynamic.FACE_P.gender.v1.value"].replace(["Male","Female",np.nan],[0,1,0])
            
            map_id.ix[:,cov] = map_id.loc[:,cov].apply(lambda x: x.fillna(x.mean()),axis=0)
            res_frame=pd.merge(remap, map_id, on='subject_id')
            PCs = res_frame["jsonData"].apply(lambda s: pd.Series(JsonConverter.split_array(s, title="dynamic.FACE.face."+vers+"_visit1."+pc_vers+"PC.")))
            res_frame=res_frame.join(PCs)
            res_frame.index = res_frame['ds.index.sample_name']
            model_pcs = ["dynamic.FACE.face."+vers+"_visit1."+pc_vers+"PC."+str(x) for x in range(1,ydim+1,1)]
            res_frame = res_frame.drop_duplicates(cols='ds.index.sample_name', take_last=True)
            
            res_frame['fold']=idx_cv
            

            #IF FOLDS CHANGE RUN THIS: and uncomment import
            #sample_key_training = res_frame.loc[self.ds_sample_key_training,'ds.index.sample_key'].values
            #sample_key_test = res_frame.loc[self.ds_sample_key_test,'ds.index.sample_key'].values
            #write_csv_files(pd.DataFrame(sample_key_training),file_name='./sample_key_train_fold_'+str(idx_cv)+'.csv',header=['sample_key_train'],append=False)
            #write_csv_files(pd.DataFrame(sample_key_test),file_name='./sample_key_test_fold_'+str(idx_cv)+'.csv',header=['sample_key_test'],append=False)
                    
            #embed.computeEmbeddingForFold('./sample_key_train_fold_'+str(idx_cv)+'.csv', self.COVARIATES, self.TYPE, str(directory)+"/FaceEmbeddings/Fold_"+str(idx_cv)+"/"+ending+"/"+self.TYPE)
            #embed.computeEmbeddingForFold('./sample_key_train_fold_'+str(idx_cv)+'.csv', ['pheno.gender','pheno.age','pheno.bmi'], self.TYPE, str(directory)+"/FaceEmbeddings/Fold_"+str(idx_cv)+"/GAB/"+self.TYPE)
            #embed.computeEmbeddingForFold('./sample_key_train_fold_'+str(idx_cv)+'.csv', [], self.TYPE, "/data/notebooks/FaceEmbeddings/Fold_"+str(idx_cv)+"/NoCov/"+self.TYPE)
            
        return res_frame,model_pcs 


    def _getSVDValuesFace(self,ydim,directory="/data/notebooks"):
        
        '''
        If doing face prediction over-write y-values for face SVD (position and color) from face embedding files generated with/without holdout depending on HOLDOUT flag

        Parameters:
        
        directory - directory where files reside
        ydim - number of columns in y

        Returns:
        data frame with all the necessary info to over-write y's
        model_pcs - names of the columns where y pcs reside 
        '''
        
        # only do this if we are doing face run
        res_frame=pd.DataFrame()
        model_pcs = []
        
        if self.VERSION is not None and self.TYPE is not None:
            ending=''
            vers=''
            pc_vers =''
            ef=''
            if self.VERSION=='V8Cov':
                ending = 'GAB'
                vers='v8'
                ef='Cov'
            else:
                if self.VERSION=='V11Cov':
                    ending = 'GAB_Ethnicity'
                    vers = 'v11'
                    ef='Cov'
                else:
                    if self.VERSION=='V8NoCov':
                        ending = 'NoCov'
                        vers = 'v8'
                        ef='NoCov'
            
            if self.TYPE == 'Position':
                pc_vers='Depth'
            else:
                if self.TYPE == 'Color':
                    pc_vers = 'Color'

            pc_version = "dynamic.FACE.3dface."+vers+"_"+self.TYPE+"_Face_"+ef+".visit1.sessionId"
            if self.HOLDOUT:
                file_name = os.path.join(str(directory),"FaceEmbeddingsAll_Holdout/")+ending+"/"+self.TYPE+"_Embedding.csv"
            else:
                file_name = os.path.join(str(directory),"FaceEmbeddingsAll_NoHoldout/")+ending+"/"+self.TYPE+"_Embedding.csv"
            
            remap=pd.read_csv(file_name,header=0,sep=',')
            print "Finally,reading in {}".format(file_name)
            remap=remap.drop(remap.columns[0], axis=1)
            remap.columns = ['jsonData','sessionId','lab.ClientSubjectID']
            remap['sessionId'] = remap['sessionId'].astype(int).astype(str)
            remap['subject_id'] = remap[["lab.ClientSubjectID", 'sessionId']].apply(lambda x: '_'.join(x), axis=1)
            
            
            rdb_version_hg38 = settings.ROSETTA_VERSION
            rdb_namespace_hg38 = settings.ROSETTA_NAMESPACE
            rdb_hg38 = rosetta.RosettaDBMongo(host = settings.ROSETTA_URL,port=27017)
            rdb_hg38.initialize(version=rdb_version_hg38, namespace=rdb_namespace_hg38)

            
            map_id = rdb_hg38.query(filters = {"lab.ProjectID": "FACE"},
                                            keys    = ["ds.index.sample_key","lab.ClientSubjectID","ds.index.sample_name"]+
                                        [pc_version]+
                                        [],
                                        with_nans=False,
                                                            )
            map_id[pc_version] = map_id[pc_version].astype(int).astype(str)
            map_id['subject_id'] = map_id[["lab.ClientSubjectID", pc_version]].apply(lambda x: '_'.join(x), axis=1)
            
            res_frame=pd.merge(remap, map_id, on='subject_id')
            PCs = res_frame["jsonData"].apply(lambda s: pd.Series(JsonConverter.split_array(s, title="dynamic.FACE.face."+vers+"_visit1."+pc_vers+"PC.")))
            res_frame=res_frame.join(PCs)
            res_frame.index = res_frame['ds.index.sample_name']
            model_pcs = ["dynamic.FACE.face."+vers+"_visit1."+pc_vers+"PC."+str(x) for x in range(1,ydim+1,1)]
            res_frame = res_frame.drop_duplicates(cols='ds.index.sample_name', take_last=True)

        return res_frame,model_pcs

