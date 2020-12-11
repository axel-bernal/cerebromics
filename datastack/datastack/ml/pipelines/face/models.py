from sknn import mlp
import numpy as np
import math
import operator
from datastack.ml.pipelines.face import settings
from  scipy.spatial.distance import cosine
import os
import pandas as pd
from sklearn import linear_model
from sklearn import neighbors
from sklearn import svm


class Model(object):

    STORE_M_BASE = "model"

    def __init__(self, *argv, **kwargs):
        self.model_class = argv[0]
        self.params      = kwargs
        self.model_name  = kwargs.pop("name", self.model_class.__name__)

    def fit(self, x, y):
        pass

    def predict(self, x):
        pass


# -------------------------------------
#   Model wrapper
# -------------------------------------

class ModelExecutor(object):

    def __init__(self, model):

        import os
        import uuid

        self.model      = model
        self.model_info = {}

        self.model_id   = str(uuid.uuid1())
        self.root_dir   = os.path.join(os.getcwd(), settings.MODEL_ROOT_PREFIX, "{}.{}".format(self.model.model_name, self.model_id))
        self.stored     = False

    def store(self, full=False):

        import json
        import os

        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)

        with open(os.path.join(self.root_dir, "info.json"), "w") as f:
            f.write(json.dumps(self.model_info))

        # Depth
        _depth_dir = os.path.join(self.root_dir, "depth")
        if not os.path.exists(_depth_dir):
            os.makedirs(_depth_dir)

        if full:
            self.model_depth.store(_depth_dir)


        self.predicted_y_depth.to_csv(os.path.join(_depth_dir, "predicted_y_depth.csv"))
        self.test_y_depth.to_csv(os.path.join(_depth_dir, "test_y_depth.csv"))

        if self.predicted_heldout_y_depth is not None:
            self.predicted_heldout_y_depth.to_csv(os.path.join(_depth_dir, "predicted_heldout_y_depth.csv"))

        # Color
        if self.color:

            _color_dir = os.path.join(self.root_dir, "color")
            if not os.path.exists(_color_dir):
                os.makedirs(_color_dir)

            if full:
                self.model_color.store(_color_dir)
                
            self.predicted_y_color.to_csv(os.path.join(_color_dir, "predicted_y_color.csv"))
            self.test_y_color.to_csv(os.path.join(_color_dir, "test_y_color.csv"))

            if self.predicted_heldout_y_color is not None:
                self.predicted_heldout_y_color.to_csv(os.path.join(_color_dir, "predicted_heldout_y_color.csv"))


        self.stored = True

    def fit_single(self, x, y, selected_samples):
        _new = self.model.clone()
        
        if issubclass(_new.__class__,  MlPipelineModel):
            _new.fit(x, y, samples_keys=selected_samples)
        else:
            _new.fit(x, y)
            
        return _new

    def extract_info(self, tt):

        self.model_info['data'] = {}
        self.model_info['data']['num_keys']        = len(tt.da.model_keys)

        self.model_info['facerenderer'] = {}
        self.model_info['facerenderer']['version'] = tt.da.frr.db_version

        self.model_info['data']['rosetta'] = {}
        self.model_info['data']['rosetta']['keys']           = tt.da.rdb_key_names
        self.model_info['data']['rosetta']['version']        = tt.da.rdb.r_version
        self.model_info['data']['rosetta']['filters']        = tt.da.rdb_filters
        self.model_info['data']['rosetta']['normalize_keys'] = tt.da.rdb_normalize_keys

        if hasattr(tt.da, 'vdb_keys'):
            self.model_info['data']['variantdb'] = {}
            self.model_info['data']['variantdb']['keys']      = tt.da.vdb_key_names
            self.model_info['data']['variantdb']['operation'] = tt.da.vdb_operation
            self.model_info['data']['variantdb']['params']    = tt.da.vdb_params
            self.model_info['data']['variantdb']['version']   = tt.da.vdb.db_version


        self.model_info['name']       = self.model.model_name
        self.model_info['params']     = self.model.params

        self.model_info['data']['num_samples']      = len(tt.X)
        self.model_info['data']['num_train_sample'] = len(tt.train_x)
        self.model_info['data']['num_test_sample']  = len(tt.test_x)


    def get_info(self):
        return self.model_info

    # ------------------------------------------
    #    Model execution
    # ------------------------------------------

    def fit(self, tt, color=True):

        self.extract_info(tt)

        # Depth
        self.model_depth                 = self.fit_single(tt.train_x, tt.train_y_depth, tt.train_x_samples)
        self.test_y_depth                = tt.test_y_depth
        
        self.predicted_y_depth           = pd.DataFrame()
        if len(tt.test_x) >0:
            self.predicted_y_depth           = self.model_depth.predict(tt.test_x)

        self.test_heldout_y_depth        = None
        self.predicted_heldout_y_depth   = None

        if hasattr(tt, "X_heldout"):
          if len(tt.X_heldout) > 0:
            self.test_heldout_y_depth        = tt.test_y_depth_heldout
            self.predicted_heldout_y_depth   = self.model_depth.predict(tt.X_heldout)


        # Color
        self.color = color
        if self.color:

            self.model_color                 = self.fit_single(tt.train_x, tt.train_y_color, tt.train_x_samples)

            self.test_y_color                = tt.test_y_color
            
            self.predicted_y_color           = pd.DataFrame()
            if len(tt.test_x) >0:
                self.predicted_y_color           = self.model_color.predict(tt.test_x)
            
            self.test_heldout_y_color        = None
            self.predicted_heldout_y_color   = None
            
            if hasattr(tt, "X_heldout"):
              if len(tt.X_heldout) > 0:
                self.test_heldout_y_color        = tt.test_y_color_heldout
                self.predicted_heldout_y_color   = self.model_color.predict(tt.X_heldout)


    # ------------------------------------------
    #    Distances and ranking
    # ------------------------------------------

    # def _norm(self, x):
    #     s = 0
    #     for i in range(0,len(x)):
    #         s = s + x[i]*x[i]
    #     return math.sqrt(s)
    #
    # def _dist(self, x,y):
    #     s = 0
    #     nx = self._norm(x)
    #     ny = self._norm(y)
    #     for i in range(0,len(x)):
    #         #s += (x[i]/nx - y[i]/ny)**2
    #         s += (x[i]/nx)*(y[i]/ny)/(i+10)
    #     s = s/len(x)
    #     return -s

    def _indetify(self, _id, t_y, p_y, topn=5):

        mse_dict = {}
        for i in range(len(t_y)):
            mse = cosine(p_y[_id],t_y[i])
            mse_dict[i] = mse

        sorted_x       = sorted(mse_dict.items(), key=operator.itemgetter(1))
        best_candidate = [sorted_x[bc][0] for bc in range(topn)]
        rank           = [ii[0] for ii in sorted_x].index(_id)+1

        return sorted_x, mse_dict, best_candidate, rank

    def compute_model_identification_ranking(self, kind="depth"):

        if kind == "depth":
            p_y = self.predicted_y_depth.values
            t_y = self.test_y_depth.values

        elif kind == "color":
            if not self.color:
                raise Exception("No model trained on color")
            p_y = self.predicted_y_color.values
            t_y = self.test_y_color.values

        elif kind == "combined":
            if not self.color:
                raise Exception("No model trained on color")

            p_y = np.hstack((self.predicted_y_color.values, self.predicted_y_depth.values))
            t_y = np.hstack((self.test_y_color.values, self.test_y_depth.values))

        ranks            = []
        one_over_ranks   = []
        topn_count       = 0

        for i in range(len(p_y)):
            sorted_x, mse_dict, best_candidate, rank = self._indetify(i, t_y, p_y, topn=5)
            ranks.append(rank)
            one_over_ranks.append(1.0/rank)
            if i in best_candidate:
                topn_count+=1

        ss_res         = (np.array(t_y)-np.array(p_y))**2
        mse            = np.mean(ss_res)
        rmse           = np.sqrt(mse)
        rank           = np.mean(ranks)
        one_over_rank  = np.mean(one_over_ranks)
        precision      = 100-rank/len(p_y)*100.0
        topn_prec      = 100.0*topn_count/len(p_y)

        if 'results' not in self.model_info.keys():
            self.model_info['results'] = {}

        self.model_info['results'][kind] = {}

        self.model_info['results'][kind]['avg_rank']          = rank
        self.model_info['results'][kind]['precision']         = precision
        self.model_info['results'][kind]['rmse']              = rmse
        self.model_info['results'][kind]['topn_prec']         = topn_prec
        self.model_info['results'][kind]['avg_one_over_rank'] = one_over_rank

        return rmse, rank, len(p_y), precision, topn_prec, one_over_rank

    # ------------------------------------------
    #    Visualizer
    # ------------------------------------------

    def _get_comapre_data(self, name):


        if name in self.predicted_y_depth.index:

            _test_depth = list(self.test_y_depth[self.test_y_depth.index==name].values[0])
            _pred_depth = list(self.predicted_y_depth[self.predicted_y_depth.index==name].values[0])

            if self.color:
                _test_color = list(self.test_y_color[self.test_y_color.index==name].values[0])
                _pred_color = list(self.predicted_y_color[self.predicted_y_color.index==name].values[0])

        elif name in self.predicted_heldout_y_color.index:

            _test_depth = list(self.test_heldout_y_depth[self.test_heldout_y_depth.index==name].values[0])
            _pred_depth = list(self.predicted_heldout_y_depth[self.predicted_heldout_y_depth.index==name].values[0])

            if self.color:
                _test_color = list(self.test_heldout_y_color[self.test_heldout_y_color.index==name].values[0])
                _pred_color = list(self.predicted_heldout_y_color[self.predicted_heldout_y_color.index==name].values[0])

        else:
            raise Exception("Cannot find this name")


        return name, _test_depth, _test_color, _pred_depth, _pred_color

    def render_3d(self, frr, name):

        name, _test_depth, _test_color, _pred_depth, _pred_color = self._get_comapre_data(name)

        print "Subject {}".format(name)
        _, _, id_test = frr.generate_obj(x=_test_depth, z=_test_color, quality=5, extended=True)
        _, _, id_pred = frr.generate_obj(x=_pred_depth, z=_pred_color, quality=5, extended=True)


    def render_compare_2d(self, frr, name, store=False, plot=True):

        import matplotlib.image as mpimg
        import matplotlib.pylab as plt
        from numpy import linalg, array

        name, _test_depth, _test_color, _pred_depth, _pred_color = self._get_comapre_data(name)

        res_png_test, _, id_test = frr.generate_2d(x=_test_depth, z=_test_color, quality=5)
        res_png_pred, _, id_pred = frr.generate_2d(x=_pred_depth, z=_pred_color, quality=5)
        tp_dist                  = linalg.norm(array(_test_depth) - array(_pred_depth))

        if self.color:
            tp_dist_color            = linalg.norm(array(_test_color) - array(_pred_color))

        img_test = mpimg.imread(res_png_test)
        img_pred = mpimg.imread(res_png_pred)

        fig     = plt.figure()
        ax      = fig.add_subplot(1,2,1)
        imgplot = plt.imshow(img_test)
        ax.set_title("Subject {}".format(name))
        ax.axis('off')

        ax      = fig.add_subplot(1,2,2)
        imgplot = plt.imshow(img_pred)
        if self.color:
            ax.set_title('D: {0:.2e} C: {1:.2e}'.format(tp_dist, tp_dist_color))
        else:
            ax.set_title('D: {0:.2e}'.format(tp_dist))
        ax.axis('off')

        if store:

            if not self.stored:
                self.store()

            fname = os.path.join(self.root_dir, "{}.png".format(name))
            plt.savefig(fname)
            plt.close(fig)

        elif plot:
            plt.show()

    def store_face_matrixes(self, frr, name):

        import cPickle

        name, _test_depth, _test_color, _pred_depth, _pred_color = self._get_comapre_data(name)

        u_test, w_test = frr.compute_mesh(x=_test_depth, z=_test_color)
        u_pred, w_pred = frr.compute_mesh(x=_pred_depth, z=_pred_color)

        with open(os.path.join(self.root_dir, "{}.u_test.pkl".format(name)), "wb") as f:
            cPickle.dump(u_test, f)

        with open(os.path.join(self.root_dir, "{}.w_test.pkl".format(name)), "wb") as f:
            cPickle.dump(w_test, f)

        with open(os.path.join(self.root_dir, "{}.u_pred.pkl".format(name)), "wb") as f:
            cPickle.dump(u_pred, f)

        with open(os.path.join(self.root_dir, "{}.w_pred.pkl".format(name)), "wb") as f:
            cPickle.dump(w_pred, f)


# -------------------------------------
#   Models
# -------------------------------------


class PredictorByComponent(Model):

    def __init__(self, *argv, **kwargs):

        super(PredictorByComponent, self).__init__(*argv, **kwargs)
        self.model_name  = self.__class__.__name__

    def fit(self, x, y):

        self.model_names = {}
        self.cols   = y.columns

        for c in range(len(self.cols)):
            clf = self.model_class(**self.params)
            clf.fit(x, y[self.cols[c]])
            self.model_names[c] = clf

    def predict(self, x):

        self.predict_y = []
        for c in range(len(self.cols)):
            self.predict_y.append(self.model_names[c].predict(x))

        _y   = np.array(self.predict_y).T
        _dfy = pd.DataFrame(_y, index=x.index)
        return _dfy

    def store(self, _dir):

        import pickle

        for c in range(len(self.cols)):
            _mfname = os.path.join(_dir, "{}.array.{}.pkl".format(self.STORE_M_BASE, c))
            with open(_mfname, "wb") as _f:
                pickle.dump(self.model_names[c], _f)

        _mtname = os.path.join(_dir, "{}.cols.pkl".format(self.STORE_M_BASE))
        with open(_mtname, "wb") as _f:
            pickle.dump(self.cols, _f)


    def load(self, _dir):

        import pickle

        _mtname = os.path.join(_dir, "{}.cols.pkl".format(self.STORE_M_BASE))
        with open(_mtname, "rb") as _f:
            self.cols = pickle.load(_f)

        self.model_names = {}
        for c in range(len(self.cols)):
            _mfname = os.path.join(_dir, "{}.array.{}.pkl".format(self.STORE_M_BASE, c))
            with open(_mfname, "rb") as _f:
                self.model_names[c] = pickle.load(_f)


    def clone(self):
        return self.__class__(self.model_class, self.params)


    from sknn.mlp import Regressor, Layer


class RidgeCVByComponent(PredictorByComponent):

    def __init__(self, *argv, **kwargs):
        super(RidgeCVByComponent, self).__init__(linear_model.RidgeCV, **kwargs)

class kNNByComponent(PredictorByComponent):

    def __init__(self, *argv, **kwargs):
        super(kNNByComponent, self).__init__(neighbors.KNeighborsRegressor, **kwargs)

class SVRByComponent(PredictorByComponent):

    def __init__(self, *argv, **kwargs):
        super(SVRByComponent, self).__init__(svm.SVR, **kwargs)

#------------------------------------------
#   Neural networks
#------------------------------------------

class SimpleMlp(Model):

    def __init__(self, **kwargs):

        super(SimpleMlp, self).__init__(SimpleMlp, **kwargs)

    def _init_nn(self, num):

        import copy

        self.params_clean = copy.copy(self.params)
        layer_type        = self.params_clean.pop("layer_type", "Rectifier")


        self.nn = mlp.Regressor(
            layers=[mlp.Layer(layer_type, units=int(num/2)),
                    mlp.Layer("Linear")],
            **self.params
        )

        self.model_name = self.nn.__class__.__name__

    def fit(self, x, y):

        self._init_nn(len(x))
        self.nn.fit(x.values, y.values)

    def predict(self, x):

        _y   = self.nn.predict(x.values)
        _dfy = pd.DataFrame(_y, index=x.index)
        return _dfy

    def store(self, _dir):

        import pickle

        _mtname = os.path.join(_dir, "{}.nn.pkl".format(self.STORE_M_BASE))
        with open(_mtname, "wb") as _f:
            pickle.dump(self.nn, _f)

    def load(self, _dir):

        import pickle

        _mtname = os.path.join(_dir, "{}.nn.pkl".format(self.STORE_M_BASE))
        with open(_mtname, "rb") as _f:
            self.nn = pickle.load(_f)

    def clone(self):
        return self.__class__(**self.params)




class DeepMlp(Model):

    def __init__(self, **kwargs):

        super(DeepMlp, self).__init__(DeepMlp, **kwargs)

    def _init_nn(self, num):

        import copy

        self.params_clean = copy.copy(self.params)
        layer_type        = self.params_clean.pop("layer_type", "Rectifier")


        self.nn = mlp.Regressor(
            layers=[mlp.Layer(layer_type, units=int(num/2)),
                    mlp.Layer(layer_type, units=int(num/2)),
                    mlp.Layer(layer_type, units=int(num/2)),
                    mlp.Layer("Linear")],
            **self.params
        )

        self.model_name = self.nn.__class__.__name__

    def fit(self, x, y):

        self._init_nn(len(x))
        self.nn.fit(x.values, y.values)

    def predict(self, x):

        _y   = self.nn.predict(x.values)
        _dfy = pd.DataFrame(_y, index=x.index)
        return _dfy

    def store(self, _dir):

        import pickle

        _mtname = os.path.join(_dir, "{}.nn.pkl".format(self.STORE_M_BASE))
        with open(_mtname, "wb") as _f:
            pickle.dump(self.nn, _f)

    def load(self, _dir):

        import pickle

        _mtname = os.path.join(_dir, "{}.nn.pkl".format(self.STORE_M_BASE))
        with open(_mtname, "rb") as _f:
            self.nn = pickle.load(_f)

    def clone(self):
        return self.__class__(**self.params)


# -----------------------------------------
#    Demo
# -----------------------------------------

class MlPipelineModel(Model):
    pass
    
class MlPipelineMultivariateRidge(MlPipelineModel):

    def __init__(self, **kwargs):

        super(MlPipelineMultivariateRidge, self).__init__(MlPipelineMultivariateRidge, **kwargs)
        self.model_name  = self.__class__.__name__

    def fit(self, x, y, samples_keys = None):
        
        import datastack.ml.baseregress as baseregress
        
        target_dict={}
        for i in xrange(1,len(y.columns)+1):
            target_dict['c{}'.format(i)]=y.columns[i-1]
        
        # Definition of the target values
        base = baseregress.BaseRegress(target_dict,multi=True)
        xy   = x.merge(y, left_index=True, right_index=True)
        xy   = xy.rename(columns = {"qc.sample_key":"ds.index.sample_key"})
        
        if samples_keys is None:
            xy['ds.index.sample_key'] = xy.index.values
        else:
            xy['ds.index.sample_key'] = samples_keys
        
        base.addData(xy)
        
        # Estimators
        m_params     = {'alpha': [ 0.0001, 0.001, 0.01,  0.1, 1, 10, 1000]}
        if "alpha" in self.params:
            m_params = {'alpha': self.params['alpha']} 
            
        estimator    = {'est' : linear_model.Ridge(), 'params' : m_params}
        
        print estimator
        
        # drop some error columns        
        base.covariates['BMI']            = self.params["bmi_key"]    # Name of the DF columns, normalized data
        base.covariates['Age']            = self.params["age_key"]    # Name of the DF columns, normalized data
        base.covariates['Gender']         = self.params["gender_key"] # Gender is -1, 1
        base.covariates['Ethnicity']      = self.params["ethnicity_key"]     # 1000 KinshipPC
        
        base.estimatorsToRun['Ethnicity'] = [estimator]     # Gender is -1, 1
        base.estimatorsToRun['AGE']       = [estimator]
        base.estimatorsToRun['AGE + BMI'] = [estimator]
        
        #Do the training
        base.run( with_aggregate_covariates = True, kfargs = {'n_folds_holdout': 0} )
        
        # Table output parameters
        fmt = {'R2':3, 'SELECT10':3}
        base.dropCol('MSE')
        base.dropCol('MAE')
        base.display(fmtdict=fmt)
        
        base_cv     = base.cv[('AGE + BMI', 'multi', 0)]
        
        self.model       = base_cv.get_estimator()
        self.model_order = base_cv.y.columns
        self.y_order     = y.columns
        
    def predict(self, x):
        
        if not isinstance(x, pd.DataFrame):
            _x  = np.array(x)
            if len(_x.shape)==1:
                _x = pd.DataFrame([x])
            else:
                _x = pd.DataFrame(x)
        else:
            _x  = x[self.model.hli_features]

        self.predict_y = self.model.predict(_x)
        
        _dfy           = pd.DataFrame(self.predict_y, index=_x.index, columns=self.model_order)
        _dfy           = _dfy[self.y_order]
         
        return _dfy

    def store(self, _dir):

        import pickle
        
        # Pickle the model
        _mfname = os.path.join(_dir, "{}.array.pkl".format(self.STORE_M_BASE))
        with open(_mfname, "wb") as _f:
            pickle.dump(self.model, _f,protocol=pickle.HIGHEST_PROTOCOL)
        
        # Pickle the model order
        _mfname = os.path.join(_dir, "{}.order.pkl".format(self.STORE_M_BASE))
        with open(_mfname, "wb") as _f:
            pickle.dump(self.model_order, _f,protocol=pickle.HIGHEST_PROTOCOL)
        
        # Pickle the model order
        _mfname = os.path.join(_dir, "{}.order_y.pkl".format(self.STORE_M_BASE))
        with open(_mfname, "wb") as _f:
            pickle.dump(self.y_order, _f,protocol=pickle.HIGHEST_PROTOCOL)
    
    def load(self, _dir):

        import pickle

        _mfname = os.path.join(_dir, "{}.array.pkl".format(self.STORE_M_BASE))
        with open(_mfname, "rb") as _f:
            self.model = pickle.load(_f)

        _mfname = os.path.join(_dir, "{}.order.pkl".format(self.STORE_M_BASE))
        with open(_mfname, "rb") as _f:
            self.model_order = pickle.load(_f)
        
        _mfname = os.path.join(_dir, "{}.order_y.pkl".format(self.STORE_M_BASE))
        with open(_mfname, "rb") as _f:
            self.y_order = pickle.load(_f)

    def clone(self):
        return self.__class__(**self.params)


    from sknn.mlp import Regressor, Layer
