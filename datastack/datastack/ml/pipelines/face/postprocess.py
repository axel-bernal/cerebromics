from datastack.ml.pipelines import face
from datastack.dbs.rdb import RosettaDBMongo
from datastack.dbs.vdb import HpcVarianceDB
from datastack.tools import facerenderer
from datastack.common.settings import get_logger
from datastack.ml.pipelines.face.settings import MODEL_ROOT_PREFIX
import os

logger = get_logger('default')

try:
    from model_info import *
except:
    raise Exception("You need to have a model_info.py file in the directory")


def main():


    # Initialize datastack

    logger.info("Initializing the datastack ...")
    frr = facerenderer.FaceRenderer(host=jarvis_host)
    frr.initialize(face_version)

    rdb = RosettaDBMongo(host=rosetta_host)
    rdb.initialize(namespace=rosetta_namespace)

    vdb = HpcVarianceDB(host=variantdb_host, port=variantdb_port)
    vdb.initialize(version=variantdb_version)


    # Models to postprocess
    _dir       = os.getcwd()
    all_models = {}
    print os.listdir(os.path.join(_dir, MODEL_ROOT_PREFIX))
    _subdirs   = [o for o in os.listdir(os.path.join(_dir, MODEL_ROOT_PREFIX)) if os.path.isdir(o) and os.path.isfile(os.path.join(o, "info.json"))]

    print _subdirs
    # Models definitions

    # for mname in models_to_run.keys():
    #
    #     logger.info("Running {} model...".format(mname))
    #
    #     mparams = {}
    #     if mname in models_params.keys():
    #         mparams = models_params[mname]
    #         logger.info("Parameters found for {} model: {}".format(mname, mparams))
    #
    #     m      = models_to_run[mname](**mparams)
    #
    #     m_exec = models.ModelExecutor(m)
    #     m_exec.fit(tt)
    #
    #     _out = m_exec.compute_model_identification_ranking(kind="depth")
    #     logger.info("{} model trained for Depth with precision: {}".format(mname, _out[3]))
    #
    #     _out = m_exec.compute_model_identification_ranking(kind="color")
    #     logger.info("{} model trained for Color with precision: {}".format(mname, _out[3]))
    #
    #     logger.info("Storing {} model".format(mname))
    #     m_exec.store()
    #
    #     if render_heldout:
    #         for i in range(len(m_exec.predicted_heldout_y_depth)):
    #             logger.info("Exporting {} renders {}/{}".format(mname, i, len(m_exec.predicted_heldout_y_depth)))
    #
    #             _name = m_exec.predicted_heldout_y_depth.index[i]
    #             m_exec.render_compare_2d(frr, _name, store=True, plot=False)
    #             m_exec.render_3d(frr, _name)
    #
    #     if store_test_face_matrixes:
    #         for i in range(len(m_exec.test_y_color.index)):
    #             logger.info("Exporting {} face matrixes {}/{}".format(mname, i, len(m_exec.test_y_color)))
    #
    #             _name = m_exec.test_y_color.index[i]
    #             m_exec.store_face_matrixes(frr, _name)



if __name__ == '__main__':
	main()

