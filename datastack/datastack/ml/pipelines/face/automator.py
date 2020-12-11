from datastack.ml.pipelines import face
from datastack.dbs.rdb import RosettaDBMongo
from datastack.dbs.vdb import HpcVarianceDB
from datastack.tools import facerenderer
from datastack.common.settings import get_logger

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

    if variantdb_host is not None:
        vdb = HpcVarianceDB(host=variantdb_host, port=variantdb_port)
        vdb.initialize(version=variantdb_version)
    else:
        vdb = None

    # Rosetta data
    rosetta_converted        = {}
    for k in model_rosetta_keys:
        if type(rosetta_groups[k])==list:
            rosetta_converted[k]      = []
            for l in rosetta_groups[k]:
                rosetta_converted[k] += rdb.find_keys(l)
        else:
            rosetta_converted[k]      = rdb.find_keys(rosetta_groups[k])

    # Create the dataset

    logger.info("Creating the dataset ...")
    da = face.DatasetAssembler(rdb, vdb, frr,
                               rosetta_map   = rosetta_converted,
                               variantdb_map = variantdb_groups)


    _out = da.prepare_rosetta(model_rosetta_keys,
                              normalize_keys = model_rosetta_norm,
                              max_samples    = max_samples,
                              force_test_set = force_test_set
                              )
    logger.info("Rosetta extraction return {} records with {} keys".format(_out[0], _out[1]))

    if model_variant_keys is not None and len(model_variant_keys)>0:
        _out = da.add_variants(model_variant_keys, model_variant_operation, **model_variant_parametrs)
        logger.info("Variant DB extraction return {} records with {} keys".format(_out[0], _out[1]))

    # Create the train test

    logger.info("Creating the train/test split ...")
    tt = face.TrainTestContainer(da)

    _out = tt.create_train_test_split(heldout_samples  = heldout_samples,
                                      remove_samples   = error_samples,
                                      train_test_ratio = train_test_ratio,
                                      N_face_PC        = N_face_PC,
                                      force_test_set   = force_test_set
                          )

    logger.info("Train test split initialized with {} to {} samples".format(_out[0], _out[1]))

    # Models definitions

    for mname in models_to_run.keys():

        logger.info("Running {} model...".format(mname))

        mparams = {}
        if mname in models_params.keys():
            mparams = models_params[mname]
            logger.info("Parameters found for {} model: {}".format(mname, mparams))

        m      = models_to_run[mname](**mparams)

        m_exec = model_names.ModelExecutor(m)
        m_exec.fit(tt)

        _out = m_exec.compute_model_identification_ranking(kind="depth")
        logger.info("{} model trained for Depth with precision: {} - one over rank {}".format(mname, _out[3], _out[5]))

        _out = m_exec.compute_model_identification_ranking(kind="color")
        logger.info("{} model trained for Color with precision: {} - one over rank {}".format(mname, _out[3], _out[5]))

        _out = m_exec.compute_model_identification_ranking(kind="combined")
        logger.info("{} model trained for Combined with precision: {} - one over rank {}".format(mname, _out[3], _out[5]))

        logger.info("Storing {} model".format(mname))
        m_exec.store(full=store_full_model)

        if render_heldout:
            for i in range(len(m_exec.predicted_heldout_y_depth)):
                logger.info("Exporting {} renders {}/{}".format(mname, i, len(m_exec.predicted_heldout_y_depth)))

                _name = m_exec.predicted_heldout_y_depth.index[i]
                m_exec.render_compare_2d(frr, _name, store=True, plot=False)
                m_exec.render_3d(frr, _name)

        if store_test_face_matrixes:
            for i in range(len(m_exec.test_y_color.index)):
                logger.info("Exporting {} face matrixes {}/{}".format(mname, i, len(m_exec.test_y_color)))

                _name = m_exec.test_y_color.index[i]
                m_exec.store_face_matrixes(frr, _name)



if __name__ == '__main__':
	main()

