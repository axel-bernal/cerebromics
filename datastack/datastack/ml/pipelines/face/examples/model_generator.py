from datastack.ml.pipelines.face import models
import os
import shutil

def generete_file(model_rosetta_keys, max_samples, face_version,
                  vdb_genes=None, model_rosetta_norm=None,
                  maf=0.05):


  if vdb_genes is None:
    vdb_genes = ""
  else:
    vdb_genes = "\"{}\"".format(vdb_genes)

  if model_rosetta_norm is None:
    model_rosetta_norm = ""
  else:
    model_rosetta_norm = "\"{}\"".format(model_rosetta_norm)

  base_model ="""import sys
sys.path.insert(0, '../')

from model_setup import *
from datastack.common.settings import get_logger
logger = get_logger("default")


# ----------------------------------------
#   Medel keys
# ----------------------------------------

face_version = '"""+str(face_version)+"""'

# Keys: use a list from the rosetta_gropus keys

model_rosetta_keys = ["""+str(model_rosetta_keys)+"""]
model_rosetta_norm = ["""+str(model_rosetta_norm)+"""]

# Keys: use a list from the variantdb_groups keys
# Operations: get_gene_range, get_transcript_ranges, get_cdna_ranges

model_variant_keys      = ["""+str(vdb_genes)+"""]
model_variant_operation = "get_cdna_ranges"

# Options: run_pca, pca_components, maf, add_left, add_right, num_based
model_variant_parametrs = {"maf":"""+str(maf)+"""}

# ----------------------------------------
#   Medel parameters
# ----------------------------------------

N_face_PC                                      = 40
max_samples                                    = """+str(max_samples)+"""

models_params = {}
models_params["PredictorByComponent.RidgeCV"]  = {"cv": 10}
#models_params["PredictorByComponent.kNN"]     = {}
#models_params["PredictorByComponent.SVR"]     = {}
#models_params["SimpleMultilayerPerceptron"]   = {"learning_rate": 0.00002, "n_iter": 500, "learning_rule": "sgd"}


# ----------------------------------------
#   Output options
# ----------------------------------------

render_heldout           = False
store_test_face_matrixes = False
                  
"""

  return base_model

face_versions      = ["v1", "v2", "v3", "v4", "v5"]
max_samples        = [50, 100, 200, 400, 800, 1000]
models             = ["ethnicity", "ethnicity.pheno_sex", "ethnicity.pheno_sex.pheno_ab", \
                      "genomepc", "genomepc.pheno_sex", "genomepc.pheno_sex.pheno_ab", \
                      "genomepc.ethnicity", "genomepc.ethnicity.pheno_sex", "genomepc.ethnicity.pheno_sex.pheno_ab"]

vbd_models         = [None, "g_human", "g_shriver"]

for m in models:
  for f in face_versions:
    for s in max_samples:
      for v in vbd_models:

        _dir = "{}_{}_{}".format(m, f, s)
        if not v is None:
          _dir = "{}-{}_{}_{}".format(m, v, f, s)

        # Make the directories
        if os.path.isdir(_dir):
          shutil.rmtree(_dir)
        os.mkdir(_dir)

        keys   = "\"{}\"".format("\",\"".join(m.split(".")))

        r_norm = None
        if "pheno_ab" in keys:
          r_norm = "'dynamic.FACE.pheno.v1.bmi', 'dynamic.FACE.pheno.v1.age'"

        print "Writing {}".format(_dir)

        # Open the file
        with open(os.path.join(_dir, "model_info.py"), "w") as fw:
          fw.write(generete_file(keys, s, f, vdb_genes=v, model_rosetta_norm=r_norm))

