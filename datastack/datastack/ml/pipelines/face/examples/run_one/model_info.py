import sys
sys.path.insert(0, '../')

from model_setup import *
from datastack.common.settings import get_logger
logger = get_logger("default")


# ----------------------------------------
#   Model parameters
# ----------------------------------------

model_rosetta_keys = ["genome_pc_keys", "pheno_keys_gender", "pheno_keys_age_bmi"]
model_rosetta_norm = ['dynamic.FACE.pheno.v1.bmi', 'dynamic.FACE.pheno.v1.age']

model_variant_keys = [] 

# ----------------------------------------
#   Model parameters
# ----------------------------------------

models_params = {}
models_params["MlPipelineMultivariateRidge"] = {'alpha'         : [ 0.0001, 0.001, 0.01,  0.1, 1, 10, 1000], 
                                                'bmi_key'       : ['dynamic.FACE.pheno.v1.bmi'],
                                                'age_key'       : ['dynamic.FACE.pheno.v1.age'],
                                                'gender_key'    : ['dynamic.FACE.pheno.v1.gender_plusminus'],
                                                'ethnicity_key' : ["genomics.kinship.pc.{}".format(i+1) for i in range(1000)]}


# ----------------------------------------
#   TT Parameters
# ----------------------------------------

train_test_ratio         = 1.0
N_face_PC                = 40

# ----------------------------------------
#   Rendering and other options
# ----------------------------------------

store_full_model         = True
