from datastack.ml.pipelines.face import models

# ----------------------------------------
#   General imports and definitions
# ----------------------------------------

jarvis_host       = "jarvis.hli.io"

rosetta_host      = "rosetta.hli.io"
rosetta_namespace = "hg19"

variantdb_host    = "variantdb.hli.io"
variantdb_port    = "8080"
variantdb_version = "HG19_IsaacVariantCaller_AllSamples_PASS_MAF_001_gVCF"


# ----------------------------------------
#   Models map
# ----------------------------------------

models_to_run                                 = {}
models_to_run["MlPipelineMultivariateRidge"]  = models.MlPipelineMultivariateRidge
#models_to_run["PredictorByComponent.RidgeCV"] = models.RidgeCVByComponent
#models_to_run["PredictorByComponent.kNN"]     = models.kNNByComponent
#models_to_run["PredictorByComponent.SVR"]     = models.SVRByComponent
#models_to_run["SimpleMultilayerPerceptron"]   = models.SimpleMlp

# ----------------------------------------
#   Rosetta map
# ----------------------------------------

rosetta_groups                        = {}
rosetta_groups['genome_pc_keys']      = "genomics.kinship.pc."
rosetta_groups['pheno_keys_gender']   = ["dynamic.FACE.pheno.v1.gender_plusminus"]
rosetta_groups['pheno_keys_age_bmi']  = ["dynamic.FACE.pheno.v1.age", "dynamic.FACE.pheno.v1.bmi"]


# ----------------------------------------
#   VariantDB map
# ----------------------------------------

variantdb_groups                      = {}
# variantdb_groups['g_human'] = ["ENSG00000065618", #"COL17A1"
#                                "ENSG00000073282", #"TP63"
#                                "ENSG00000135903", #"PAX3L"
#                                "ENSG00000142611", #PRDM16
#                                "ENSG00000185662"  #C5orf50
#                               ]
# 
# variantdb_groups['g_shriver'] = ["ENSG00000189056", "ENSG00000150893", "ENSG00000108557", "ENSG00000187741", \
#                                  "ENSG00000126934", "ENSG00000164692", "ENSG00000164190", "ENSG00000105372", \
#                                  "ENSG00000166147", "ENSG00000068078", "ENSG00000157764", "ENSG00000198363", \
#                                  "ENSG00000137203", "ENSG00000120725", "ENSG00000078401", "ENSG00000117298", \
#                                  "ENSG00000182533", "ENSG00000184058", "ENSG00000157933", "ENSG00000147655", \
#                                  "ENSG00000124587", "ENSG00000169862", "ENSG00000142798", "ENSG00000034693", \
#                                  "ENSG00000106571", "ENSG00000186184", "ENSG00000066468", "ENSG00000104447", \
#                                  "ENSG00000078114", "ENSG00000125378", "ENSG00000213341", "ENSG00000142303", \
#                                  "ENSG00000165671", "ENSG00000125798", "ENSG00000160789", "ENSG00000169554", \
#                                  "ENSG00000125965", "ENSG00000169032", "ENSG00000169071", "ENSG00000119042", \
#                                  "ENSG00000188641", "ENSG00000142655", "ENSG00000088305", "ENSG00000197594", \
#                                  "ENSG00000184937", "ENSG00000106511", "ENSG00000128739", "ENSG00000121680", 
#                                  "ENSG00000070010"]

# ----------------------------------------
#   Face version
# ----------------------------------------

# Keys: v1, v2, v3, v4, v5
face_version = 'v2'

# ----------------------------------------
#   Sample details
# ----------------------------------------

force_test_set  = False
heldout_samples = None

# Samples not to use in training or testing, https://github.com/hlids/cerebro/wiki/Face-Problems
error_samples   = ["15-0153", "15-0041", "15-0047", "15-0275", \
                  "15-1072", "15-0032", "15-0156", "15-0179", \
                  "15-0107", "15-0191", "15-0787", "15-0107", \
                  "15-0656", "15-0346", "15-0959", "15-0444", "15-0444"]

free_release     = []
force_test_set   = free_release + []

# ----------------------------------------
#   Output options
# ----------------------------------------

max_samples              = None

render_heldout           = False
store_test_face_matrixes = False
store_full_model         = False
