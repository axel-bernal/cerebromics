"""
There are unfortunately several different settings files.
We are making ../settings.py the standard location for column names and
similar information.
DO NOT add that kind of thing here.
"""
from datastack.settings import QUICKSILVER_METADATA_KEYS, ROSETTA_SUBJECT_NAME_KEY, ROSETTA_STUDY_KEY, ROSETTA_INDEX_KEY # @UnusedImport

NAMESPACE = "hg38_noEBV"
PROJECT_KEY_OLD = "qc.ProjectID"
AGE = "pheno.age"
SAMPLE_NAME = ROSETTA_SUBJECT_NAME_KEY 
PROJECT_KEY = ROSETTA_STUDY_KEY
SAMPLE_KEY = ROSETTA_INDEX_KEY
