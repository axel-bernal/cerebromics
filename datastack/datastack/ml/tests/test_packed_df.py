"""
Test file for f2ace_io

At the moment this tests the ability to read the three kinds of files that describe data pushed to
Rosetta.   We don't test the actual push, because we'd have to stand up a fake mongo server running
Rosetta code.  Or we'd have to delete dynamic entries.  Both of which are rather painful or undocumented.
Instead we pass in text or csv files that should create legal dataframes.
"""

import pytest
import pandas as pd
import os
import hashlib
import json

from datastack.ml.model_utils import PackedFrame
from datastack.tools.fivector import common

def _cksum(data):
    hash = hashlib.md5()
    for i in data:
        hash.update(bytes(i))
    return hash.hexdigest()

PACK_COL = 'jsonData'

def test_packed_frame(ml_config_fixture):
    """
    Create PackedFrame with test data.
    Sum test data.   Unpack.  Repack.  Sum.

    Args:
        ml_config_fixture:
    Returns:
    """
    print "Testing datastack.ml.model_utils.PackedFrame reading {}".format(ml_config_fixture.TEST_DATA)

    is_csv = True
    df = common.read_f2ace_csv(is_csv, os.path.join(ml_config_fixture.TEST_DATA,"test_packedframe.csv"))
    df[PACK_COL] = df[PACK_COL].apply(lambda x: json.loads(x))
    # NOTE: We could use pandas.util.testing assert_frame_equal instead
    test_checksums = df[PACK_COL].apply(_cksum)

    pdf = PackedFrame(df)
    unpacked_df = pdf.unpack()  # pdf.df no longer has packed column, unpacked_df has all columns
    print "unpacked df {}".format(unpacked_df.head(1))

    # Hack to put the exploded columns back into the packed df with it's column map
    pdf.df = unpacked_df
    repacked_df = pdf.pack(PACK_COL, to_json=False)
    print "repacked df {}".format(repacked_df.head(1))

    repacked_df.to_csv(ml_config_fixture.TEST_DATA+"/test_REpackedframe.csv",sep="\t",index=False)
    assert PACK_COL in repacked_df.columns
    repacked_checksums = repacked_df[PACK_COL].apply(_cksum)

    print "packed checksums:   {}".format(test_checksums)
    print "repacked checksums: {}".format(repacked_checksums)

    assert list(repacked_checksums) == list(test_checksums)



