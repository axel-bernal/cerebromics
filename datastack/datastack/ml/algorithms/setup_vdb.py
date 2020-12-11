from datastack.dbs import rdb as rosetta
from datastack.dbs import vdb as variantdb


def initialize_datastack(build="38", get_rdb=True, get_vdb=True):
    """
    initialize and return the Rosetta and variant databases

    Parameters
    ----------
    build       reference genome build ("38" and "19" are supported, default "38")
    get_rdb     If false, the returns None for rdb. (boolean, default = True)
    get_vdb     If false, the returns None for rdb. (boolean, default = True)

    Returns
    -------
    rdb, vdb
    """
    if get_vdb:
        vdb = variantdb.HpcVarianceDB(host="variantdb.hli.io", port="8080")
        vdb.initialize(version='HG%s_IsaacVariantCaller_AllSamples_PASS_MAF_001_gVCF' % build)
    else:
        vdb = None
    if get_rdb:
        rdb = rosetta.RosettaDBMongo(host="rosetta.hli.io")
        if build == "38":
            namespace = u'hg38_noEBV'
        elif build == "19":
            namespace = 'hg19'
        rdb.initialize(namespace=namespace)
    else:
        rdb = None
    return rdb, vdb