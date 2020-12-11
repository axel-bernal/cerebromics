import sys
sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
import datastack.ml.preprocess as pp
import pandas as pd
import numpy as np
from pandas.util import testing as tt
from datastack.ml import rosetta_settings as ros_vars


def demo_outlier_removal():
    from sklearn.grid_search import GridSearchCV
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Read in data
    XY = pd.read_table("../testdata/XY_coverage_normalized_face_08012015.txt", sep='\t')

    # Initialize model
    model = pp.FlexWindowOutlier()

    # Get parameters to search
    params = model.get_gridsearch(n_evals=100)

    # Create gridsearch object
    gs = GridSearchCV(model, params, n_jobs=4)

    # Optimize parameters for multiple outcomes
    for outcome in ["Xc", "Yc"]:
        print "\n====== %s ======" % outcome
        gs.fit(XY[["facepheno.Age"]], XY[outcome])
        print "\tBest trimming parameters are:", gs.best_params_
        print "\tBest r2 score is: %s" % gs.best_score_
        XY["%s_trim" % outcome] = gs.best_estimator_.trim(XY["Yc"])
        sns.jointplot("facepheno.Age", "%s_trim" % outcome, XY, kind="reg")
        plt.show()


def test_filters():
    # Test (row-wise) missingness and user filters
    data1 = pd.DataFrame({"X": [1, 2, np.nan, 4, 5], "Y": [1, 2, 3, np.nan, 5], "F1": ["Filter"] + ["Yay"] * 4, "F2": ["Yay"] * 4 + ["Filter"]})
    xygen1 = pp.XYGenerator("X", "Y", data_filters={"F1": [("!=", "Filter")], "F2": [("!=", "Filter")]}, clean=True, data=data1)
    X1, y1 = xygen1()
    tt.assert_frame_equal(X1, pd.DataFrame({"X": [2.]}, index=[1]))
    assert (xygen1.filter_info.ix["Missing Values", ["F1", "F2", "X", "Y"]] == [0, 0, 1, 1]).all()
    assert (xygen1.filter_info.ix["User Filters", ["F1", "F2", "X", "Y"]] == [1, 1, 0, 0]).all()
    assert (data1.shape[0] - X1.shape[0]) == xygen1.filter_info.loc[["Missing Values", "User Filters"]].sum().sum()

    # Test column wise missingness filter and outlier flagging
    data2 = pd.DataFrame({"X": range(99) + [1e6], "Y": range(100), "Z": [np.nan] * 100})
    xygen2 = pp.XYGenerator("X|Z", "Y", clean=True, data=data2)
    assert (xygen2.filter_info.ix["Outliers (warn)", ["X", "Y"]] == [1, 0]).all()
    assert (xygen2.filter_info.ix["Dropped", ["X", "Y", "Z"]] == [False, False, True]).all()

    # Test outlier flagging (with trimming)
    xygen3 = pp.XYGenerator("X|Z", "Y", clean=True, data=data2, trim_outliers=True)
    assert (xygen3.filter_info.ix["Outliers (trim)", ["X", "Y"]] == [1, 0]).all()
    assert (xygen3.filter_info.ix["Outliers (warn)", ["X", "Y"]] == [0, 0]).all()

    print "\nPassed filter tests..."


def test_patsy_formula():
    df_test = pd.DataFrame({"X.1": [1, np.nan, 6], "X 5": ["T", "F", "T"], "Y.2": [3, 4, 9], "G": ["M", "F", "F"]})
    # Include X and G as regressors, as well as interaction thereof.
    # Use Y as outcome
    xygen = pp.XYGenerator("(X)|(G)", "Y", interactions=["G:X 5"], data=df_test, clean=True)
    X, y = xygen()

    # X.1 dropped due to missingness

    expected = pd.DataFrame({"X.1": [1., 6.], "X_5[T.T]": [1., 1.], 
                             "G[T.M]:X_5[F]": [0., 0.], "G[T.M]:X_5[T]": [1., 0.]}, 
                             index=[0, 2], columns=X.columns)
    tt.assert_frame_equal(expected, X)
    print "\nPatsy formula OK..."


def test_fix_non_numeric():
    data = pd.DataFrame({"X": range(99) + ["Yay"], "Y": range(100)})
    xygen = pp.XYGenerator("X", "Y", data=data, clean=True)
    X, y = xygen()
    expected_X = pd.DataFrame({"X": range(99)}).astype(float)
    expected_Y = pd.DataFrame({"Y": range(99)})  # y not transformed so datatype unchanged
    tt.assert_frame_equal(X, expected_X)
    tt.assert_frame_equal(y, expected_Y)
    print "\nFix non-numeric OK..."


def test_filtercount_realdata():
    covariates = pd.Series({
                        'Ethnicity': "facepheno.ancestry.*[a-z]$",
                        'Telomeres': "telomeres.tel_lengths.CCCTAA.k_4",
                        'ChrM copy': "mcn.ChrM_copy$",
                        'ChrM HetP': "mcn.HetpFreqSum",
                        'X copy': "dynamic_mingfu_chrX_cn",
                        'Y copy': "dynamic_mingfu_chrY_cn",
                        'ChrM PCA': "mcn.ReadDepth",
                        'Voice': "voice\.i_vectors\.1\.v1"
                        })
    annot_vars = [ros_vars.PROJECT_KEY, ros_vars.SAMPLE_KEY, 'qc.MEANCOVERAGE']
    xygen = pp.XYGenerator(pp.get_regex(list(covariates)), "%s$" % ros_vars.AGE,
                           annot_expr=pp.get_regex(annot_vars), loglevel="INFO", clean=True,
                           data_filters={ros_vars.PROJECT_KEY: [("==", "FACE")], "qc.MEANCOVERAGE": [(">", 35)]}
                           )
    X, y = xygen()
    assert (xygen.n_init - X.shape[0]) == xygen.filter_info.loc[["Missing Values", "User Filters"]].sum().sum()
    assert xygen.data.shape[0] > 500
    print "\nFilter stats OK on real data..."


def test_preprocess():
    from datastack.ml import preprocess as pp
    from datastack.dbs.rdb import RosettaDBMongo
    rdb = RosettaDBMongo(host=ros_vars.ROSETTA_URL)
    rdb.initialize(namespace='hg19')

    df = rdb.query(rdb.find_keys("pheno.weight", regex=True))

    df["pheno.weight"] *= 100
    xy = pp.XYGenerator("(mcn.ChrM_Copy)|(pheno.weight)", "pheno.age", data=df, 
                        db_namespace="hg19", loglevel="DEBUG")

    X, y = xy()
    assert (X["pheno.weight"] == df["pheno.weight"].loc[X.index]).all(), "User provided data over-written by Rosetta version"

if __name__ == "__main__":
    test_preprocess()
    test_fix_non_numeric()
    test_patsy_formula()
    test_filters()
    test_filtercount_realdata()
