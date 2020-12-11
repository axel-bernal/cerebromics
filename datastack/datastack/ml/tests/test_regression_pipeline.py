import sys
sys.path.extend(["../" * xx for xx in range(1, 4)])
import pandas as pd
import numpy as np
import datastack.ml.model_utils as mu
from sklearn.linear_model import LinearRegression
from pandas.util import testing as tt
import datastack.ml.preprocess as pp
from collections import OrderedDict

def make_df():
    np.random.seed(47)

    n_pts = 100

    n_missing_x = 5
    n_missing_y = 3

    n_outlier_x = 2
    n_outlier_y = 3
    n_nan_v = 1

    secular_trend = np.arange(n_pts) / (n_pts / 2.)

    v = pd.Series(["Male"] * 9 + [np.nan] * n_nan_v + ["Female"] * (n_pts - 12) + ["Warning"] * 2)
    w = pd.Series(index=range(n_pts))

    x = secular_trend + np.random.randn(n_pts)
    x[2:(2 + n_outlier_x)] = x[2:(2 + n_outlier_x)] + np.random.normal(loc=500, scale=20, size=n_outlier_x)
    x[15:(15 + n_missing_x)] = np.nan

    y = 2 * secular_trend
    y[10:(10 + n_missing_y)] = np.nan
    y[6:(6 + n_outlier_y)] = y[6:(6 + n_outlier_y)] + np.random.normal(loc=500, scale=20, size=n_outlier_y)

    df = pd.DataFrame({
                      "V": v,
                      "W": w,
                      "X": x,
                      "Y": y,
                      "Z": np.arange(n_pts) + np.random.randn(n_pts)
                      })
    # The legacy baseline regression algorithms require the
    # 'ds.index.sample_name' column, so simulate one.
    df['ds.index.sample_name'] = list(df.index)

    cols = ["Outliers (trim)", "Missing Values", "User Filters", "Dropped", "Drop Reason"]
    x_vals = [n_outlier_x, n_missing_x, 0, False, "N/A"]
    y_vals = [n_outlier_y, n_missing_y, 0, False, "N/A"]
    w_vals = [np.nan, np.nan, np.nan, True, "%s%% missing" % 100.]
    v_vals = [0, 1, 2, False, "N/A"]
    v_orig_vals = [0, 1, 2, True, "Recoding"]
    expected = pd.DataFrame({"X": x_vals,
                             "Y": y_vals,
                             "W": w_vals,
                             "V[T.Male]": v_vals,
                             "V": v_orig_vals},
                            index=cols)
    expected.columns.name = "Column name"
    return df, expected


def test_var_update():

    df, expected = make_df()
    # Define some options
    xygen_kwargs = {
        "data": df,
        "loglevel": "DEBUG",
        "trim_outliers": True,
        "outlier_stdevs": 4,
        "data_filters": {"V": [("!=", "Warning")]}
    }
    # Specify covariates
    covariates = ["X", "W", "V"]

    # Run pipeline
    pipeline = mu.CumulativeRegression(covariates, "Z",
                                       xygen_kwargs=xygen_kwargs,
                                       model_names={"Linear Regression": LinearRegression()})

    # Test baseline pipeline
    fi = pipeline.filter_info
    print fi
    # pipeline.plot("X")
    # import matplotlib.pyplot as plt
    # plt.show()
    pipeline.run()
    #pipeline.plot_fit("Linear Regression", "indiv", "V")
    #import matplotlib.pyplot as plt
    #plt.show()
    tt.assert_frame_equal(fi, expected.T.loc[fi.index])

    # Test appending variables
    pipeline.add_covariate(10, "Y")
    fi = pipeline.filter_info
    pipeline.run()

    tt.assert_frame_equal(fi, expected.T.loc[fi.index])

    res = pipeline.results.ix[0, :, :].filter(regex="^R2.*((indiv)|(cumm))")
    #assert (res.loc["Y"] > 0).all(), "R2 for Y should be positive"


def test_real_data():
    covariates = pd.Series(OrderedDict([
        ('Ethnicity', "facepheno.ancestry.*[a-z]$"),
        ('ChrM HetP', "mcn.HetpFreqSum"),
        ('ChrM copy', "mcn.ChrM_copy$"),
        ('Y copy', "dynamic.mingfu.chrY_cn"),
        ('X copy', "dynamic.mingfu.chrX_cn"),
        ('ChrM PCA', "mcn.ReadDepth"),
        ('Telomeres', "telomeres.tel_lengths.CCCTAA.k_4"),
        ('Voice', ".*i_vectors\.1\.v1")
        ])
    )

    xygen_kwargs = {
        "data_filters": {"ds.index.ProjectID": [("==", "FACE")], "qc.MEANCOVERAGE": [(">=", 35)]},
        "annot_expr": pp.get_regex(["ds.index.ProjectID", "ds.index.sample_name", "qc.MEANCOVERAGE"]),
        "loglevel": "DEBUG"
    }
    kfold_kwargs = {'index_column': 'ds.index.sample_name'}
    # Used to use "pheno.HLI_CALC_Age_Sample_Taken$" but that key doesn't
    # exist any more.
    outcome = "facepheno.Age$"
    pipeline = mu.CumulativeRegression(covariates, outcome, xygen_kwargs=xygen_kwargs)
    assert int(pipeline.xygen.n_init - pipeline.filter_info[["Missing Values", "User Filters"]].sum().sum()) == pipeline.data.shape[0]
    # Following check deleted since we can't be sure how many records the
    # modified outcome key will retrieve. 
    #assert pipeline.data.shape[0] >= 655

    pipeline.run()


def test_missing_vs_filter():
    df = pd.DataFrame({"X": [1, 2, np.nan, np.nan], "Y": ["Keep", "Keep", "Filter", "Filter"], "Z": [1, 2, 3, 4]})
    cols = ["Outliers (warn)", "Missing Values", "User Filters", "Dropped", "Drop Reason"]

    x_vals = [0, 0, 0, False, "N/A"]
    y_vals = [0, 0, 2, False, "N/A"]
    expected = pd.DataFrame({"X": x_vals, "Y": y_vals}, index=cols)
    xygen_kwargs = {
        "data": df,
        "loglevel": "DEBUG",
        "data_filters": {"Y": [("!=", "Filter")]}
    }
    pipeline = mu.CumulativeRegression(["X"], "Z",
                                       xygen_kwargs=xygen_kwargs,
                                       model_names={"Linear Regression": LinearRegression()})

    # No missing values should be reported since those are caught by the user filter
    fi = pipeline.filter_info
    tt.assert_frame_equal(fi, expected.T.loc[fi.index])


def test_other_real():
    covariates = pd.Series(OrderedDict([
                          ('Ethnicity', "facepheno.ancestry.*[a-z]$"),
                          ('ChrM HetP', "mcn.HetpFreqSum"),
                          ('ChrM copy', "mcn.ChrM_copy$"),
                          ('Y copy', "dynamic.mzhu.chrY_cn"),
                          ('X copy', "dynamic.mzhu.chrX_cn"),
                          ('ChrM PCA', "mcn.ReadDepth"),
                          ('Telomeres', "telomeres.tel_lengths.CCCTAA.k_4"),
                          ('Voice', ".*i_vectors\.1\.v1")
        ])
    )
    xygen_kwargs = {
        "data_filters": {"ds.index.ProjectID": [("==", "FACE")], "qc.MEANCOVERAGE": [(">=", 25)]},
        "annot_expr": pp.get_regex(["ds.index.ProjectID", "ds.index.sample_name", "qc.MEANCOVERAGE"]),
        "loglevel": "DEBUG",
        "trim_outliers": True
    }
    kfold_kwargs = {'index_column': 'ds.index.sample_name'}
    outcome = "facepheno.Age$"
    pipeline = mu.CumulativeRegression(covariates, outcome, xygen_kwargs=xygen_kwargs, kfold_kwargs=kfold_kwargs)
    pipeline.run()
    # pipeline.plot_fit("Linear Regression", "indiv", "facepheno.ancestry.*[a-z]$")
    # import matplotlib.pyplot as plt
    # plt.show()

if __name__ == "__main__":
    test_var_update()
    test_real_data()
    test_missing_vs_filter()
    test_other_real()
