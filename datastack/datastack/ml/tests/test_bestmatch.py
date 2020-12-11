"""
Copyright 2015, Human Longevity Inc. Author: M. Cyrus Maher

Test `bestmatch.py`. Right now this is not integrated with a formal testing framework.
Run using `python test_bestmatch.py`
"""

import pandas as pd
import numpy as np
from datastack.ml import bestmatch as bm
import pandas.util.testing as tt
import logging
import re
from datastack.ml import rosetta_settings as ml_vars


def simulate_mvnorm(n_vars=10, n_people=1000, correlated_errors=False, seed=47):
    mean = pd.Series(0, index=range(n_vars))

    np.random.seed = seed

    cov = pd.DataFrame(1., index=range(n_vars), columns=range(n_vars))

    for ii in cov.index:
        for jj in cov.columns:
            if ii > jj:
                if correlated_errors:
                    element = np.random.uniform(low=-1, high=1, size=1)[0]

                else:
                    element = 0
                cov.ix[ii, jj] = element
                cov.ix[jj, ii] = element

    errors = np.random.multivariate_normal(mean, cov, size=n_people)

    observed = np.random.uniform(low=0, high=2, size=((n_people, n_vars)))
    inferred = observed + errors
    return observed, inferred


def simulate_signoise(N, D, seed=None, signal_to_noise=None):
    # This code is lightly modified from:
    # https://github.com/hlids/cerebro/blob/develop/gwas/python/hlipymodels/maxent.py
    # each column in the reconstruction has a different signal to noise ratio.
    # the algorithm should weigh up the columns with higher signal to noise
    np.random.seed(seed)
    if signal_to_noise is None:
        signal_to_noise = np.random.uniform(size=D)
    X = np.random.randn(N, D)
    noise = np.random.randn(N, D)

    # the noisy reconstruction:
    X_noisy = signal_to_noise[np.newaxis, :] * X + \
        (1 - signal_to_noise[np.newaxis, :]) * noise
    # the true vector of inputs:
    return pd.DataFrame(X), pd.DataFrame(X_noisy), signal_to_noise


def sim_data(N_obs, N_vars, signal_to_noise=None, sim_kwargs={}, model_kwargs={}, resid_kwargs={}):
    X, X_noisy, signal_to_noise = simulate_signoise(N_obs, N_vars, seed=None, signal_to_noise=signal_to_noise)
    distances = bm.get_residual(X, X_noisy, **resid_kwargs)
    return distances, signal_to_noise


def test_build_graph():
    probs = pd.DataFrame({"Genome_num": [0, 0, 1, 1], "Face_num": [0, 1, 0, 1],
                          0: [.99, .01, .01, .99]})
    probs.set_index(["Genome_num", "Face_num"], inplace=True)
    probs[1] = 1 - probs[0]
    model = bm.WeightedNearestNeighbor()
    graph = model.build_graph(probs)

    assert len(graph.nodes()) == 4, "Wrong number of nodes"
    assert len(graph.edges()) == probs.shape[0], "Wrong number of edges"

    for ee in graph.edges(data=True):
        if ee[0][(model.labellen + 6):] == ee[1][(model.labellen + 6):]:
            assert ee[2]["weight"] == .99
        else:
            assert ee[2]["weight"] == .01


def test_get_resid():
    observed = pd.DataFrame({"X": [1, 2], "Y": [3, 5]})
    inferred = observed.copy()
    res = bm.get_residual(observed, inferred)

    expected = pd.DataFrame({"X": [0, 1, 1, 0],
                             "Y": [0, 2, 2, 0]},
                            index=pd.MultiIndex.from_tuples([(0, 0), (0, 1),
                                                             (1, 0), (1, 1)]),
                            columns=["X", "Y"]
                            )
    expected.index.names = ["Source_num", "Target_num"]
    tt.assert_frame_equal(res, expected)


def test_sim_data():
    distances, _ = sim_data(3, 2)
    assert distances.shape == (3 * 3, 2), "Distance matrix is the wrong size"
    distances, _ = sim_data(3, 2, resid_kwargs={"max_pairs": 6})
    assert distances.shape == (6, 2), "Distance matrix is the wrong size"


def test_weightednearest(N=30, v=5):
    X = pd.DataFrame(np.random.randn(N, v))
    X_noisy = pd.DataFrame(np.random.randn(N, v))
    model = bm.WeightedNearestNeighbor()
    X_train, y = model.get_XY(X, X_noisy)

    model.fit(X_train, y)

    X2 = pd.DataFrame(np.random.randn(N, v))
    X2_noisy = pd.DataFrame(np.random.randn(N, v))
    X_test, y_test = model.get_XY(X2, X2_noisy)
    model.predict(X_test)
    assert not np.isnan(model.score()), "Score is nan"
    assert model.score() < 2. * N / (N * (N - 1)), "Model accuracy implausibly high: %s" % model.score()


def create_eval_select(X, X_noisy, n_lineups=10, **kwargs):
    X.columns = [str(xx) + "_obs" for xx in X.columns]
    X_noisy.columns = [str(xx).replace('_obs', "_pred") for xx in X.columns]
    df = pd.concat([X, X_noisy], axis=1)
    df[ml_vars.SAMPLE_NAME] = map(str, range(df.shape[0]))
    df[ml_vars.PROJECT_KEY] = "Test"
    df[ml_vars.QUICKSILVER_KFOLDS_HOLDOUT_COLUMNS[0]] = False
    df[ml_vars.QUICKSILVER_KFOLDS_TOGETHER_COLUMNS[0]] = "[]"
    model = bm.DistSelect()
    obs_cols = [xx for xx in df.columns if "obs" in xx]
    pred_cols = [xx.replace("obs", "pred") for xx in obs_cols]
    eval_sel = bm.EvalSelect(df, model,
                             to_cols=obs_cols,
                             from_cols=pred_cols,
                             n_lineups=n_lineups,
                             loglevel=logging.DEBUG, **kwargs)
    return eval_sel


def run_eval(X, X_noisy, n_lineups=10):
    eval_sel = create_eval_select(X, X_noisy, n_lineups=n_lineups)
    # No subjects should overlap between train and test
    assert len(set(eval_sel.train_lineups.values.ravel()) &
               set(eval_sel.test_lineups.values.ravel())) == 0

    print "fitting"
    eval_sel.fit()

    print "evaluating holdout"
    accuracy_test = eval_sel.predict()
    print accuracy_test
    return accuracy_test


def test_eval_pos(N=500, v=100, n_lineups=50):
    """
    Positive control. Test accuracy with known association.
    """
    X, X_noisy, signal_to_noise = simulate_signoise(N, v)
    accuracy_test = run_eval(X, X_noisy, n_lineups=n_lineups)
    assert accuracy_test["match"] > 0.90, "Accuracy too low on low-noise data"


def test_eval_neg(N=500, v=100, n_lineups=50):
    """
    Negative control. Test accuracy with random data
    """
    X = pd.DataFrame(np.random.randn(N, v))
    X_noisy = pd.DataFrame(np.random.randn(N, v))

    accuracy_test = run_eval(X, X_noisy, n_lineups=n_lineups)
    print accuracy_test
    assert accuracy_test.max() < .2, "Accuracy too high on random data"


def eval_gridsearch(N=500, v=10):
    """
    Check whether we get the same results with or without gridsearch directly

    """
    print "Evaluating gridsearch"
    from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
    from sklearn.grid_search import GridSearchCV
    X, X_noisy, signal_to_noise = simulate_signoise(N, v)

    X.columns = [str(xx) + "_obs" for xx in X.columns]
    X_noisy.columns = [str(xx).replace('_obs', "_pred") for xx in X.columns]

    df = pd.concat([X, X_noisy], axis=1)
    df[ml_vars.SAMPLE_KEY] = map(str, range(df.shape[0]))
    df[ml_vars.PROJECT_KEY] = "Test"

    Cs = [.01, .1, 1, 10, 100]
    LRCV_args = {"model": GridSearchCV,
                 "model_args": [LogisticRegression(class_weight="auto"),
                                {"C": Cs}],
                 "model_kwargs": {"n_jobs": 4, "verbose": 2}
                 }
    LR_args = {"model": LogisticRegressionCV, "model_kwargs": {"Cs": Cs}}

    model1 = bm.WeightedNearestNeighbor(**LRCV_args)
    model2 = bm.WeightedNearestNeighbor(**LR_args)

    eval_sel1 = bm.EvalSelect(df, model1)
    eval_sel2 = bm.EvalSelect(df, model2)

    print "fitting..."
    eval_sel1.fit(X.columns, X_noisy.columns)
    eval_sel2.fit(X.columns, X_noisy.columns)

    print "predicting..."
    accuracy1 = eval_sel1.predict()
    accuracy2 = eval_sel2.predict()
    assert (accuracy1 - accuracy2).max() < .01, "Grid search yielded discordant results"


def test_predictions(model=bm.DistSelect(metric="cosine"), impute=True, normalize=False, group="skin"):
    best_matches = pd.read_table("../data_local/pred_to_obs_colnames.txt", sep="\t", index_col=0)
    best_matches = best_matches[best_matches["Group"].str.contains(group)]

    df = pd.read_table("../data_local/select_df.txt", sep="\t")
    
    pred_cols = [xx for xx in df.columns if (xx in best_matches.index)]

    obs_cols = list(best_matches.ix[pred_cols, "Best Match"].values)

    eval_sel = bm.EvalSelect(df, model,
                             to_cols=obs_cols, from_cols=pred_cols, 
                             impute=impute, normalize=normalize)
    eval_sel.fit()
    return eval_sel.predict()


def test_unpacking(n_depthPCs=20, namespace="hg38_noEBV"):
    """
    Test whether packed columns in Rosetta are accurately unpacked and matched up
    """
    def unpack_df(bm_small, db, **kwargs):
        cols = list(bm_small.index) + list(bm_small["Best Match"])
        df = db.query(cols)
        packed_df = mutil.PackedFrame(df, **kwargs)
        df = packed_df.unpack()
        bm_final = packed_df.update_colmaps(bm_small, **kwargs)
        return df, bm_final
    from datastack.dbs.rdb import RosettaDBMongo
    db = RosettaDBMongo(host="rosetta.hli.io")
    db.initialize(namespace=namespace)

    best_matches = pd.read_table("../data_local/visit1_to_visit2_colnames.txt", sep="\t", index_col=0)
    
    bm_small = best_matches[(best_matches["Group"] == "face.v6") & (~best_matches["Best Match"].str.contains("mirror"))]
    df_v1_v2, bm_final_v1_v2 = unpack_df(bm_small, db,
                                         from_str="visit1", to_str="visit2", 
                                         maxElements={"ColorPC": n_depthPCs, "DepthPC": n_depthPCs})
    
    assert pd.Series(bm_final_v1_v2.index).str.contains("visit1").all(), "From column is wrong format"
    assert bm_final_v1_v2["Best Match"].str.contains("visit2").all(), "To column is wrong format"
    assert (bm_final_v1_v2["Score"] < 100).sum() == 0, "Did not retrieve exact matches"
    assert bm_final_v1_v2.shape[0] == 2 * n_depthPCs, "Got the wrong number of mappings. Expected %s, got %s" % (2 * n_depthPCs, bm_final_v1_v2.shape[0])
    print "Passed test 1"
    # Observed/predicted test
    best_matches = pd.read_table("../data_local/pred_to_obs_colnames.txt", sep="\t", index_col=0)
    bm_small = best_matches.filter(like="Depth", axis=0)

    df_obs_pred, bm_final = unpack_df(bm_small, db, from_str=".FACE_P.", to_str=".FACE.", 
                                          maxElements={"DepthPC": n_depthPCs})
    assert pd.Series(bm_final.index).str.contains(".FACE_P.").all(), "From column is wrong format"
    assert bm_final["Best Match"].str.contains(".FACE.").all(), "To column is wrong format"
    assert (bm_final["Score"] < 100).sum() == 0, "Did not retrieve exact matches"
    assert bm_final.shape[0] == n_depthPCs, "Got the wrong number of mappings. Expected %s, got %s" % (n_depthPCs, bm_final.shape[0])
    print "Passed test 2"
    print "Passed all unpacking tests"


def test_specify_index():
    X = pd.DataFrame(np.random.randn(1000, 10))
    X_noisy = pd.DataFrame(np.random.randn(1000, 10))
    train_inds = range(600)[::2]
    test_inds = range(600)[1::2]
    eval_select = create_eval_select(X, X_noisy, n_lineups=10, train_inds=train_inds)
    assert (eval_select.train_inds == train_inds).all(), "Train index specification is broken"
    
    eval_select = create_eval_select(X, X_noisy, n_lineups=10, test_inds=train_inds)
    assert (eval_select.test_inds == train_inds).all(), "Test index specification is broken"
    
    eval_select = create_eval_select(X, X_noisy, n_lineups=10, train_inds=train_inds, test_inds=test_inds)
    assert (eval_select.test_inds == test_inds).all() and (eval_select.train_inds == train_inds).all(), "Train/Test split specification is broken"
    print "Passed specify index tests"


def test_grouping():
    # df = pd.DataFrame({"X1": [0],
    #                    "X2": [0],
    #                    "Y": [0],
    #                    "p.X1": [1],
    #                    "p.X2": [-1],
    #                    "p.Y": [0]})
    # obs_cols = ["X1", "X2", "Y"]
    # pred_cols = ["p.X1", "p.X2", "p.Y"]
    # groups = ["X", "X", "Y"]

    # # Test grouping on WeightedNearestNeighbor
    # model_wnn = bm.WeightedNearestNeighbor()
    # distances, discordant = model_wnn.get_XY(df[pred_cols], df[obs_cols], col_groups=groups)
    # expected = pd.DataFrame({"X": [1], "Y": [0]}, index=pd.MultiIndex.from_tuples([(0, 0)]))
    # expected.index.names = ["Source_num", "Target_num"]
    # tt.assert_frame_equal(distances, expected)
    
    # # Test grouping on DistSelect
    # model_ds = bm.DistSelect(metric="cityblock")
    # distances_ds, discordant = model_ds.get_XY(df[pred_cols], df[obs_cols], col_groups=groups)
    # assert (distances.sum(axis=1) == distances_ds["value"]).all(), "DistSelect and WNN grouped distances don't match"

    # # Test no grouping on WNN
    # distances, discordant = model_wnn.get_XY(df[pred_cols], df[obs_cols], col_groups=None)
    # expected = pd.DataFrame({"X1": [1], "X2": [1], "Y": [0]}, index=pd.MultiIndex.from_tuples([(0, 0)]))
    # expected.index.names = ["Source_num", "Target_num"]
    # tt.assert_frame_equal(distances, expected)

    # # Test no grouping DistSelect
    # distances_ds, discordant = model_ds.get_XY(df[pred_cols], df[obs_cols], col_groups=None)
    # assert (distances.sum(axis=1) == distances_ds["value"]).all(), "DistSelect and WNN grouped distances don't match"

    df = pd.DataFrame({"X1": [0., 2.],
                       "X2": [0., 2.],
                       "Y": [0., 2.],
                       "p.X1": [1., 2.],
                       "p.X2": [-1., 0.],
                       "p.Y": [0., 1.]})
    obs_cols = ["X1", "X2", "Y"]
    pred_cols = ["p.X1", "p.X2", "p.Y"]
    groups = ["X", "X", "Y"]

    expected = pd.DataFrame(np.nan, index=pd.MultiIndex.from_product([df.index, df.index]), columns=np.unique(groups))
    expected.index.names = ["Source_num", "Target_num"]
    
    for ii in df.index:
        for jj in df.index:
            expected.loc[(ii, jj)] = pd.Series(np.abs(df.loc[ii, pred_cols].values - df.loc[jj, obs_cols].values)).groupby(groups).mean()

    expected_nogroups = pd.DataFrame(np.nan, index=pd.MultiIndex.from_product([df.index, df.index]), columns=pred_cols)
    expected_nogroups.index.names = ["Source_num", "Target_num"]
    
    for ii in df.index:
        for jj in df.index:
            expected_nogroups.loc[(ii, jj)] = pd.Series(np.abs(df.loc[ii, pred_cols].values - df.loc[jj, obs_cols].values)).values  
    
    # Test grouping on WNN
    model_wnn = bm.WeightedNearestNeighbor()
    distances, discordant = model_wnn.get_XY(df[pred_cols], df[obs_cols], col_groups=groups)
    tt.assert_frame_equal(distances, expected)

    # Test no grouping on WNN
    distances_ng, discordant = model_wnn.get_XY(df[pred_cols], df[obs_cols], col_groups=None)
    tt.assert_frame_equal(distances_ng, expected_nogroups)

    # Test grouping on DistSelect
    model_ds = bm.DistSelect(metric="cityblock")
    distances_ds, discordant = model_ds.get_XY(df[pred_cols], df[obs_cols], col_groups=groups)

    assert (distances.sum(axis=1).loc[distances_ds.index] == distances_ds["value"]).all(), "DistSelect and WNN grouped distances don't match"

    distances_ng_ds, discordant = model_ds.get_XY(df[pred_cols], df[obs_cols], col_groups=None)
    assert (distances_ng.sum(axis=1).loc[distances_ng_ds.index] == distances_ng_ds["value"]).all(), "DistSelect and WNN grouped distances don't match"

if __name__ == "__main__":
    test_grouping()
    test_specify_index()
    print "Testing positive"
    test_eval_pos()
    print "Testing negative"
    test_eval_neg()

    test_sim_data()
    test_get_resid()
    test_build_graph()
    print "All tests passed"
