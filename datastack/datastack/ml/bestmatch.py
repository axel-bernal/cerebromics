"""
Copyright 2015, Human Longevity Inc. Author: M. Cyrus Maher

Implement best match prediction based on observed and predicted values.

This works in three stages:
    1.) (optional) Learn weights on input features using known training examples
    2.) For unknown pairs, predict the probability of a match using the fitted model from 1
        -Or calculate distances, which are converted to similarities
    3.) Similarities from 2. are then inserted as edge weights into a bipartite graph
    4.) We then call pairs based, maximizing the sum of edge weights (global maximum likelihood)

This approach performs better than generating an independent guess
for each person

Common distance metrics can be found here:
http://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.spatial.distance.cdist.html


On top of the basic framework is some code to easily run match/select experiments over large numbers of variables and model_names
This is implemented in the `EvalSelectPanel` class. 

Classes:
EvalSelectPanel: Runs EvalSelect over an array of model_names and variable sets

EvalSelect: Evaluates Select@N performance
    - _GraphMatcher: Generic class for Select@N methods
        - DistSelect: Perform distance-based selection (unweighted across features)
        - WeightedNearestNeighbor: Perform probability-based selection (weights features)
"""
from datastack.ml import cross_validation as cv
from datastack.dbs.rdb import RosettaDBMongo
import numpy as np
import sys
import pandas as pd
import itertools
from sklearn.linear_model import LogisticRegressionCV
import networkx as nx
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import random
import string
import logging
import os
from multiprocessing import Pool
from datastack.ml import match_utils as mu
import model_utils as mod
import re
import rosetta_settings as ml_vars
import sklearn

assert float(".".join(sklearn.__version__.split(".")[:2])) >= 0.17, "bestmatch.py requires sklearn version 0.17 or great. Please `conda install scikit-learn`"


def runfunc(tup):
    """
    Must accept a single argument to be used with `map` or `Pool.map`

    Args:
        tup: A tuple of (variable set, method, `EvalSelectPanel` instance)

    Run match/select@N using a given model on a specified column set. 
    """
    ss, mm, self = tup
    print "Running:", ss, mm

    model = self.model_names[mm](**self.model_kwargs[mm])
    todo = self.df[self.from_col_dict[ss] + self.to_col_dict[ss] + self.key_cols]

    print todo.shape, len(self.from_col_dict[ss]), len(self.to_col_dict[ss])
    eval_sel = EvalSelect(todo, model,
                          from_cols=self.from_col_dict[ss],
                          to_cols=self.to_col_dict[ss],
                          **self.eval_kwargs
                          )
    eval_sel.fit()
    print "Done fitting..."
    p1 = eval_sel.predict()
    p2 = eval_sel.predict(sampset="train")
    return {(ss, mm): {"test": p1, "train": p2}}


class EvalSelectPanel:
    """
    Given data, model_names, and column sets, run an array of match/select@10 experiments in parallel
    """
    def __init__(self, df, model_names=None, model_kwargs=None,
                 model_order=None,
                 from_col_dict=None,
                 to_col_dict=None,
                 set_order=None,
                 n_jobs=4,
                 eval_kwargs=None,
                 key_cols=[ml_vars.SAMPLE_NAME, ml_vars.PROJECT_KEY],
                 async=True
                 ):
        """
        Args:
            df:            A pandas dataframe

        Kwargs:
            model_names:        A dictionary of uninstantiated match/select model_names
            model_kwargs:  A dictionary of kwarg dictionaries to pass to each model
            from_col_dict: A dictionary mapping a "from" column set name to a list of columns
            to_col_dict:   A dictionary mapping a "to" column set name to a list of columns 
                                in order corresponding to that in `from_col_dict`
            set_order:     An optional list specifying the order in which to loop through "from"/"to" column sets
            n_jobs:        The number of jobs to launch in parallel
            eval_kwargs:   Kwargs to pass to `EvalSelect`
            key_col:       Columns that may be used as sample/subject keys
        """
        self.n_jobs = n_jobs
        self.df = df.copy()

        self.key_cols = key_cols
        self.async = async

        assert from_col_dict is not None
        assert to_col_dict is not None

        self.from_col_dict = from_col_dict
        self.to_col_dict = to_col_dict
        if model_names is None:
            self.model_names = {
                "LogisticReg": WeightedNearestNeighbor,
                "MinEucDist": DistSelect,
                "MinCosDist": DistSelect,
                "MinManDist": DistSelect
            }
        else:
            self.model_names = model_names
        if model_kwargs is None:
            self.model_kwargs = {
                "LogisticReg": {"model": LogisticRegressionCV,
                                "model_kwargs": {"class_weight": "balanced",
                                                 "n_jobs": 1}
                               },
                "MinEucDist": {},
                "MinCosDist": {"metric": "cosine"},
                "MinManDist": {"metric": "cityblock"},
            }
        else:
            self.model_kwargs = model_kwargs

        if eval_kwargs is None:
            self.eval_kwargs = {"impute": True, "normalize": True}
        else:
            self.eval_kwargs = eval_kwargs

        if set_order is None:
            self.set_order = from_col_dict.keys()
        else:
            self.set_order = set_order

        if model_order is None:
            self.model_order = self.model_names.keys()
        else:
            self.model_order = model_order

    def run(self):
        all_combos = [(xx, yy, self) for xx in self.set_order for yy in self.model_order]

        if self.async:
            pool = Pool(self.n_jobs)
            reslist = pool.map(runfunc, all_combos)
            self.reslist = reslist
        else:
            reslist = []
            for aa in all_combos:
                reslist.append(runfunc(aa))
            self.reslist = reslist
        return reslist


class EvalSelect:
    """
    Evaluate a Select@N method.
    """
    def __init__(self, df, model, to_cols=None, from_cols=None, select_n=10, n_lineups=100,
                 seed=47, loglevel=logging.INFO, impute=False, normalize=False,
                 test_lineups=None, train_lineups=None, 
                 anno_cols=ml_vars.QUICKSILVER_METADATA_KEYS + [ml_vars.SAMPLE_NAME, ml_vars.PROJECT_KEY], 
                 train_inds=None, test_inds=None, dump_matrices=False, col_groups=None,
                 dump_path="%s/data_local/match_debug" % os.path.dirname(__file__)):
        """
        df:        A dataframe. Must have `qc.sample_key` and `lab.ProjectID` columns
        model:     The select/match@N method to evaluate.
                       - Must have `fit` and `predict`, and `get_XY` methods.
        select_n:  The lineup size to use
        n_lineups: The number of lineups to produce
        seed:      The random seed for lineup generation
        loglevel:  Amount of info to report. See python module `logging`
        """
        assert to_cols is not None, "Please specify observed columns"
        assert from_cols is not None, "Please specify predicted columns"
        logging.basicConfig(level=loglevel)
        
        if model.__class__ == DistSelect:
            self.train_model = False
        else:
            self.train_model = True

        n_lineups = min(n_lineups, df.shape[0])

        self.model = model
        self.df = df.copy()
        self.dump_matrices = dump_matrices
        self.dump_path = dump_path
        self.col_groups = col_groups

        assert len(to_cols) > 0, "Need at least one 'to col'"
        assert len(from_cols) > 0, "Need at least one 'from col'"

        self.to_cols = to_cols
        self.from_cols = from_cols
        self.train_inds = train_inds
        self.test_inds = test_inds
        self.anno_cols = anno_cols if (train_inds is None and test_inds is None) else []

        if normalize:
            for xx in to_cols + from_cols:
                # Convert to mean zero, std of one
                self.df.loc[:, xx] = (self.df.ix[:, xx] - self.df.ix[:, xx].mean()) / self.df.ix[:, xx].std()
        
        if impute:
            self.df = self.df[self.to_cols + self.from_cols + self.anno_cols]
            
            # Keep track of all-null values so you can impute their similarities later
            # This greatly increases accuracy.
            self.to_allnull = self.df[to_cols].isnull().all(axis=1).nonzero()[0]
            self.from_allnull = self.df[from_cols].isnull().all(axis=1).nonzero()[0]

            if normalize:
                # If normalized, the mean should be zero, so just fill in with exactly zero.
                # Filling with the column mean (which is nearly zero but not *exactly*) can lead to some 
                # numerical strangeness, particularly with the cosine distance 
                self.df.loc[:, self.to_cols + self.from_cols] = self.df[to_cols + from_cols].fillna(0.)
            else:
                self.impute(self.df, to_cols + from_cols)
        else:
            self.df = self.df[self.to_cols + self.from_cols + self.anno_cols].dropna()
            self.to_allnull = []
            self.from_allnull = []
        
        # Test whether the output makes sense
        assert self.df.shape[0] > 0, "No valid rows in data frame"
        try:
            assert (self.df[to_cols + from_cols].isnull().sum(axis=1) == 0).all(), "Null values in dataset"
        except Exception as e1:
            print e1
            print model
            print from_cols
            sys.exit()
        assert self.df.index.is_unique, "Dataframe index must be unique"
        
        # Get lineups
        self.get_lineups(n_lineups, select_n, seed)

        assert pd.Series(np.ravel(self.train_lineups)).isin(self.df.index).all(), "Some train lineups not in index"
        assert pd.Series(np.ravel(self.test_lineups)).isin(self.df.index).all(), "Some train lineups not in index"
        print "Done fetching lineups"

    def get_lineups(self, n_lineups, select_n, seed):
        """
        Retrieve match/select lineups of the specified size using the specied random seed. 
        """
        # Match train and test splits to those in Rosetta
        if (self.train_inds is None) and (self.test_inds is None):
            if ml_vars.SAMPLE_NAME not in self.df.columns and self.df.index.name == ml_vars.SAMPLE_NAME:
                self.df[ml_vars.SAMPLE_NAME] = self.df.index
            assert ml_vars.SAMPLE_NAME in self.df.columns, "For KFoldPredefined, `%s` must be in the dataframe columns or as the labeled index" % ml_vars.SAMPLE_NAME
            
            if ml_vars.PROJECT_KEY_OLD not in self.df.columns and ml_vars.PROJECT_KEY in self.df.columns:
                self.df[ml_vars.PROJECT_KEY_OLD] = self.df[ml_vars.PROJECT_KEY] 
            assert ml_vars.PROJECT_KEY_OLD in self.df.columns 
            kfold = cv.KFoldPredefined(data=self.df.copy())
            train_data = kfold.df
            test_data = kfold.df_holdout
        else:
            if self.train_inds is None:
                self.train_inds = self.df.index.difference(self.test_inds)
            if self.test_inds is None:
                self.test_inds = self.df.index.difference(self.train_inds)
            train_data = self.df.loc[self.train_inds]
            test_data = self.df.loc[self.test_inds]

        np.random.seed(seed)

        if (test_data.shape[0] < n_lineups):
            print "Not enough data for number of lineups. Reducing to %s" % test_data.shape[0]
            n_lineups = test_data.shape[0]
        self.train_inds = train_data.index
        self.test_inds = test_data.index

        self.train_lineups = pd.DataFrame(self.train_inds[0], index=range(n_lineups), columns=range(select_n))
        self.test_lineups = pd.DataFrame(self.test_inds[0], index=range(n_lineups), columns=range(select_n))

        # want random subsample same size as test set
        train_inds_same_size = np.random.choice(train_data.index,
                                                size=min(train_data.shape[0], test_data.shape[0]),
                                                replace=False)

        # Choose lineups
        print "Getting lineups"
        for xx in range(n_lineups):
            train_query = self.train_inds[0]
            test_query = self.test_inds[0]
            train_start = True
            test_start = True
            # Make sure queries are not repeated
            while train_start or ((self.train_lineups.iloc[:, 0] == train_query).sum() > 0):
                train_lineup = np.random.choice(train_inds_same_size, size=select_n, replace=False)
                train_query = train_lineup[0]
                train_start = False

            while test_start or ((self.test_lineups.iloc[:, 0] == test_query).sum() > 0):
                test_lineup = np.random.choice(self.test_inds, size=select_n, replace=False)
                test_query = test_lineup[0]
                test_start = False

            self.train_lineups.loc[xx] = train_lineup
            self.test_lineups.loc[xx] = test_lineup

    @staticmethod
    def impute(df, cols, **kwargs):
        """
        Replace missing values with the column mean
        """
        for xx in cols:
            inds = df[xx].isnull()
            inds = inds[inds].index
            df.loc[inds, xx] = df[xx].mean()

    def fit(self):
        """
        Fits Select@N model to training data
        to_cols: Column names for set 1
        from_cols: Column names for set 2
        """
        if self.train_model:
            logging.info("Fitting model")

            X_to_train = self.df.ix[self.train_inds, self.to_cols]
            X_from_train = self.df.ix[self.train_inds, self.from_cols]
            X_train, y = self.model.get_XY(X_from_train, X_to_train, col_groups=self.col_groups)

            logging.info("Training model on %s columns" % X_train.shape[1])

            self.model.fit(X_train, y)
        else:
            logging.info("No fit necessary (unsupervised method)")

    def predict(self, sampset="test", **kwargs):
        """
        Predicts matches on test data
        sampset: Can be `test` or `train` to estimate effect of overfitting
        """
        logging.debug("predicting...")
        lineups = self.test_lineups if sampset == "test" else self.train_lineups
        inds = np.unique(lineups.values.ravel())
        assert pd.Series(inds).isin(self.df.index).all(), "Some lineup members not in dataframe"
        X_to = self.df.loc[inds, self.to_cols]
        X_from = self.df.loc[inds, self.from_cols]

        X, y = self.model.get_XY(X_from, X_to, col_groups=self.col_groups)
        logging.info("Training model on %s columns" % X.shape[1])

        # Handle the case when you have entirely imputed rows in both the "from" or "to" set
        if (len(self.from_allnull) > 0) or (len(self.to_allnull) > 0):
            print "Imputing distances"
            meanval = X.iloc[:, 0].mean()
            from_inds = [ii for ii in self.from_allnull if (ii in inds)]
            to_inds = [jj for jj in self.to_allnull if (jj in inds)]

            from_locs = pd.Series([xx in from_inds for xx, yy in X.index.values])
            to_locs = pd.Series([yy in to_inds for xx, yy in X.index.values])
            X.loc[(to_locs | from_locs).values, :] = meanval
            print "Done imputing distances"

        # With distances fixed, run through and predict lineups
        accuracy = pd.DataFrame(np.nan, index=lineups.index, columns=["match", "select", "select_unique", "match_query", "select_query"])
        match_pred_graph = []
        match_pred_naive = []
        
        if self.dump_matrices:
            if not os.path.exists(self.dump_path):
                os.makedirs(self.dump_path)

        for tt, row in lineups.iterrows():
            if (tt % 10) == 0:
                logging.debug(tt)
            subset = [(xx, yy) for xx, yy in X.index.values if (xx in row.values) and (yy in row.values)]
            self.model.predict(X.loc[subset], **kwargs)

            accuracy.loc[tt, "match"] = self.model.score(scoretype="match")
            accuracy.loc[tt, "select"] = self.model.score(scoretype="select")
            accuracy.loc[tt, "select_unique"] = self.model.score(scoretype="select_unique")

            # For analysis purposes, dump similarity matrices if requested
            if self.dump_matrices:
                suff = "OK" if accuracy.ix[tt, "match"] >= accuracy.ix[tt, "select_unique"] else "FAIL"
                self.model.similarity.to_csv("%s/lineup%s_%s.txt" % (self.dump_path, tt, suff), sep="\t")
                score_df = pd.merge(self.model.match_pred, 
                                    self.model.match_pred_simonly_unique, 
                                    left_on="Source_num", right_on="Source_num", 
                                    suffixes=["_m", "_u"]).rename(columns={"Similarity": "Similarity_u"})
                score_df["Similarity_m"] = self.model.similarity.loc[zip(score_df["Source_num"], score_df["Target_num_m"])].values
                assert score_df["Similarity_m"].sum() >= score_df["Similarity_u"].sum(), "Suboptimal choice!"
                score_df.to_csv("%s/lineup_choices_%s_%s.txt" % (self.dump_path, tt, suff), sep="\t")
            
            test_graph = self.model.match_pred.set_index("Source_num")
            test_naive = self.model.match_pred_simonly.set_index("Source_num")

            query = str(row[0])
            accuracy.loc[tt, "match_query"] = (test_graph.loc[query, "Target_num"] == query) + 0
            accuracy.loc[tt, "select_query"] = (test_naive.loc[query, "Target_num"] == query) + 0
            match_pred_graph.append(test_graph.loc[[query]])
            match_pred_naive.append(test_naive.loc[[query]])
        self.accuracy = accuracy.mean(axis=0)
        self.match_pred_graph = pd.concat(match_pred_graph, axis=0)
        self.match_pred_naive = pd.concat(match_pred_naive, axis=0)
        return self.accuracy


class _GraphMatcher:
    """
    Based on a similarity metric, build a bipartite graph and
    pick the maximum weight matching. Alternatively, take individual best guesses
    """
    def __init__(self, opt_kwargs={}):
        self.opt_kwargs = opt_kwargs
        self.all_similarities = None

    def opt_graph(self, similarity):
        self.graph = self.build_graph(similarity)
        res = nx.max_weight_matching(self.graph, maxcardinality=True, **self.opt_kwargs)
        # The result dictionary includes mapping in both directions,
        # so filter to only the (Source -> Target) relations, then get the
        # indices back out from the uniquified node labels
        match_pred = pd.DataFrame(map(lambda x: [x[0][(self.labellen + 6):], x[1][(self.labellen + 6):]],
                                      filter(lambda x: "Source" in x[0], res.items())
                                      ),
                                  columns=["Source_num", "Target_num"]
                                  )
        return match_pred

    def calc_maxsim_match(self, similarity):
        # We have a hierarchical index,
        # so idxmax returns a tuple array that
        # needs to converted to a dataframe
        tmp = similarity.iloc[:, 0].groupby(level=0).idxmax().values.ravel()
        return pd.DataFrame(map(list, tmp),
                            columns=["Source_num", "Target_num"]
                            )

    def calc_maxsim_match_unique(self, similarity):
        dist_col = similarity.columns[0]
        df = similarity.reset_index()
        cols = list(df.columns)
        cols[:2] = ["Source_num", "Target_num"]
        df.columns = cols
        hitlist = []
        while len(df) > 0:
            best_match = df[dist_col].argmax()
            source_num = df.loc[best_match, "Source_num"]
            target_num = df.loc[best_match, "Target_num"]
            hitlist.append((source_num, target_num, df.loc[best_match, dist_col]))
            df = df[[(xx != source_num) and (yy != target_num) 
                    for (xx, yy) in zip(df["Source_num"], df["Target_num"])
                   ]]
        return pd.DataFrame(hitlist, columns=["Source_num", "Target_num", "Similarity"])
    
    def predict(self, X, norm_sim=False, annot_df=None):
        """
        X: A dataframe containing features from which to predict matches.
        Produces tables of predicted matches
        """
        if not (X.isnull().sum(axis=1) == 0).all():
            logging.info("Distance matrix contains null values")
            X.iloc[:, 0] = X.iloc[:, 0].fillna(X.iloc[:, 0].mean())
        
        if self.metric_kind == "distance":
            self.similarity = pd.DataFrame({"value": X.sum(axis=1)})
        else:
            if self.all_similarities is None:
                # Get the probability the each pair is a match
                probs = self.model.predict_proba(X)
                self.similarity = pd.DataFrame(probs, index=X.index)
            else:
                self.similarity = self.all_similarities.loc[X.index]
        
        if norm_sim:
            for kk, xx in self.similarity.groupby(level=0):
                tot = xx.iloc[:, 0].sum()
                for ii in xx.index:
                    self.similarity.loc[ii] /= tot
        assert (self.similarity.isnull().sum(axis=1) == 0).all(), "Found NaN for distance"

        if annot_df is not None:
            fill_val = self.similarity.loc[:, "value"].max() if self.metric_kind == "distance" else 0
            ii, jj = zip(*X.index.values)
            zero_out = (annot_df.loc[list(ii)] != annot_df.loc[list(jj)].values).any(axis=1)
            self.similarity[zero_out.values] = fill_val
        
        # Apply distance transformation *after* setting negligible values to zero
        if self.metric_kind == "distance":        
            self.similarity.loc[:, "value"] = self.distance_transform(self.similarity.loc[:, "value"])

        if self.use_graph:
            logging.debug("Optimizing graph")
            # Pick the final pairs in a way that maximizes the edge weights
            self.match_pred = self.opt_graph(self.similarity)

            self.match_pred_simonly = self.calc_maxsim_match(self.similarity)
        else:
            # If you don't use the graph, then just take maximum similarity
            # Independent of all other observations
            logging.debug("\tCalc max sim")
            self.match_pred = self.match_pred_simonly = self.calc_maxsim_match(self.similarity)
        self.match_pred_simonly_unique = self.calc_maxsim_match_unique(self.similarity)

        assert self.match_pred.shape == self.match_pred_simonly.shape

    def build_graph(self, p, labellen=6):
        """
        Using match probabilities/similarity derived from the fit model,
        build a graph using those probabilities/similarity as edge weights.
        Args:
            p: A matrix containing pairwise match probabilities/similarities.
        """
        self.labellen = labellen

        def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
            return ''.join(random.choice(chars) for _ in range(size))

        S_edges, T_edges = zip(*p.index.values)
        S_edges = map(str, S_edges)
        T_edges = map(str, T_edges)

        p.index = pd.MultiIndex.from_tuples(zip(S_edges, T_edges))

        graph = nx.Graph()

        # Node names must be unique or networkx will complain
        s_edges = ["Source%s%s" % (id_generator(size=self.labellen), xx) for xx in np.unique(S_edges)]
        t_edges = ["Target%s%s" % (id_generator(size=self.labellen), xx) for xx in np.unique(T_edges)]

        # A bit of paranoid computing: the graph algorithm may use the 
        # input node/edge order as a default. 
        # This could artificially inflate accuracy, 
        # so make sure s and t are in (different) random orders
        np.random.shuffle(s_edges)
        np.random.shuffle(t_edges)

        # add nodes
        graph.add_nodes_from(s_edges, color="red",
                             bipartite=0)
        graph.add_nodes_from(t_edges, color="green",
                             bipartite=1)
        # add edges
        for e1 in s_edges:
            for e2 in t_edges:
                s_index = e1[(self.labellen + 6):]
                t_index = e2[(self.labellen + 6):]
                if (s_index, t_index) in p.index:
                    weight = p.loc[(s_index, t_index)].iloc[0]
                    graph.add_edge(e1, e2, weight=weight)
                else:
                    print (s_index, t_index), "Not in probabilities!"
        return graph

    def score(self, scoretype="match"):
        """
        Return percent of correct predictions
        """
        if scoretype == "match":
            return (self.match_pred["Source_num"] == self.match_pred["Target_num"]).mean()
        elif scoretype == "select":
            return (self.match_pred_simonly["Source_num"] == self.match_pred_simonly["Target_num"]).mean()
        elif scoretype == "select_unique":
            return (self.match_pred_simonly_unique["Source_num"] == self.match_pred_simonly_unique["Target_num"]).mean()
    
    def draw_graph(self, kwargs={"with_labels": True}):
        """
        A helper function for plotting graphs
        """
        nodes = self.graph.nodes(data=True)
        edges = self.graph.edges(data=True)
        nx.draw(self.graph, node_color=[nn[1]['color'] for nn in nodes],
                width=[2 * ee[2]["weight"] ** 2 for ee in edges],
                **kwargs)

    def graph_preds(self):
        sys.exit("Finish implementing")

        def myscatter(x, y, **kwargs):
            if len(x) > 10:
                alpha = .4
            else:
                alpha = .9
            plt.scatter(x, y, alpha=alpha, **kwargs)

        sim = self.similarity.reset_index()
        pred = self.match_pred.set_index("Source_num")
        pred_simonly = self.match_pred_simonly.set_index("Source_num")
        sim["true match"] = (sim["Source_num"] == sim["Target_num"])
        sim["graph_predicted"] = (pred.loc[sim["Source_num"]] == sim["Target_num"]).values
        sim["select_predicted"] = (pred_simonly.loc[sim["Source_num"]] == sim["Target_num"]).values
    
    def get_XY(self, X_from, X_to, col_groups=None, max_pairs=None):
        """
        X_from: Dataframe of features
        X_to: Related dataframe of matched features (same order)
        col_groups: Group key for columns
        max_pairs: Not used. Included to be consistent with WeightedNearestNeighbor interface.
        """
        # convert distance matrics from long form to short form
        X_from.index.name = "ind_from"
        X_to.index.name = "ind_to"
        
        idx1 = X_from.index.name
        idx2 = X_to.index.name
        
        if col_groups is not None:
            tmpdf = pd.DataFrame({"Col_index": range(X_to.shape[1]), "To": X_to.columns, "From": X_from.columns, "Groups": col_groups})
            count = 0
            for kk, dd in tmpdf.groupby("Groups"):
                val = pd.DataFrame(cdist(X_from.iloc[:, list(dd["Col_index"])], X_to.iloc[:, list(dd["Col_index"])], metric=self.metric), index=X_from.index, columns=X_to.index)
                val /= (1. * (pd.Series(col_groups) == kk).sum())
                if count == 0:
                    pairwise_dists = val
                else:
                    pairwise_dists += val
                count += 1
            pairwise_dists = pairwise_dists.reset_index()
        else:
            pairwise_dists = pd.DataFrame(cdist(X_from, X_to, metric=self.metric), index=X_from.index, columns=X_to.index).reset_index()
        pairwise_dists = pd.melt(pairwise_dists, id_vars=idx1).set_index([idx1, idx2])
        pairwise_dists.index.names = ["Source_num", "Target_num"]
        discordant = pd.Series([(xx[0] != xx[1]) + 0 for xx in pairwise_dists.index.values], index=pairwise_dists.index)
        return pairwise_dists, discordant

class DistSelect(_GraphMatcher):
    """
    Class for Select@N using a wide variety of distance metrics.
    """
    def __init__(self, metric="euclidean", distance_transform=lambda x: x.max() - x,
                 graph_opt_kwargs={}, use_graph=True):
        """
        metric: can be a function or any string recognized by scipy.spatial.distance.cdist
            http://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.spatial.distance.cdist.html
        metric_kind: distance, probability
        distance_transform: function for converting distance to similarity
        graph_opt_kwargs: dict. arguments to networkx.max_weight_matching
        use_graph: T/F. Whether to perform graph optimization
        """
        self.use_graph = use_graph
        self.metric = metric
        self.metric_kind = "distance"
        self.distance_transform = distance_transform
        self.opt_kwargs = graph_opt_kwargs

    def fit(self, X, y):
        pass


class WeightedNearestNeighbor(_GraphMatcher):
    """
    Learn a weighted distance metric to pick out matching values
    """
    def __init__(self, p=2,
                 model=LogisticRegressionCV, model_args=[], model_kwargs={'class_weight': "balanced"},
                 opt_kwargs={}, debug=False, labellen=5, metric_kind="prob", use_graph=True, max_pairs=1e5):
        """
        By default, we fit logistic regression, re-weighting the matches/non-matches
        so they contribute equally (`class_weight="balanced"`)
        p: the power to which to raise the abs(observed - predicted) residual
        model: sklearn model to fit
        model_args: args to model
        model_kwargs: kwargs to model
        opt_kwargs: kwargs to nx.max_weight_matching
        labellen: node labels are obfuscated to prevent sort order effects in matching algorithm
        max_pairs: The maximum number of edges to consider for training. Subsamples mismatches.
        """
        self.max_pairs = int(max_pairs)
        self.use_graph = use_graph
        self.metric = "cityblock"
        self.metric_kind = metric_kind
        self.model = model(*model_args, **model_kwargs)
        self.p = p
        self.opt_kwargs = opt_kwargs
        self.debug = debug
        self.labellen = labellen
        self.all_similarities = None

    def fit(self, X, y):
        self.model.fit(X, y)

    def get_XY(self, X_from, X_to, col_groups=None, max_pairs=np.inf):
        distances = get_residual(X_from, X_to, max_pairs=max_pairs, col_groups=col_groups)
        distances.dropna(inplace=True)

        discordant = pd.Series([(xx[0] != xx[1]) + 0 for xx in distances.index.values], index=distances.index)
        return distances, discordant


def get_residual(X_from, X_to, p=1, max_pairs=int(1e4), labels=["Target_num", "Source_num"], col_groups=None):
    """
    Given matrices for observed and inferred values (NxM), calculate the residuals for all observed, inferred pairs (rows)
    for each attribute (columns).
    """
    def get_resid(X_from, X_to, p0, p1, p=1):
        resid = np.abs(
                        X_from[p0] -
                        X_to[p1]
                    ) ** p

        return resid
    # TODO make probability of pair omission equal across the source nodes.

    # Because pandas dataframes are stored by column,
    # for speed we want our operations to work on columns instead of rows.
    X_to.columns = X_from.columns
    X_to = X_to.T.copy()
    X_from = X_from.T.copy()

    max_disc_pairs = int(min(max_pairs - X_to.shape[1], X_to.shape[1] * (X_to.shape[1] - 1))
                         )

    inds_obs = X_to.columns
    inds_inf = X_from.columns

    disc_count = 1
    overall_count = 0

    # Loop through in random order
    # This is desirable because we omit some pairs to save time.
    pairs = [xx for xx in itertools.product(inds_inf, inds_obs)]
    
    output_index = list(X_to.index)
    
    output = pd.DataFrame(np.nan,
                          index=output_index,
                          columns=pd.MultiIndex.from_tuples(pairs)
                          )
    output.columns.names = ["Source_num", "Target_num"]
    output.index.name = None

    np.random.shuffle(pairs)
    for p0, p1 in pairs:
        # If a pair is discordant, check whether we have enough discordant pairs
        if p0 != p1:
            if disc_count <= max_disc_pairs:
                
                output[(p0, p1)] = get_resid(X_from, X_to, p0, p1, p=p)
                disc_count += 1
                overall_count += 1
        else:
            # Get distance between pair
            output[(p0, p1)] = get_resid(X_from, X_to, p0, p1, p=p)
    if col_groups is not None:
        return output.groupby(col_groups).mean().T.dropna(how="all")
    return output.T.dropna(how="all")


def run_v1_v2(model_names, model_kwargs, maxElements, use_local=False, async=True, short_run=False,
                test_col="eyecolor.v1", namespace=ml_vars.NAMESPACE):
    """
    Run pipeline for matching of visit1 to visit2 values. 

    Args:
        model_names: A dictionary of Select@N model_names
        model_kwargs: A dictionary of kwarg dictionaries for these model_names
        maxElements: A dictionary mapping from column substring to the number of leftmost elements to 
                     keep in the unpacked form of that column. Useful for PCs.
    """
    # Infer the columns that can be used for v1/v2 matching   
    best_matches = mu.get_v1_v2_cols()
    print best_matches["Group"].value_counts()
    
    # Read data from Rosetta if you're not using a local version
    if not os.path.exists("data_local/df_v1_v2.txt") or not os.path.exists("data_local/best_match_v1_v2.txt") or not use_local:
        
        # The first checkpoint is the raw dataframe (has packed columns)
        print "Querying Rosetta"
        if not os.path.exists("data_local/df_v1_v2_raw.txt"):
            db = RosettaDBMongo(host="rosetta.hli.io")
            db.initialize(namespace=namespace)
            df = db.query(list(best_matches.index) + 
                          list(best_matches["Best Match"]) + 
                          [ml_vars.PROJECT_KEY, ml_vars.SAMPLE_NAME],
                          filters={ml_vars.PROJECT_KEY: "FACE"}
                          )
            df.to_csv("data_local/df_v1_v2_raw.txt", sep="\t", encoding="utf-8")
        else:
            df = pd.read_table("data_local/df_v1_v2_raw.txt", sep="\t", index_col=0, encoding="utf-8")
        
        if short_run:
            df = df[[xx for xx in df.columns if test_col in xx or "qc." in xx]]

        # Now that you have the data as it is in Rosetta, unpack columns
        packed_df = mod.PackedFrame(df, maxElements=maxElements)
        
        print "Unpacking columns (size_init=%s)" % str(df.shape)
        df_v1_v2 = packed_df.unpack()
        print "Size (unpacked) = %s" % str(df_v1_v2.shape)
        best_matches = packed_df.update_colmaps(best_matches, from_str="visit1", to_str="visit2")
        
        # Write data to disk so you can start from this point next time if you want
        df_v1_v2.to_csv("data_local/df_v1_v2.txt", sep="\t", encoding="utf-8")
        best_matches.to_csv("data_local/best_match_v1_v2.txt", sep="\t")
    
    # Sometimes preferable to just read from a locally cached version
    else:
        print "Reading data"
        df_v1_v2 = pd.read_table("data_local/df_v1_v2.txt", sep="\t", index_col=0, encoding="utf-8")
        best_matches = pd.read_table("data_local/best_match_v1_v2.txt", sep="\t", index_col=0)
    
    if short_run:
        model_names = {kk: vv for kk, vv in model_names.iteritems() if "CosDist" in kk or "ManDist" in kk}
        df_v1_v2 = df_v1_v2[[xx for xx in df_v1_v2.columns if test_col in xx or "qc." in xx]]

    # Now pull out the column sets that you want to run over
    cols_obs = {}
    cols_pred = {}
    for gg, dd in best_matches.groupby("Group"):
        if short_run and gg != test_col:
            # Skip all sets but one if you're doing a short run
            continue
        pred_cols = [xx for xx in df_v1_v2.columns if (xx in dd.index)]
        obs_cols = list(dd.ix[pred_cols, "Best Match"].values)
        if len(obs_cols) > 0 and len(pred_cols) > 0:
            print gg, len(pred_cols), len(obs_cols)
            cols_obs[gg] = obs_cols
            cols_pred[gg] = pred_cols
    
    # Run the panel!
    eval_kwargs = {"normalize": True, "impute": False}
    eval_panel = EvalSelectPanel(df_v1_v2, model_names=model_names, model_kwargs=model_kwargs,
                                    from_col_dict=cols_pred,
                                    to_col_dict=cols_obs,
                                    eval_kwargs=eval_kwargs, async=async)
    res = eval_panel.run()

    suffix = "_shortrun" if short_run else ""
    with open("data_local/match10_v1_v2%s.pickle" % suffix, "w") as fout:
        cPickle.dump(res, fout)


def run_obs_pred(model_names, model_kwargs, maxElements, use_local=False, async=True, short_run=False):
    """
    Run the pipeline for matching individuals based on observed/predicted vector pairs.

    Args:
    model_names: A dictionary of Select@N model_names
    model_kwargs: A dictionary of kwarg dictionaries for these model_names
    maxElements: A dictionary mapping from column substring to the number of leftmost elements to 
                 keep in the unpacked form of that column. Useful for PCs.
    """
    from sklearn.cross_validation import KFold

    # Read in data from local cache, or from rosetta
    if use_local:
        df = pd.read_table("data_local/df_pred_obs.txt", sep="\t", index_col="index")
        best_matches = pd.read_table("data_local/pred_to_obs_colnames.txt", sep="\t", index_col=0)
    else:
        # This makes sense to hard-code. The input sample key names are hg19 versions
        df, best_matches = mu.get_select_data(namespace="hg19")
    # Use only one column set for a short run
    if short_run:
        df = df[[xx for xx in df.columns if "skin" in xx or "qc." in xx]]

    # Unpack columns if applicable
    packed_df = mod.PackedFrame(df, maxElements=maxElements)
    df_obs_pred = packed_df.unpack()
    
    best_matches = packed_df.update_colmaps(best_matches, to_str=".FACE.", from_str=".FACE_P.")

    # end

    # Map set names for sets of columns
    if short_run:
        sets = {"skin": best_matches[best_matches["Group"].str.contains("skin.v1")]}
        model_names = {kk: vv for kk, vv in model_names.iteritems() if "CosDist" in kk or "ManDist" in kk}
    else:
        skin_regex = re.compile("dynamic.FACE.skin.v3.[ablRGB]")
        bald_regex = re.compile("dynamic.FACE.MalePatternBaldness.v2.value")
        body_regex = re.compile("(dynamic.FACE.age.v1.value)|(dynamic.FACE.bmi.v1.value)|(pheno.height)|\
(dynamic.FACE.weight.v1.value)|(dynamic.FACE.gender.v1.value)")
        eyes_regex = re.compile("dynamic.FACE.eyecolor.v1_visit1.[abl]")
        lmdist_regex = re.compile("dynamic.FACE.face_lmdist.v3_visit1.*")
        facepc_regex = re.compile('dynamic.FACE.face.v1_visit1.DepthPC')  # Beware upper case vs. lower case
        voice_regex = re.compile("dynamic.FACE.voice.v1_visit1.complete.coeff[0-9]?[0-9]")
        
        def matchfunc(regex):
            return [regex.search(xx) is not None for xx in best_matches["Best Match"]]
        
        sets = {
            "skin": best_matches[matchfunc(skin_regex)],
            "bald": best_matches[matchfunc(bald_regex)],
            "body": best_matches[matchfunc(body_regex)],
            "eye": best_matches[matchfunc(eyes_regex)],
            "lmdist": best_matches[matchfunc(lmdist_regex)],
            "facePC": best_matches[matchfunc(facepc_regex)],
            "voice": best_matches[matchfunc(voice_regex)],
        }
        
        notdemo = pd.concat([sets[ss] for ss in sets if ss in 
                                        ["lmdist", "eye", "bald", "facePC", "skin"]
                                    ], axis=0).drop_duplicates()
        notvoice = pd.concat([sets[ss] for ss in sets if ss not in ["voice"]], axis=0).drop_duplicates()
        all_vars = pd.concat([sets[ss] for ss in sets], axis=0).drop_duplicates()
        sets["notdemo"] = notdemo
        sets["notvoice"] = notvoice
        sets["all"] = all_vars

        for ss in sets:
            print "%s uses %s columns" % (ss, sets[ss].shape[0])
    # end
    
    # Package column names into observed and predicted dictionaries
    cols_obs = {}
    cols_pred = {}
    for ss in sets:
        pred_cols = [xx for xx in df.columns if (xx in sets[ss].index)]
        obs_cols = list(sets[ss].ix[pred_cols, "Best Match"].values)
        print ss, len(pred_cols), len(obs_cols)
        cols_obs[ss] = obs_cols
        cols_pred[ss] = pred_cols
    # end

    subjects = pd.read_table("data_local/pred_obs_sampleset.txt", sep="\t", header=None)[0]

    df_obs_pred = df_obs_pred[df_obs_pred[ml_vars.SAMPLE_NAME].isin(subjects)]
    assert df_obs_pred.shape[0] == len(subjects), "Did not find all specified subjects"
    n_folds = int(np.floor(df_obs_pred.shape[0] / 100.))
    folds = KFold(df_obs_pred.shape[0], n_folds=n_folds, random_state=47)
    fold = 1
    for train_index, test_index in folds:
        # Run panel!
        eval_kwargs = {"normalize": True, "impute": True, 
                       "train_inds": list(df_obs_pred.index[train_index]),
                       "test_inds": list(df_obs_pred.index[test_index])
                        }
        eval_panel = EvalSelectPanel(df_obs_pred, model_names=model_names, model_kwargs=model_kwargs,
                                        from_col_dict=cols_pred,
                                        to_col_dict=cols_obs,
                                        eval_kwargs=eval_kwargs, async=async)
        res = eval_panel.run()

        suffix = "_shortrun" if short_run else ""
        with open("data_local/match10_pred_obs%s_fold%s.pickle" % (suffix, fold), "w") as fout:
            cPickle.dump(res, fout)

        fold += 1


if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegressionCV
    import cPickle
    
    use_local = False
    async = True
    fit_obs_pred = True
    fit_v1_v2 = False
    short_run = False

    N_pcs = {"DepthPC": 20, "ColorPC": 20}
    
    # Model args
    model_names = {
        "LogisticReg": WeightedNearestNeighbor,
        "MinEucDist": DistSelect,
        "MinCosDist": DistSelect,
        "MinManDist": DistSelect,
    }

    LR_args = {"model": LogisticRegressionCV, "model_kwargs": {"class_weight": "balanced"}}

    model_kwargs = {
        "LogisticReg": LR_args,
        "MinEucDist": {},
        "MinCosDist": {"metric": "cosine"},
        "MinManDist": {"metric": "cityblock"},
    }
    # end model args

    if fit_obs_pred:
        run_obs_pred(model_names, model_kwargs, N_pcs, use_local=use_local, async=async, short_run=short_run)

    if fit_v1_v2:
        run_v1_v2(model_names, model_kwargs, N_pcs, use_local=use_local, async=async, short_run=short_run)
