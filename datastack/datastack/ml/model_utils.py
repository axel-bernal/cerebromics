import sys
from datastack.cerebro.cross_validation import HashedKfolds
sys.path.append("../")
sys.path.append("../../../")
import datastack.common.settings as settings
import preprocess as pp
import pandas as pd
import numpy as np
import cross_validation as cv
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
import re
from datastack.dbs.rdb import RosettaDBMongo
import rosetta_settings as ml_vars
import json

class PackedFrame:
    """
    Take in a dataframe and detect and unpack columns that contain vectors.

    Optionally, match "from" and "to" columns to each other after unpacking. 
    "From" columns may be, e.g. visit 1 or predicted values. 
    The corresponding "To" columns would be visit 2 and observed values, respectively.
    """
    def __init__(self, df, copy=True, maxElements={}, **kwargs):
        """
        Args: 
            df:          A dataframe
            copy:        T/F. Whether to copy the input dataframe
            maxElements: A dictionary with column names as keys. 
                         Specifies how many of the leftmost elements to take in input vectors.
                         This is intended for use with PC columns.
        """
        if copy:
            self.df = df.copy()
        else: 
            self.df = df
        self.maxElements = maxElements
        self.col_maps = {}

    def unpack(self, inplace=False):
        """
        Infer which columns are packed and unpack them.
        """
        packed_cols = self.get_packed_cols()
        col_maps = {}
        dfs = []

        self.filter_cols = []

        # Unpack columns
        for cc in packed_cols:
            # Don't unpack missing values
            col_short = self.df[cc].dropna()
            toadd = pd.DataFrame(map(lambda x: eval(str(x)), col_short), index=col_short.index)
            toadd.columns = ['%s.%s' % (cc, xx + 1)
                               for xx in range(toadd.shape[1])
                               ]

            # Keep all column names so you can filter mates of dropped columns (e.g. v1 vs v2, pred vs obs)
            col_maps[cc] = toadd.columns

            # See if this is a column where you want the first N elements
            for ee in self.maxElements:
                if ee in cc:
                    # This is where you track what to filter later
                    self.filter_cols.extend(toadd.columns[self.maxElements[ee]:])

                    # But don't deal with unnecessary data
                    toadd = toadd.iloc[:, :self.maxElements[ee]]
                    break
            dfs.append(toadd)

        self.col_maps = col_maps
        
        orig_to_return = self.df[[cc for cc in self.df.columns if cc not in packed_cols]]

        full_df = pd.concat([orig_to_return] + dfs, axis=1)
        
        if inplace:
            self.df = full_df
        else:
            # TODO: Should we just constrain it so each col_map is a number of columns and not the columns themselves?
            return full_df

    def pack(self, packed_col, topack_cols=None, delete_unpacked=True, to_json=True, inplace=False):
        """
        Pack the unpacked packed_col as json string.  This uses the col_map
        to find the exlpoded column names.
        Else if topack_cols set, then this is entirely manual.

        :param topack_cols: explicit list of column names to pack
        :param packed_col: name of new packed column
        :param delete_unpacked: default True.  Delete the topack_col columns.
        :param to_json: default True.  Create JSON packing.
        :paeram inplace: default False.
        :return: dataframe with modifications.
        """
        if topack_cols is None:
            if packed_col in self.col_maps:
                topack_cols = self.col_maps[packed_col]
            else:
                print "model_utils.PackedFrame.pack error: {} does not have a column map entry".format(packed_col)
                return None
        else: # make sure if in manual mode we don't mess up auto-packed columns
            if packed_col in self.col_maps:
                print "model_utils.PackedFrame.pack error: pre-existing column map entry {}".format(packed_col)
                return None

        # Create a new column with a list of all the numbers
        def make_packed_column(row):
            out = [row[i] for i in topack_cols]
            if to_json:
                out = json.dumps(out)
            return out

        # Pandas bug.  If df column count == topack_col count, then it won't return a series
        # but instead return a dataframe.
        if len(self.df.columns) == len(topack_cols):
            self.df['_FAKE'] = 0xdeadbeef

        if not inplace:
            copy_df = self.df.copy()
        else:
            copy_df = self.df

        copy_df[packed_col] = self.df.apply(make_packed_column, axis=1, reduce=True)

        if '_FAKE' in self.df.columns:
            self.df = self.df.drop('_FAKE', axis=1)

        if delete_unpacked:
            copy_df = copy_df.drop(topack_cols, axis=1)
            if packed_col in self.col_maps:
                del(self.col_maps[packed_col])

        if not inplace:
            return copy_df

    def get_packed_cols(self):
        """
        Infer which columns are packed by examining the first non-null value
        """
        possible_unpack = self.df.columns[self.df.dtypes == 'object']
        packed_cols = []
        for cc in possible_unpack:
            if (self.df[cc].notnull().sum() > 0) and re.match("\[.*\]$", str(self.df[cc].dropna().iloc[0])):
                packed_cols.append(cc)
        print "\n".join(packed_cols)
        print "Found %s packed columns" % len(packed_cols)
        return packed_cols

    def find_bestmatch(self, ii, row, all_cols, to_str, from_str):
        """
        Given a "from" column, find its corresponding "to" column
        Args:
            ii:       The name of the 'from' column name
            row:      The row of this 'from' columns, containing its current best match, the score of that match, etc.
            all_cols: All relevant column names, a fallback in case a match is not found right away
            to_str:   The substring which uniquely identifies "to" columns
            from_str: The substring which uniquely identifies "from" columns
        """
        from fuzzywuzzy import process
        mate = row["Best Match"]
        query = ii.replace(from_str, to_str)
        incl_str = to_str

        if mate in self.col_maps:
                # Search only in the expansion of this column
                if query.lower() in [xx.lower() for xx in self.col_maps[mate]]:
                    for xx in self.col_maps[mate]:
                        if xx.lower() == query.lower():
                            print "\tExact match for %s!" % query
                            return xx, 100
                else:
                    print "\tNo exact match for %s" % query
                    res = process.extract(query, self.col_maps[mate], limit=1)
                    if len(res) == 0:
                        return np.nan, 0
                    return res[0]
        else:
            # Otherwise re-do the search
            print "\tSearching in all cols"
            group = row["Group"]
            searchset = [xx for xx in all_cols if (group in xx) and (incl_str in xx)] 
            if query in searchset:
                print "\tIn all_cols, found exact match for %s!" % query
                return query, 100
            else:
                print "\t\tUsing fuzzy matching for %s" % query
                res = process.extract(query, searchset, limit=1)
                if len(res) == 0:
                    return np.nan, 0
                return res[0]
    
    def expand_match_fromcols(self, mapdf, all_cols, to_str, from_str, 
                              check_exceptions_bestmatch=["pheno.height"]):
            """
            For "from" columns, expand the matching dictionary based on the new columns produced by unpacking
            """
            mapdf["From_orig"] = mapdf.index
            assert (pd.Series(mapdf.index).str.contains(from_str)).all(), "From column didn't contain 'from_str'"

            bestmatch_check = mapdf["Best Match"][~pd.Series(mapdf["Best Match"]).str.contains(to_str)]

            assert (bestmatch_check.isin(check_exceptions_bestmatch)).all(), "'Best Match' column didn't contain 'to_str'"
            
            newrows = []
            tokeep = pd.Series(True, index=mapdf.index)
            mapped_inds = []
            for ii in mapdf.index:
                if ii in self.col_maps:
                    print ii
                    row = mapdf.loc[ii]
                    tokeep[ii] = False
                    for new_col in self.col_maps[ii]:
                        print "\t", "mapping", new_col
                        # res = self.find_bestmatch(new_col, row, all_cols, to_str, from_str, kind="from")
                        newrows.append(pd.DataFrame({"Best Match": [row["Best Match"]], 
                                                     "Group": [row["Group"]],
                                                     "Score": [np.nan],
                                                     "From_orig": ii}, index=[new_col]))
                        mapped_inds.append(new_col)
            if newrows:
                if tokeep.sum() > 0:
                    newrows.append(mapdf[tokeep])
                mapdf = pd.concat(newrows, axis=0)
                
            else:
                mapdf = mapdf[tokeep]
            return mapdf, mapped_inds

    def update_colmaps(self, mapdf, colset="dynamic.FACE", to_str=".FACE.", from_str=".FACE_P.", 
                        namespace=ml_vars.NAMESPACE, **kwargs):
        """
        Update the "from" column to "to" column matching after unpacking all columns
        """
        db = RosettaDBMongo(host="rosetta.hli.io")
        db.initialize(namespace=namespace)
        all_cols = db.find_keys(colset, regex=True)
        print "Expanding from columns"
        mapdf, mapped_inds = self.expand_match_fromcols(mapdf, all_cols, to_str, from_str)

        print "Expanding to columns"
        for ii in mapdf.index:  # .difference(mapped_inds):
            jj = mapdf.loc[ii, "Best Match"]
            row = mapdf.loc[ii]
            if (jj in self.col_maps) or (ii in mapped_inds):
                res = self.find_bestmatch(ii, row, all_cols, to_str, from_str)
                mapdf.loc[ii, ["Best Match", "Group", "Score"]] = [res[0], row["Group"], res[1]]

        # Remove values where the query or its mate were filtered during unpacking
        tokeep = pd.Series(True, index=mapdf.index)
        for ii in mapdf.index:
            if (ii in self.filter_cols) or (mapdf.loc[ii, 'Best Match'] in self.filter_cols):
                tokeep.loc[ii] = False
        
        mapdf = mapdf[tokeep]
        for ii in mapdf.index:
            if ii.replace(from_str, to_str).lower() == mapdf.loc[ii, "Best Match"].lower():
                mapdf.loc[ii, "Score"] = 100

        return mapdf

        # Find out if there are columns we need to retrieve and merge


class _RegressionPipeline:
    def __init__(self, covariates, outcome, fit_order=None,
                 xygen_kwargs={'annot_expr': '(%s)|(%s)' % (settings.ROSETTA_STUDY_KEY, settings.ROSETTA_INDEX_KEY)},
                 model_names=None, model_params=None, kfold_kwargs={}, use_named_covs=False):
        self.use_named_covs = use_named_covs
        self.xygen = pp.XYGenerator(pp.get_regex(list(covariates)), outcome, **xygen_kwargs)
        self.kfold_kwargs = kfold_kwargs
        self.X_and_y = self.xygen()
        self.outcome = self.xygen.y_cols[0]
        self.covariates = pd.Series(covariates)
        self.results = None
        self.fitorder_specified = fit_order

        if self.fitorder_specified is None:
            self.fit_order = self.covariates.index
        else:
            self.fit_order = fit_order

        if model_names is None:
            model_names = {
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge()
            }

        self.model_names = model_names
        if model_params is None:
            model_params = {
                "Linear Regression": {},
                "Ridge": {'alpha': [.0001, .001, .01, .1, 1, 10]}
            }
        self.model_params = model_params

    @property
    def X_cols(self):
        return self.xygen.X_cols

    @property
    def y_cols(self):
        return self.xygen.y_cols

    @property
    def annot_cols(self):
        return self.xygen.annot_cols

    @property
    def report_cols(self):
        return self.xygen.report_cols

    @property
    def data(self):
        return self.xygen.data

    @property
    def X(self):
        return self.xygen.X

    @property
    def y(self):
        return self.xygen.y

    def add_covariate(self, key, new_val):
        if key in self.covariates:
            # Delete all variables
            old_val = self.covariates[key]
            old_cols = [xx for xx in self.data.columns if re.search(old_val, xx)]

            # Delete records for the old value
            for oo in old_cols:
                self.xygen.delete_records(oo)

            # Update to new value
            self.covariates[key] = new_val
        else:
            self.covariates = self.covariates.append(pd.Series(new_val, index=[key]))

        self.fit_order = self.covariates.index

        # Add columns
        cols = [xx for xx in self.data.columns if re.search(new_val, xx)]
        self.add_columns(cols)

    def add_columns(self, cols):
        # Add records
        for cc in cols:
            self.xygen.add_col(cc)

        # Clean
        if self.xygen.clean:
            self.xygen.clean_data(cols=cols)

        # Regenerate X and y
        self.X_and_y = self.xygen()

    def filterxy(self, X_and_y, regex):
        """
        Purpose: Subset X columns
        Args:
            X_and_y: A tuple of X and y
            regex: A regular expression to filter columns of X
        """
        cols = [xx for xx in self.X_cols if re.search(regex, xx)]
        return (X_and_y[0][cols], X_and_y[1])

    def fit_and_score(self, regex, model, params={}):
        """
        Purpose: Fit model and retrieve score
        Args:
            name: Name of the covariate group
            regex: Regular expression to retrieve covariates
            kind: ['indiv', 'plusbase', 'overbase', 'cumm']

        """

        res = cv.EvaluationRegression(self.filterxy(self.X_and_y, regex),
                                      self.kfold,
                                      estimator=model,
                                      params=params)
        scores = res.metrics_df.iloc[0]
        return pd.Series({"R2": scores["0:R2"], "MAE": scores["0:MAE"], "RegObj": res})

    @property
    def filter_info(self):
        cols = []
        for cc in set(self.X_cols + self.y_cols + self.annot_cols + self.report_cols):
            if (self.xygen.filter_info[cc] != self.xygen.filter_null_col).any():
                cols.append(cc)
        outlier_col = "Outliers (trim)" if self.xygen.trim_outliers else "Outliers (warn)"
        if len(cols) == 0:
            print "No observations filtered..."
        return self.xygen.filter_info.ix[[outlier_col, "Missing Values", "User Filters", "Dropped", "Drop Reason"], cols].T

    def view_results(self):
        from IPython.html import widgets

        fixed = widgets.interaction.fixed

        def print_combo(result_panel=None, regressors_all=None, compare=None, metric=None, regressors=None):
            pd.set_option("precision", 3)
            pd.set_option('display.max_columns', 500)

            if compare == "All model_names":
                if regressors == "All":
                    print "Please specify a model, or a regressor set across which to compare."
                    return None
                col = "%s_%s" % (metric, regressors_dict[regressors])
                print result_panel.ix[:, :, col]

            else:
                regrW.value = "All"
                if regressors == "All":
                    cols = ["%s_%s" % (metric, regressors_dict[yy]) for yy in regressors_all if yy != "All"]
                    print result_panel.ix[compare, :, cols]

        regressors_dict = {
            "Individual": "indiv",
            "Plus base": "plusbase",
            "Over base": "overbase",
            "Cumulative": "cumm"
            }

        regressors_all = ["Individual", "Plus base", "Over base", "Cumulative", "All"]

        metrics = ['R2', 'MAE']
        try:
            compareW = widgets.Dropdown(options=["All model_names"] + list(self.results.axes[0]))
        except RuntimeError as e1:
            print e1
            print "Failed to create widget. Are you running from an ipython notebook?"
            return None
        metricW = widgets.Dropdown(options=metrics)
        regrW = widgets.Dropdown(options=regressors_all)
        i = widgets.interactive(
            print_combo, result_panel=fixed(self.results),
            regressors_all=fixed(regressors_all),
            regressors_dict=fixed(regressors_dict),
            compare=compareW, metric=metricW,
            regressors=regrW
        )

        return i

    def plot_outliers(self, panel, xcol, ycol):
        panel.ax_joint.scatter(self.data.ix[self.xygen.outliers.flagged[ycol], xcol],
                               self.data.ix[self.xygen.outliers.flagged[ycol], ycol],
                               facecolors='none', edgecolors='indianred',
                               s=60, alpha=.8, linewidth=2)

    def plot(self, colname, plotfunc=sns.jointplot, **kwargs):
        if plotfunc == sns.jointplot and "kind" not in kwargs:
            kwargs["kind"] = "reg"
        panel = plotfunc(self.outcome, colname, self.data, **kwargs)

        if plotfunc == sns.jointplot:
            if len(self.xygen.outliers.flagged[colname]) > 0:
                self.plot_outliers(panel, self.outcome, colname)

    def __getitem__(self, key):
        return self.xygen.data[key]

    def __setitem__(self, key, value):
        self.xygen.set_feature(key, value)

    def prep_run(self):
        self.X_and_y = self.xygen()
        self.kfold = HashedKfolds(df=self.xygen.data.loc[self.X_and_y[0].index], **self.kfold_kwargs)

    def plot_fit(self, model, kind, covariate, **jointplot_kwargs):
        """
        For a given fit, plot observed versus predicted y values
        """
        if self.results is None:
            print "No fit to report"
        else:
            reg_obj = self.results.ix[model, covariate, "Fit_%s" % kind]
            df = pd.DataFrame({self.outcome: reg_obj.y.iloc[:, 0], self.outcome + " pred": reg_obj.y_pred})
            panel = sns.jointplot(self.outcome, self.outcome + " pred", df, kind="reg", **jointplot_kwargs)

            # This needs to be moved to Cumulative Regression
            reg_count = pd.Series(self.results.axes[1] == covariate).idxmax()
            var = self.covariates.index[reg_count]
            regex = self.get_regex(var, kind, reg_count)
            outlier_dict = {k: v for k, v in self.xygen.outliers.flagged.iteritems() if re.search(regex, k)}
            n_vars = len(outlier_dict)
            scale = max(1., n_vars / 2.)
            alpha = 1. / scale

            for ycol in outlier_dict:
                panel.ax_joint.scatter(df.ix[self.xygen.outliers.flagged[ycol], self.outcome],
                                       df.ix[self.xygen.outliers.flagged[ycol], self.outcome + " pred"],
                                       facecolors='none', edgecolors='indianred',
                                       s=60, alpha=alpha, linewidth=2)


class CumulativeRegression(_RegressionPipeline):
    def get_cumm_regex(self, count):
        """
        Purpose: Get regex for cumulative model
        Args:
            fit_order: a list of strings, corresponding to the order of covariates.
        """
        all_vars = []
        for ll in self.fit_order[:(count + 1)]:
            all_vars.append(self.covariates[ll])
        expr_c = pp.get_regex(all_vars)
        return expr_c

    def get_regex(self, var, kind, count):
        """
        Retrieve different regular expressions depending on the fit mode.
        Args:
            var: Fit variable
            kind: "indiv": Individual fit, "plusbase": variable + baseline,
                    "cumm": Fit with this variable and all previous variables
            count: Index in covariate series. This is necessary for retrieving all previous
                    variables
        """
        if kind == "indiv":
            regex = self.covariates[var]
        elif kind == "plusbase":
            regex = pp.get_regex(self.covariates[[var, self.fit_order[0]]])
        elif kind == "cumm":
            regex = self.get_cumm_regex(count)
        return regex

    def record_score(self, var, model, model_params, output_table, kind, count=0):
        """
        Record the scores of a given fit
        """
        regex = self.get_regex(var, kind, count)
        if not [xx for xx in self.X_cols if re.search(regex, xx)]:
            return False

        res = self.fit_and_score(regex, model, params=model_params)

        output_table.ix[var,
                        ["%s_%s" % (xx, kind)
                            for xx in ["R2", "MAE", "Fit"]]
                        ] = res.loc[["R2", "MAE", "RegObj"]].values
        return True

    def summarize_model(self, model, model_params):
        """
        Given a model and its parameters, run and save summary measures
        """
        output_table = pd.DataFrame(
            np.nan,
            index=self.fit_order,
            columns=[
                "R2_indiv", "R2_plusbase", "R2_overbase", "R2_cumm",
                "MAE_indiv", "MAE_plusbase", "MAE_cumm",
                "Fit_indiv", "Fit_plusbase", "Fit_cumm"
            ]
        )

        for count, ff in enumerate(self.fit_order):
            present = self.record_score(ff, model, model_params, output_table, "indiv")
            if not present:
                continue

            if count > 0:
                self.record_score(ff, model, model_params, output_table, "plusbase")
                self.record_score(ff, model, model_params, output_table, "cumm", count=count)
            else:
                for xx in ["R2", "MAE", "Fit"]:
                    output_table.ix[ff, "%s_cumm" % xx] = output_table.ix[ff, "%s_indiv" % xx]
        output_table.ix[1:, "R2_overbase"] = output_table.ix[1:, "R2_plusbase"] - output_table.ix[0, "R2_indiv"]
        output_table.ix[1:, "MAE_overbase"] = output_table.ix[1:, "MAE_plusbase"] - output_table.ix[0, "MAE_indiv"]

        if not self.use_named_covs:
            output_table.index = self.covariates[self.fit_order]

        return output_table

    def run(self):
        """
        Run pipeline
        """
        self.prep_run()
        all_results = {}
        for mm in self.model_names:
            all_results[mm] = self.summarize_model(self.model_names[mm], self.model_params[mm])
        self.results = pd.Panel(all_results)
