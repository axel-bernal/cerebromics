"""
Copyright Human Longevity Inc. 2015. Authored by M. Cyrus Maher

Define classes to preprocess and serve data for analysis. Functions include:

- Retrieving data from database
- Fixing known data issues
- Flagging outliers based on robust stdevs
- Salvaging columns that are mostly numeric
- Coding categorical variables

There is also some functionality to do automatic correction of outliers.
TODO:
    - Support other categorical variable encodings besides "one hot" ("a", "b", "c", "d") --> (0 0 0, 0 0 1, 0 1 0, 1 0 0)
    - Pass in flag to trim outliers
    - Add clustering prior to outlier detection
    - Add PCA based outlier detection across sets of covariates?
"""

import sys
sys.path.append("../../")
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import re
from patsy import dmatrix, NAAction
from datastack.dbs.rdb import RosettaDBMongo
from datastack.ml import rosetta_settings as ros_vars
import logging
import outliers


def get_regex(expr_list):
    """
    Take a list of regexes and `or` them together
    """
    regex = ""
    for ee in expr_list:
        if regex:
            sep = "|"
        else:
            sep = ""
        regex += '%s(%s)' % (sep, ee)
    return regex


def replace_all(string, from_list, to_list):
    """
    For a given `string`, replace all patterns in `from_list` with the corresponding value in `to_list`
    """
    for ff, tt in zip(from_list, to_list):
        string = string.replace(ff, tt)
    return string


class _RegModel:
    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return r2_score(y, self.model.predict(X))


class FlexWindowOutlier(_RegModel):
    """
    This class mimics a sklearn regression class in order to facilitate outlier removal via
    optimization of bivariate out-of-sample prediction.
    """
    def __init__(self, top_quantile=1.0,
                 bottom_quantile=0.0, n_evals=10, min_unchanged_frac=.30, debug=False,
                 model=LinearRegression, model_kwargs={}):
        """
            Args:
                top_quantile: The maximum quantile to consider for trimming
                bottom_quantile: The minimum quantile to consider for trimming
                n_evals: The number of quantiles to search over for top and bottom threshold
                min_unchanged_frac: the minumum fraction of the data to leave unchanged
                debug: True/False
                model: An uninitialized sklearn-like model object
                model_kwargs: kwargs to pass to model.
        """

        self.top = top_quantile
        self.bottom = bottom_quantile
        self.min_unchanged = min_unchanged_frac
        self.n_evals = n_evals
        self.debug = debug
        self.model = model(**model_kwargs)
        assert self.top > self.bottom, \
            "Top quantile must be greater than bottom quantile"

    def get_gridsearch(self, n_evals=None):
        """
            Given a number of evals to do for both top and bottom of the window, produce a constrained grid search.
            self.min_unchanged controls how small the window can get.

        """
        if n_evals is None:
            n_evals = self.n_evals

        # Define ranges over which to search
        top_range = np.linspace(self.bottom + self.min_unchanged, self.top, self.n_evals)
        bottom_range = np.linspace(self.bottom, self.top - self.min_unchanged, self.n_evals)

        # Exclude combinations that leave too little data unchanged
        dict_list = []
        for tt in top_range:
            for ii in bottom_range[(tt - bottom_range) >= self.min_unchanged]:
                dict_list.append({'top_quantile': [tt], 'bottom_quantile': [ii]})
        return dict_list

    def fit(self, X, y):
        self.model.fit(X, self.trim(y))

    def score(self, X, y):
        corr, p = pearsonr(self.trim(y), self.model.predict(X))
        return corr * corr

    def get_params(self, **kwargs):
        return {'top_quantile': self.top, "bottom_quantile": self.bottom}

    def set_params(self, **params):
        if "top_quantile" in params:
            self.top = params["top_quantile"]
        if "bottom_quantile" in params:
            self.bottom = params["bottom_quantile"]
        return self

    def trim(self, y):
        """
        Perform outlier trimming
        """
        if self.debug:
            self.logger.debug("Top is %s, bottom is %s" % (self.top, self.bottom))
        y2 = y.copy()
        top_value = y2.quantile(self.top)
        bottom_value = y2.quantile(self.bottom)
        top = (y2 > top_value)
        bottom = (y2 < bottom_value)
        y2.loc[top] = top_value
        y2.loc[bottom] = bottom_value
        return y2


class XYGenerator:
    def __init__(self, X_expr, y_expr, X_exprs_inv=[], y_exprs_inv=[], annot_expr=None, annot_expr_inv=[], data=None, 
                 db_version=None, db_namespace='hg38_noEBV', loglevel="INFO", log_to="stream", interactions=[], 
                 data_filters={}, rosetta_filters={"ds.index.BAM": (None, "!=")},
                 clean=True, trim_outliers=False, outlier_stdevs=6):
        """
        Purpose: Pull data out of the database (optional), polish out imperfections (optional),
            Args:
                X_expr: A regular expression for retrieving X variables
                y_expr: A regular expression for retrieving y variables
                X_exprs_inv: A list of regular expressions for excluding X variables
                y_exprs_inv: A list of regular expressions for excluding y variables
                annot_expr: A regular expression for retrieving annotation variables
                annot_expr_inv: A list of regular expressions for excluding annotation variables
                db_version: version of Rosetta to use (default: None. Results in using latest version)
                data: A pandas dataframe (optional), otherwise query rosetta
                loglevel: ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]. See: https://docs.python.org/2/library/logging.html
                log_to: ["stream"]
                interactions: List of interaction terms to include in the linear model. Expected form is ["Var1:Var2", "VarX:VarY"]
                filters: Filters for query to rosetta
        }
        """
        self.data_filters = data_filters
        self.trim_outliers = trim_outliers
        self.outlier_stdevs = outlier_stdevs
        self.data_arg = data
        self.report_cols = data_filters.keys()
        self.clean = clean
        self.patsy_colsmap = {}

        # Prepare logging
        self.add_logger(log_to, loglevel)

        # Retrieve data
        self.get_data(data, db_version, db_namespace, X_expr, y_expr, annot_expr, rosetta_filters)
        self.user_filter_keep = pd.Series(True, index=self._data.index)
        self.filter_missing_keep = pd.Series(True, index=self._data.index)

        # Get starting number of observations
        self.n_init = self.data.shape[0]

        self.filter_info_countcols = ["Outliers (trim)", "Outliers (warn)", "Missing Values", "User Filters"]
        # Log filtering along the way
        self.filter_info = pd.DataFrame(0, index=self.filter_info_countcols +
                                        ["Dropped", "Drop Reason"],
                                        columns=self.data.columns)
        self.filter_info.columns.name = "Column name"
        self.filter_info.loc["Dropped"] = False
        self.filter_info.loc["Drop Reason"] = "N/A"
        self.filter_null_col = self.filter_info.iloc[:, 0].copy()

        # Save some miscellaneous informations
        self.n_obs_orig = self._data.shape[0]
        self.interactions = interactions
        self.col_map_reverse = {}

        # make sure you keep track of the columns included in interaction terms
        self.patsy_cols_non_formula = []
        self.patsy_cols = []

        assert type(self.interactions) in [list, tuple], "`interactions` should be a tuple or a list"
        for xx in self.interactions:
            variables = xx.split(":")
            assert len(variables) == 2, "Each interaction should contain a `:` separating two variables"
            self.patsy_cols_non_formula.extend(variables)

        # Retrieve X columns
        self.X_cols = self.get_cols(self.data.columns, X_expr, X_exprs_inv)
        assert len(self.X_cols) >= 1, "No columns retrieved for X"

        # Retrieve Y columns
        self.y_cols = self.get_cols(self.data.columns, y_expr, y_exprs_inv)
        assert len(self.y_cols) == 1, "Currently only one y column is supported (observed %s): %s" % (len(self.y_cols), ", ".join(self.y_cols))

        # Retrieve annotation files
        self.annot_cols = self.get_cols(self.data.columns, annot_expr, annot_expr_inv)

        self.outliers = outliers.Outliers(self.logger, self.filter_info, self.outlier_stdevs)

        if self.clean:
            self.clean_data()

    @property
    def data(self):
        """
        Return filtered underlying data
        """
        # TODO set self.data to an object w __setitem__ method so that self.data[col] = val works properly
        return self._data[self.user_filter_keep & self.filter_missing_keep]

    def get_data(self, data, db_version, db_namespace, X_expr, y_expr, annot_expr, filters, default="user"):
        """
        Purpose: Use supplied data or retrieve data from database
        Args:
            As documented in `__init__`
        """
        rdb_data = None
        if (X_expr is not None) or (y_expr is not None):
            self.logger.debug("Reading from Rosetta")
            rdb = RosettaDBMongo(host=ros_vars.ROSETTA_URL)
            if db_version is None:
                self.logger.info("No version specified. Using latest version of Rosetta...")
            self.logger.info("Using namespace %s" % db_namespace)

            rdb.initialize(version=db_version, namespace=db_namespace)

            # Get regex
            exprs = ["(%s)" % xx for xx in [X_expr, y_expr, annot_expr] if xx is not None]
            rdb_expr = "|".join(exprs)

            cols = rdb.find_keys(rdb_expr, regex=True)
            self.logger.debug("Retrieving data from rosetta...")
            rdb_data = rdb.query(keys=cols, filters=filters)
            self.logger.debug("Found %s records with %s variables" % rdb_data.shape)

        # Drop columns that overlap between rosetta dataframe and supplied dataframe. Default to user supplied version
        if data is not None and rdb_data is not None:
            for cc in rdb_data.columns:
                if cc in data.columns:
                    if default == "user":
                        self.logger.info("Column %s also present in supplied dataframe. Dropping Rosetta version...")
                        del rdb_data[cc]
                    else:
                        self.logger.info("Column %s also present in supplied dataframe. Dropping user version...")
                        del data[cc]
        self._data = pd.concat([xx for xx in [data, rdb_data] if xx is not None], axis=1)

    def add_logger(self, log_to, loglevel):
        """
        Add a logger object to handle output.
        Args:
            As described in `__init__`
        """
        self.logger = logging.getLogger()
        self.logger.setLevel(loglevel)

        if log_to == "stream":
            self.logger.handlers = []  # this seems to set a global variable, so if you re-run, don't keep adding
            self.logger.addHandler(logging.StreamHandler())
        else:
            raise Exception("log_to option %s is not recognized. Currently only logs to STDOUT (log_to='stream')." % log_to)

    def delete_records(self, col):
        self.X_cols = [xx for xx in self.X_cols if xx != col]
        self.y_cols = [yy for yy in self.y_cols if yy != col]
        self.patsy_cols = [pp for pp in self.patsy_cols if pp != col]

    def delete_col(self, col, drop_reason="Not specified", report_exprs=["missing$"]):
        """
        Delete specified column (if it exists), updating appropriate object attributes.
        """
        if col in self.data.columns:
            self.logger.info('%sDeleting column: %s' % ("\t" * 9, col))
            del self._data[col]
            self.delete_records(col)

            # If dropped column is still reportable, log it
            if [xx for xx in report_exprs if re.search(xx, drop_reason)]:
                self.report_cols.append(col)
            self.filter_info.ix["Dropped", col] = True
            self.filter_info.ix["Drop Reason", col] = drop_reason
        self.X_cols = [xx for xx in self.X_cols if xx != col]
        self.y_cols = [yy for yy in self.y_cols if yy != col]

    def add_col(self, col, kind="X"):
        """
        Handle updating of metadata when column is added
        """
        if kind == "X":
            if col not in self.X_cols:
                self.X_cols.append(col)
                self.filter_info[col] = self.filter_null_col

    def set_feature(self, col, value):
        """
        Add a new feature to the pipeline

        """
        self.add_col(col)
        toadd = value.index.intersection(self._data.index)
        self._data[col] = np.nan
        self._data.ix[toadd, col] = value[toadd]

    def drop_missing(self, cc, missingness_cut, drop_cut):
        """
        Drop a column if it is missing too much data.
        """
        missingness = (self.data[cc].isnull() | pd.Series(map(lambda x: x is None, self.data[cc]))).mean()

        if missingness >= drop_cut:
            self.logger.info("Dropping column %s due to missing data" % cc)
            self.filter_info.ix[self.filter_info_countcols, cc] = np.nan
            self.delete_col(cc, drop_reason="%.1f%% missing" % (missingness * 100))
        elif missingness >= missingness_cut:
            self.logger.warn("WARNING: %.2f%% of data is missing for %s. Should this variable be filtered?" % (missingness, cc))

    def get_cols(self, columns, expr, exprs_inv):
        """
        Retrieve a subset of columns based on regular expressions

        Args:
            columns: A list of column names
            expr: Regular expression to match
            exprs_inv: Regular expressions used to filter
        """
        if expr:
            cols = [xx for xx in columns if re.search(expr, xx)]
        else:
            return []
        for ee in exprs_inv:
            cols = [xx for xx in cols if not re.search(ee, xx)]
        return cols

    def get_badvals(self, col):
        """
        For a given column, return non-numeric values.
        """

        bad_vals = []
        for val in self._data[col]:
            try:
                float(val)
            except:
                if val not in bad_vals:
                    bad_vals.append(val)
        return bad_vals

    def type_salvageable(self, col):
        """
        Determine whether a putatively non-numeric column has a decent proportion of numeric values
        """
        salvage_count = 0
        non_missing_count = 0
        tot_count = 0
        for val in self.data[col]:
            try:
                if float(val) not in [np.nan, None]:
                    non_missing_count += 1
                salvage_count += 1
            except:  # if can't be coerced to float, skip it
                pass
            tot_count += 1

        return 1. * salvage_count / tot_count, non_missing_count

    def check_outliers(self, cols=None):
        """
        Determine whether columns contain outliers
        """
        self.logger.info("\n=== Checking for outliers ===")
        if cols is None:
            cols = self.X_cols + self.y_cols

        self.outliers.check(self.data, self._data, cols, self.trim_outliers)

    def code_categorical(self, cols=None):
        """
        Use patsy to code categorical variables.
        """

        self.logger.info("\n=== Handling categorical data and interactions ===")
        # The interaction term will automatically handle insertion of the individual terms.
        # Including both seems to be buggy
        self.patsy_cols = list(set(self.patsy_cols) - set(self.patsy_cols_non_formula))

        if cols is not None:
            self.patsy_cols = [xx for xx in self.patsy_cols if xx in cols]

        if not self.patsy_cols + self.patsy_cols_non_formula:
            return
        patsy_data = self.data[self.patsy_cols + self.patsy_cols_non_formula]

        cols_orig = patsy_data.columns
        # Remove illegal characters from column names
        patsy_illegal_chars = [".", "*", "+", ":", "-", "~", "/", " "]
        replace_with = ["_"] * len(patsy_illegal_chars)

        # save mapping
        cols_patsy = [replace_all(cc, patsy_illegal_chars, replace_with) for cc in cols_orig]
        self.col_map_forward = dict(zip(cols_orig, cols_patsy))
        self.col_map_reverse = dict(zip(cols_patsy, cols_orig))

        # convert column names
        patsy_data.columns = cols_patsy

        # For all interactions, split string, map column names, then put them back together
        self.interactions_patsy = [
            ":".join(
                [self.col_map_forward[yy] for yy in xx.split(":")]
            )
            for xx in self.interactions
        ]
        filter_cols = np.unique([self.col_map_forward[xx] for xx in self.patsy_cols + self.patsy_cols_non_formula]).tolist()

        keep_rows = self.tally_missing(patsy_data, filter_cols, colmap=self.col_map_reverse)
        patsy_data = patsy_data[keep_rows]

        # Generate formula
        x_str = " + ".join([self.col_map_forward[xx] for xx in self.patsy_cols] + self.interactions_patsy)
        formula = "%s" % (x_str)

        # retrieve X and y matrices
        # NAAction(NA_types=[]) is a hack to keep patsy from dropping rows that have missing values
        ans = dmatrix(str(formula), patsy_data, return_type="dataframe", NA_action=NAAction(NA_types=[]))  # patsy sometimes errors out if string is unicode
        transformed_cols = ans.design_info.column_name_indexes.keys()
        cat_df = pd.DataFrame(ans, columns=transformed_cols, index=patsy_data.index)

        # Get a map from input column names to those patsy creates
        new_cols_first = {}
        for xx in ans.design_info.term_name_slices:
            if xx != "Intercept":
                to_cols = ans.design_info.column_names[ans.design_info.term_name_slices[xx]]
                self.patsy_colsmap[xx] = to_cols
                if to_cols[0] != xx:
                        # print "DEBUG", "Dropping", xx
                        # self.delete_col(xx, drop_reason="Recoding")
                        new_cols_first[to_cols[0]] = xx

        del cat_df["Intercept"]
        # Convert column names back.
        # For interaction terms, patsy generates new columns (names not in dictionary). These are left "as is".
        cat_df.columns = [self.col_map_reverse.get(xx, xx) for xx in cat_df.columns]

        # TODO: We are deleting and re-adding columns (components of interaction terms) that we could leave untouched if we put in some more work.
        for cc in self.patsy_cols + self.patsy_cols_non_formula + self.interactions:
            self.delete_col(cc, drop_reason="Recoding")
        self._data = pd.concat([self.data, cat_df], join="outer", axis=1)

        # Record new column names that patsy may have created
        for xx in cat_df.columns:
            if xx not in self.col_map_reverse:
                self.add_col(xx)
                if xx in new_cols_first:
                    if ":" not in new_cols_first[xx]:
                        self.filter_info.ix[:4, xx] = self.filter_info.ix[:4, new_cols_first[xx]]

    def fix_non_numeric(self, cols=None, max_categories=10, min_salvage=.90):
        """
        Detect non-numeric columns. Salvage where possible. Encode if you can. Otherwise drop.
        """
        tabs = "\n" + "\t" * 8
        self.logger.info("\n=== Coding data ===")

        if cols is None:
            cols = self.X_cols + self.y_cols

        for cc in cols:
            if self.data[cc].dtype.kind not in 'biufc':

                # Test whether this column has some numeric data in it
                salvage_per, non_missing_count = self.type_salvageable(cc)

                # If it does, replace non-numeric values and keep it
                if salvage_per > .90:
                    bad_vals = self.get_badvals(cc)
                    self.logger.info("\t- %s is a non-numeric type. Saving %i non-missing numeric values." % (cc, non_missing_count))
                    if bad_vals:

                        # Build up string
                        if len(bad_vals) > 10:
                            # make it clear that we're reporting only the first ten
                            add_on = ["..."]
                        else:
                            add_on = []

                        bad_vals_str = ['"%s"' % xx for xx in bad_vals[:10]] + add_on
                        self.logger.info(
                            "\t\t-> Setting the following values to missing (previewing first 10 or less): %s" % (tabs + tabs.join(bad_vals_str))
                        )
                        bad_val_map = dict(zip(bad_vals, [np.nan] * len(bad_vals)))
                        self._data[cc] = self.data[cc].replace(bad_val_map)  # replace bad values with missing

                    self._data[cc] = self._data[cc].astype(float)

                else:
                    # Otherwise, see whether it has a reasonable number of categories that you could recode
                    n_cats = len(self.data[cc].value_counts())
                    if n_cats > max_categories:
                        self.logger.warn("WARNING: Omitting %s (n. categories is %s. Max is %s)" % (cc, n_cats, max_categories))
                        del self.data[cc]

                        # Remove what you deleted. You could do this by x and y separately and not have to check both, but this is fast.
                        self.logger.debug("In fix_non_numeric:")
                        self.logger.info("Deleting columns %s" % cc)
                        self.X_cols = [xx for xx in self.X_cols if xx != cc]
                        self.y_cols = [yy for yy in self.y_cols if yy != cc]
                    else:
                        self.patsy_cols.append(cc)

    def drop_missing_cols(self, cols=None, missingness_cut=.4, drop_cut=1., max_categories=10):
        """
        Drop columns if there is too much missingness
        """
        self.logger.info("\n=== Dropping colums with high missingness ===")

        assert drop_cut >= missingness_cut, "Drop cutoff must be less than missingness cutoff"

        if cols is None:
            cols = self.X_cols + self.y_cols

        for cc in cols:
            self.drop_missing(cc, missingness_cut, drop_cut)

    def apply_user_filters(self):
        keep = pd.Series(True, index=self.data.index)
        for dd in self.data_filters:
            for ff in self.data_filters[dd]:
                val = ff[1]
                string = "self.data[dd] %s val" % (ff[0])
                keep_tmp = eval(string)
                keep_tot = keep.sum()
                keep = keep & (keep_tmp)
                keep_now_tot = keep.sum()
                lost = keep_tot - keep_now_tot
                self.filter_info.ix["User Filters", dd] += lost
        self.user_filter_keep = keep

    def correct_known_issues(self):
        """
        For data coming from rosetta, manually fix known issues.
        These include:
            - Fixing non-numeric data in facepheno.ancestry.afroamerica
            - Pooling infrequent ancestry categories
            - Extracting ivectors

        """
        self.logger.info("\n=== Correcting known issues ===")
        facepheno_cols = [xx for xx in self.data.columns if re.search("facepheno.ancestry", xx)]
        ivector_cols = [ii for ii in self.data.columns if re.search("voice\.i_vectors", ii)]

        if facepheno_cols:
            self.logger.info("  For `facepheno.ancestry`:")
            # Fill in missing values with zeros
            col = "facepheno.ancestry.afroamerica"
            if col in self.data.columns:
                bad_vals = self.get_badvals(col)
                self.logger.info("\t- Fixing non-numeric data in `facepheno.ancestry.afroamerica`")
                self._data[col] = self.data[col].replace(dict(zip(bad_vals, [np.nan] * len(bad_vals)))).astype(float)

            self.logger.info("\t- Replacing missing values with zeros")

            facepheno_rows = self.data[facepheno_cols].notnull().any(axis=1)

            for cc in facepheno_cols:
                self._data.ix[facepheno_rows, cc] = self.data.ix[facepheno_rows, cc].fillna(0)
            # Delete non-informative columns
            self.delete_col("facepheno.ancestry.percentage", drop_reason="Not needed")

            # Combine ancestry classes that are too fine-grained
            self.logger.info("\t- Pooling ancestry classes")
            asian_cols = [
                            "facepheno.ancestry.aleutian",
                            "facepheno.ancestry.chinese",
                            "facepheno.ancestry.indian",
                            "facepheno.ancestry.japanese",
                            "facepheno.ancestry.korean",
                            "facepheno.ancestry.south.asian",
                        ]
            asian_cols = [aa for aa in asian_cols if aa in self.data.columns]
            other_cols = [
                            "facepheno.ancestry.eskimo",
                            "facepheno.ancestry.hawaii",
                            "facepheno.ancestry.native.american",
                            "facepheno.ancestry.other"
                        ]
            other_cols = [oo for oo in other_cols if oo in self.data.columns]

            tabs = "\n" + "\t" * 8
            self.logger.info("\t\t-> combining the following into `facepheno.ancestry.asian`:%s" % (tabs + tabs.join(asian_cols)))
            self._data["facepheno.ancestry.asian"] = self.data[asian_cols].sum(axis=1)

            for aa in asian_cols:
                self.delete_col(aa, drop_reason="Recoding")

            self.logger.info("\t\t-> combining the following into `facepheno.ancestry.other_pooled`:%s" % (tabs + tabs.join(other_cols)))
            self._data["facepheno.ancestry.other_pooled"] = self.data[other_cols].sum(axis=1)

            for oo in other_cols:
                self.delete_col(oo, drop_reason="Recoding")

        # Convert ivectors from strings to matrix
        if ivector_cols:
            ivector_dfs = []
            for ii in ivector_cols:
                self.logger.info("Converting %s to matrix form." % ii)
                # Convert strings to numeric dataframe
                tokeep = self.data[ii].notnull()

                # Evaluate series of strings to list of vectors
                ivectors = self.data.ix[tokeep, ii].apply(lambda x: eval(x)).tolist()

                # convert list of vectors to numpy array, which is read into a pandas dataframe
                df = pd.DataFrame(np.array(ivectors), index=tokeep[tokeep].index)

                # Get column names right
                suffix = re.search("voice\.i_vectors\.(.*)", ii).group(1)
                df.columns = ["i_vectors.%s_%s" % (suffix, xx + 1) for xx in df.columns]

                # Update X_cols, filter info, etc.
                for dd in df.columns:
                    self.add_col(dd)

                # Remove string column
                self.delete_col(ii, drop_reason="Recoding")

                # Save dataframes for catenation
                ivector_dfs.append(df)
            self._data = pd.concat([self.data] + ivector_dfs, join='outer', axis=1)
        # TODO fix up facepheno.HLI_CALC_Ethnicity

    def append_rosetta_data(self):
        pass
        # update self.data_arg to None to make sure you correct known issues

    def clean_data(self, cols=None):
        """
        Clean data.
        """
        # Only do this when you first read from the database
        if (cols is None):
            if self.data_arg is None:
                self.correct_known_issues()
            self.apply_user_filters()
        self.drop_missing_cols(cols=cols)
        self.fix_non_numeric(cols=cols)
        self.code_categorical(cols=cols)
        self.check_outliers(cols=cols)

    def tally_missing(self, df, cols, colmap={}):
        not_missing = df[cols].notnull()
        keep_rows = not_missing.all(axis=1)

        # Log which variables lead to exclusion
        keep_tmp = pd.Series(True, index=df.index)
        keep_tot = df.shape[0]
        print_count = 0

        # Loop in order of increasing missingness
        not_missing_count = not_missing.sum(axis=0)
        not_missing_count.sort(ascending=True)

        self.logger.info("Tallying cumulative observations excluded (columns sorted by decreasing individual missingness...):")
        for ii in not_missing_count.index:
            keep_now = keep_tmp & (not_missing[ii])
            keep_now_tot = keep_now.sum()
            lost = keep_tot - keep_now_tot

            # Print columns that result in excluding rows
            if lost > 0:
                if print_count == 0:
                    modifier = ""
                else:
                    modifier = "additional "
                self.logger.info("Excluding %s %sobservations due to %s" % (lost, modifier, ii))
                self.filter_info.ix["Missing Values", colmap.get(ii, ii)] = lost
                print_count += 1
            keep_tot = keep_now_tot
            keep_tmp = keep_now
        self.filter_missing_keep = self.filter_missing_keep & keep_rows
        return keep_rows

    def save_X_and_y(self):
        self.logger.info("\n=== Dropping missing values ===")
        # Find missing values

        keep_rows = self.tally_missing(self.data, self.X_cols + self.y_cols)

        self.X = self.data.ix[keep_rows, self.X_cols]
        self.y = self.data.ix[keep_rows, self.y_cols]

        self.X.columns = [self.col_map_reverse.get(xx, xx) for xx in self.X_cols]
        self.y.columns = [self.col_map_reverse.get(yy, yy) for yy in self.y_cols]

        n_dropped = self.n_obs_orig - self.X.shape[0]

        if n_dropped > 0:
            self.logger.warn("*** Dropped %s total observations ***" % n_dropped)
        self.logger.info("Returning %s regressors on %s observations" % (self.X.shape[1], self.X.shape[0]))

    def __call__(self):
        """
        Return a tuple of X and y
        """
        self.save_X_and_y()
        return self.X, self.y
