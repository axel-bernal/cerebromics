import datastack.cerebro.cross_validation as cross_validation
import datastack.settings as settings
import face as face_model

import logging
import matplotlib
import numpy as np
import pandas as pd
import sklearn.linear_model
import sklearn.metrics as metrics
import sklearn.preprocessing
import sklearn.utils as utils
import warnings

from datastack.dbs.rdb import RosettaDB
from mpl_toolkits.axes_grid1 import make_axes_locatable  # @UnresolvedImport
from pandas import DataFrame
from datastack.ml.outliers import Outliers

from  scipy.spatial.distance import cosine
import operator
import numbers
import os

import datastack.ml.metrics.select_n as select_n

import datastack.ml.rosetta_settings as ml_vars

from datastack.ml.unified import _Unified

import datastack.ml.metrics.select_n as select_n

from datastack.ml.algorithms.pca_covariates import CovariatesPCA


_REGRESSION_PARAMS_DEFAULT = {
    'alpha': [
        0.0001,
        0.0005,
        0.001,
        0.005,
        0.01,
        0.05,
        0.1,
        0.5,
        1,
        5,
        10]}

_logger = logging.getLogger(__name__)


def robust_r2(y, y_pred, **kwargs):
    """
    Subsample the data, calculating r2 each time.
    This will omit outliers.

    **Parameters**
    ***
    **y** True values
    **y_pred** Predicted values
    **kwargs** Passed to the outlier detector

    ** Returns **
    ***
    **r2** float: a robust r2 score
    """
    outlier_detector = Outliers(**kwargs)
    over, under, _, _ = outlier_detector.trim_vals(y - y_pred, **kwargs)
    keep = (over | under).values
    return metrics.r2_score(y[keep], y_pred[keep],multioutput='variance_weighted')


def r2_score(y, y_pred, robust=False, robust_kwargs={}):
    if len(y.shape) > 1:
        r2=[]
        for j in range(0,y.shape[1]):
            if robust:
                r2.append(y.iloc[:,j],y_pred.iloc[:,j], **robust_kwargs)
            else:
                r2.append(metrics.r2_score(y.iloc[:,j], y_pred.iloc[:,j],multioutput='variance_weighted'))
        return r2
    else:
        if robust:
            return robust_r2(y, y_pred, **robust_kwargs)
        else:
            return metrics.r2_score(y, y_pred,multioutput='variance_weighted')


def log_loss(model, X, y_true, eps=1e-15, normalize=True, sample_weight=None):
    """Log loss, aka logistic loss or cross-entropy loss.
    This is the loss function used in (multinomial) logistic regression
    and extensions of it such as neural networks, defined as the negative
    log-likelihood of the true labels given a probabilistic classifier's
    predictions. For a single sample with true label yt in {0,1} and
    estimated probability yp that yt = 1, the log loss is
        -log P(yt|yp) = -(yt log(yp) + (1 - yt) log(1 - yp))

    The logarithm used is the natural logarithm (base-e).
    More or less a copy of sklearn.metrics.log_loss, except that it will
    deal with the case where a training set contains fewer labels than
    the test set. This can happen if the input data set to the regression
    model is very unbalanced.

    Arg:
        y_true : array-like or label indicator matrix
            Ground truth (correct) labels for n_samples samples.
        y_pred : array-like of float, shape = (n_samples, n_classes)
            Predicted probabilities, as returned by a classifier's
            predict_proba method.
        eps : float
            Log loss is undefined for p=0 or p=1, so probabilities are
            clipped to max(eps, min(1 - eps, p)).
        normalize : bool, optional (default=True)
            If true, return the mean loss per sample.
            Otherwise, return the sum of the per-sample losses.
        sample_weight : array-like of shape = [n_samples], optional
            Sample weights.

    Returns:
        loss : float

    Examples:
        >>> log_loss(["spam", "ham", "ham", "spam"],  # doctest: +ELLIPSIS
        ...          [[.1, .9], [.9, .1], [.8, .2], [.35, .65]])
        0.21616...

    References:
        C.M. Bishop (2006). Pattern Recognition and Machine Learning.
        Springer, p. 209.
    """
    # Use label_binarize function instead of the LabelBinarizer class.
    T = sklearn.preprocessing.label_binarize(y_true, model.classes_)
    if T.shape[1] == 1:
        T = np.append(1 - T, T, axis=1)

    # Clipping
    Y = np.clip(model.predict_proba(X), eps, 1 - eps)

    # This happens in cases when elements in y_pred have type "str".
    if not isinstance(Y, np.ndarray):
        raise ValueError("y_pred should be an array of floats.")

    # If y_pred is of single dimension, assume y_true to be binary
    # and then check.
    if Y.ndim == 1:
        Y = Y[:, np.newaxis]
    if Y.shape[1] == 1:
        Y = np.append(1 - Y, Y, axis=1)

    # Check if dimensions are consistent.
    utils.check_consistent_length(T, Y)
    T = utils.check_array(T)
    Y = utils.check_array(Y)
    # In our version this is a more serious error, as it indicates that
    # the model supplied is not consistent with the array of true y
    # values.
    if T.shape[1] != Y.shape[1]:
        raise ValueError("y_true and model have different number of classes "
                         "%d, %d" % (T.shape[1], Y.shape[1]))

    # Renormalize
    Y /= Y.sum(axis=1)[:, np.newaxis]
    loss = -(T * np.log(Y)).sum(axis=1)

    return metrics.classification._weighted_sum(loss, sample_weight, normalize)

# selected 10 closed for approximation for the face phenotypes
def identify(_id, t_y, p_y):
    mse_dict = {}
    for i in range(len(t_y)):
        mse = cosine(p_y[_id],t_y[i])
        mse_dict[i] = mse

    sorted_x       = sorted(mse_dict.items(), key=operator.itemgetter(1))
    rank           = [ii[0] for ii in sorted_x].index(_id)

    return rank


def rank_all(t_y, p_y, distance="cosine", n=10, replacement=False):
    """
    compute the average select@n statistic over a test set

    Args:
        t_y         True values
        p_y         predicted values
        distance    "Euclidean" or "cosine" implemented (default: "cosine")
        n           n of select@n (default 10)
        replacement sample other people with replacement? (default False)
                    False uses a hypergeometric distribution
                    True uses a binomial distribution (less accurate for small test set)

    """
    return select_n.select_at_n(t_y, p_y, distance=distance, n=n, replacement=replacement)



def rank_all(t_y, p_y, distance="cosine", n=10, replacement=False):
    """
    compute the average select@n statistic over a test set

    Args:
        t_y         True values
        p_y         predicted values
        distance    "Euclidean" or "cosine" implemented (default: "cosine")
        n           n of select@n (default 10)
        replacement sample other people with replacement? (default False)
                    False uses a hypergeometric distribution
                    True uses a binomial distribution (less accurate for small test set)

    """
    return select_n.select_at_n(t_y, p_y, distance=distance, n=n, replacement=replacement)


class KFoldPredefined(cross_validation.HashedKfolds):
    """Generate k-folds from Rosetta subjects. Users can pass instances
    of this k-fold class to cross-validation classes from scikit-learn.
    """

    def __init__(
        self,
        rdb=None,
        data=None,
        keys=[],
        filters=None,
        labels=[],
        n_folds_training=10,
        n_folds_holdout=2,
        keep_in_training_columns=None,
        keep_in_holdout_columns=settings.QUICKSILVER_KFOLDS_HOLDOUT_COLUMNS,
        keep_together_columns=settings.QUICKSILVER_KFOLDS_TOGETHER_COLUMNS,
        salt=None,
        with_nans=False,
        *args, **kwargs
    ):
        """
        Construct a new k-fold generator instance. The constructor
        will select records from Rosetta and partition the records into
        a set of folds for training and cross-validation of machine
        learning models, and a set of folds for testing trained models.

        The constructor can group related subjects together in the same
        fold, using an optional list of related subjects for each
        subject. All such lists must be total, e.g., if A is related to
        B is related to C, then A must list B and C, B must list A and C,
        and C must list A and B.

        Args:
            rdb:    A handle to an open Rosetta DB connection; by default,
                    None
            data:   A dataframe of Rosetta-like records; by default, None
            keys:   A list of Rosetta columns to select. The generator
                    always selects a unique subject ID column; typically, the
                    user adds additional phenotype columns.
            filters: A dictionary of filters to apply when selecting
                    columns from Rosetta
            labels: An exclusive list of HLI labels on which to
                    filter records; by default, the generator will return all
                    records from all labels
            n_folds_training: The number of training folds to
                    generate, by default, 10
            n_folds_holdout: The number of test folds to generate and
                    hold out when the generator is used in cross-validation
            keep_in_training_columns: A list of columns with
                boolean values. If any of the values in the columns is
                true for a row, then include the subject in that row
                in the training set (and never put it in the holdout
                set).
            keep_in_holdout_columns: A list of columns with
                boolean values. If any of the values in the columns is
                true for a row, then include the subject in that row
                in the holdout set (and never put it in the training
                set).
            keep_together_columns: The column in the dataframe that specifies
                lists of subjects related to the current subject; by
                default, `None` (i.e., do not consider relations). If
                used, every related subject must specify all of the other
                related subjects in a set of related subjects; for
                example, if `A`, `B`, and `C` are related, then `A` must
                list `B` and `C`, `B` must list `A` and `C`, and `C` must
                list `A` and `B`.
            experiments; by default, 2
            salt:   A text salt to seed the randomization algorithm used
                    to partition records
            with_nans: Include records with `null` values in selected
                    columns; by default, exclude such records
                    This parameter only applies when data=None


        The user may pass through any legal arguments to the Rosetta DB
        `query` method.
        """
        warnings.warn('The old k-folds interface is going away Real Soon Now; use datastack.quicksilver.Kfold instead', UserWarning)
        if not ((rdb is None) ^ (data is None)):
            raise Exception('Expected exactly one of rdb or data to be None')
        if rdb is not None:
            raise Exception('The old k-folds interface no longer supports querying Rosetta')
        if (settings.ROSETTA_INDEX_KEY not in data.columns) and (settings.ROSETTA_SUBJECT_NAME_KEY not in data.columns):
            raise KeyError('Missing %s or %s in the dataframe' % (settings.ROSETTA_INDEX_KEY, settings.ROSETTA_SUBJECT_NAME_KEY))
        if (settings.ROSETTA_INDEX_KEY in data.columns) and (settings.ROSETTA_SUBJECT_NAME_KEY not in data.columns):
            # For hashing, only use the subject name, not the sample key
            _logger.debug('Interpolating a subject name from %s' % (settings.ROSETTA_INDEX_KEY))
            data[settings.ROSETTA_SUBJECT_NAME_KEY] = data[settings.ROSETTA_INDEX_KEY].apply(lambda k: k.split('_')[1])
        # Slide the new kfold generator underneath
        super(KFoldPredefined, self).__init__(
            df=data,
            index_column=settings.ROSETTA_SUBJECT_NAME_KEY,
            n_training=n_folds_training,
            n_holdout=n_folds_holdout,
            keep_in_training_columns=keep_in_training_columns,
            keep_in_holdout_columns=keep_in_holdout_columns,
            keep_together_columns=keep_together_columns,
            salt=salt)
        self.df_all = self._df
        self.n_folds_training = self.n_training
        self.n_folds_holdout = self.n_holdout
        self.salt = salt

    def __repr__(self):
        return '%s(n_folds_training=%i, n_folds_holdout=%i, salt=%s)' % (
            self.__class__.__name__,
            self.n_folds_training,
            self.n_folds_holdout,
            self.salt,
        )

    def get_data(self, xy_generator):
        """Get X and y training data matrices from the records selected
            from Rosetta. The X and y matrices will contain the training
            records but not the hold-out records.

            Args:
                xy_generator: A function that takes in a dataframe of
                    selected records from Rosetta, and returns a pair of
                    dataframes containing the X and y matrices.

            Returns:
                X and y matrices containing the data from the training
                folds
            """
        return xy_generator(self.df)

    def get_data_count(self):
        '''Get the total number of samples in the training folds
            '''
        return self.df.shape[0]

    def get_data_holdout(self, xy_generator):
        """Get X and y hold-out data matrices from the records selected.

            Args:
                xy_generator:   A function that takes in a dataframe o selected
                        records from Rosetta, and returns a pair of dataframes
                        containing the X and y matrices. Note that this function can
                        be the same as the one passed to `get_data`.

            Returns
                X and y matrices containing the data from the hold-out folds
            """
        return xy_generator(self.df_holdout)

_CLASSIFICATION_ESTIMATOR_DEFAULT = sklearn.linear_model.LogisticRegression()
_CLASSIFICATION_PARAMS_DEFAULT = {
    'C': [
        0.0001,
        0.0005,
        0.001,
        0.005,
        0.01,
        0.05,
        0.1,
        0.5,
        1,
        10,
        100,
        1000,
        10000]}

class CrossValidationClassification(_Unified):
    """Generic cross-validation test class for classification estimators.
    The class produces score vs. candidate classifier parameters curves
    for each split in a set of k-folds, and selects the classifier
    parameters yielding the highest score for the supplied estimator and
    training data.

    The test class should work with any classification estimator that
    conforms to the `sklearn.linear_model` API, such as
    `sklearn.linear_model.LogisticRegression'
    """

    class _Scorer():

        def __call__(self, clf, X, y, sample_weight=None):
            """Evaluate log-loss for X relative to y_true given a trained
            classifier for y on X. More or less a copy of
            sklearn.scorer._ProbaScorer.

            Parameters
            ----------
            clf : object
                Trained classifier to use for scoring.

            X : array-like or sparse matrix
                Test data covariate values

            y : array-like
                Observed target values

            sample_weight : array-like, optional (default=None)
                Sample weights.

            Returns
            -------
            score : float
                Either a negated log-loss computed on the prediction
                probabilities of classes from X, or the accuracy of
                predicted classes from X
            """
            if hasattr(clf, 'predict_proba'):
                if (not hasattr(clf, 'really_has_proba')) or clf.really_has_proba():
                    return -1.0 * log_loss(clf, X, y, sample_weight=sample_weight)
            return metrics.accuracy_score(
                    y, clf.predict(X), sample_weight=sample_weight)

    def __init__(self,
                 x_and_y,
                 kfolds=None,
                 estimator=_CLASSIFICATION_ESTIMATOR_DEFAULT,
                 nopreproc=False,
                 params=_CLASSIFICATION_PARAMS_DEFAULT,
                 fit_params=None,
                 scoring=_Scorer(),
                 *args,
                 **kwargs):
        """Run a cross-validation experiment with the supplied classifier
        estimator class. The constructor will loop over all candidate
        parameter value combinations and k-folds to select the classifier
        model yielding the highest score for the supplied estimator and
        training data.

        By default, the cross-validator uses a hybrid scorer to choose
        the best-fit hyperparameter values. If the underlying estimator
        is probability-based, the validator selects values that minimize
        the log loss. Otherwise, it selects values that maximize the
        prediction accuracy.

        x_and_y: **Either** a function that takes in a dataframe
        of selected records, and that returns a pair of dataframes
        containing the X and y matrices, **or** a tuple of X and y
        matrices
        kfolds: A predefined k-fold split of the data; by default,
        `None`
        estimator: An instance of an classifier estimator class
        that implements `fit` and `predict`; default,
        `sklearn.linear_model.LogisticRegression`
        params: A dictionary or list of dictionaries of candidate
        parameter values, as described in
        `sklearn.grid_search.GridSearchCV`; by default, a set of `C`
        values for `sklearn.linear_model.LogisticRegression`
        scorer: A scorer callable object / function with signature
        `scorer(estimator, X, y)`; default `_Scorer()`

        The user may pass through any legal arguments to
        `sklearn.grid_search.GridSearchCV`.
        """
        if estimator is None:
            estimator = _CLASSIFICATION_ESTIMATOR_DEFAULT
        if params is None:
            params = _CLASSIFICATION_PARAMS_DEFAULT
        if scoring is None:
            scoring = self._Scorer()
        super(
            CrossValidationClassification,
            self).__init__(
            x_and_y,
            kfolds=kfolds,
            estimator=estimator,
            isRegression=False,
            nopreproc=nopreproc,
            params=params,
            fit_params=fit_params,
            scoring=scoring,
            *args,
            **kwargs)

    def plot_observed_vs_predicted(
            self, axes, target, y=None, y_pred=None, axis_labels=True, colormap=None):
        """Plot a confusion matrix of observed target values from the
        training set against predicted target values computed using the
        training covariate values and the trained classifier.
        """
        # Stolen shamelessly from http://stackoverflow.com/questions/18266642/multiple-imshow-subplots-each-with-colorbar
        # to get a colorbar next to the confusion plot.
        if colormap is None:
            colormap = matplotlib.cm.RdBu_r  # @UndefinedVariable
        if y_pred is None:
            y_pred = self.get_predicted()

        labels = utils.multiclass.unique_labels(self.y, y_pred)
        cm = metrics.confusion_matrix(self.y, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        im = axes.imshow(cm_normalized, interpolation='nearest', cmap=colormap)
        divider = make_axes_locatable(axes)
        colorbar_axes = divider.append_axes("right", size="20%", pad=0.05)
        matplotlib.pyplot.colorbar(
            im,
            cax=colorbar_axes,
            ticks=matplotlib.ticker.MultipleLocator(0.2),
            format="%.2f")
        tick_marks = np.arange(len(labels))
        axes.set_ylabel(
            'Observed ' +
            target if axis_labels is True else 'Observed')
        axes.set_yticks(tick_marks)
        axes.set_yticklabels(labels)
        axes.set_xlabel(
            'Predicted ' +
            target if axis_labels is True else 'Predicted')
        axes.set_xticks(tick_marks)
        axes.set_xticklabels(labels, rotation=45)
        axes.axis('tight')

class CrossValidationRegression(_Unified):
    """Generic cross-validation test class for linear regression
    estimators. The class produces score, MSE, and explained variance vs.
    candidate alpha value curves for each split in a set of k-folds, and
    selects the regression model coefficients yielding the highest score
    for the supplied estimator and training data.

    The test class should work with any linear regression estimator that
    conforms to the `sklearn.linear_model` API, such as
    `sklearn.linear_model.Ridge` and `sklearn.linear_model.Lasso`.
    """

    def __init__(self,
                 x_and_y,
                 kfolds=None,
                 estimator=None,
                 nopreproc=False,
                 params=_REGRESSION_PARAMS_DEFAULT,
                 fit_params=None,
                 *args,
                 **kwargs):
        """Run a cross-validation experiment with the supplied regression
        estimator class. The constructor will loop over all candidate
        parameter value combinations and k-folds to select the regression
        model yielding the highest score for the supplied estimator and
        training data.

        **Parameters**
        ***
        **x_and_y** **Either** a function that takes in a dataframe
        of selected records, and that returns a pair of dataframes
        containing the X and y matrices, **or** a tuple of X and y
        matrices
        **kfolds** : split
            A predefined k-fold split of the data; by default,`None`
        **estimator** An instance of an classifier estimator class
            implements `fit` and `predict`; default, datastack.ml.algorithms.linreg.Ridge,
            which should be essentially a superset of sklearn.linear_model.Ridge
        **params** A dictionary or list of dictionaries of candidate
        parameter values, as described in
        `sklearn.grid_search.GridSearchCV`; by default, a set of `alpha`
        values for `sklearn.linear_model.Ridge`

        The user may pass through any legal arguments to
        `sklearn.grid_search.GridSearchCV`.
        """
        if params is None:
            params = _REGRESSION_PARAMS_DEFAULT
        super(
            CrossValidationRegression,
            self).__init__(
            x_and_y,
            kfolds=kfolds,
            estimator=estimator,
            nopreproc=nopreproc,
            params=params,
            fit_params=fit_params,
            *args,
            **kwargs)

    def _min (self, y):
        """
        Could be a frame, a series, whatever.
        """
        while not isinstance(y, numbers.Number):
            y = y.min()

    def _max (self, y):
        while not isinstance(y, numbers.Number):
            y = y.max()

    def plot_observed_vs_predicted(
            self, axes, target, y=None, y_pred=None, axis_labels=True, y_lim=None, *args, **kwargs):
        """Plot observed target values from the training set against
        predicted target values computed using the training covariate
        values and the trained estimator.

        **Parameters**
        ***
        **axes** The place to plot the results. If you don't know what should go here,
        plt.gca() is usually a good choice if you use the usual matplotlib imports.
        **target** A label to apply to the axes.
        **y** Observed values. Default is observed values from the training data.
        **ypred** Predicted values. Default is cross validated predictions from the training data.
        **axis_labels** Default True.
        **y_lim** Explicit limit for x and y axes. Default None.
        """
        if y is None:
            y = self.y
        if y_pred is None:
            if self.y_pred is None:
                self.get_predicted()
            y_pred = self.y_pred

        ym0 = self._min(y)
        ym1 = self._min(y_pred)
        y_min = min(ym0, ym1)
        yx0 = self._max(y)
        yx1 = self._max(y_pred)
        y_max = max(yx0, yx1)
        axes.scatter(y_pred, y)
        axes.plot([y_min, y_max], [y_min, y_max])
        axes.set_xlabel(
            'Predicted ' +
            target if axis_labels is True else 'Predicted')
        axes.set_xlim(y_lim)
        axes.set_ylabel(
            'Observed ' +
            target if axis_labels is True else 'Observed')
        axes.set_ylim(y_lim)
        axes.axis('tight')

    def plot_observed_vs_predicted_column(
                self, axes, target, y=None, y_pred=None,column=0, axis_labels=True, y_lim=None, *args, **kwargs):
        """Plot observed target values from the training set against
        predicted target values computed using the training covariate
        values and the trained estimator.

        **Parameters**
        ***
        **axes** The place to plot the results. If you don't know what should go here,
        plt.gca() is usually a good choice if you use the usual matplotlib imports.
        **target** A label to apply to the axes.
        **y** Observed values. Default is observed values from the training data.
        **ypred** Predicted values. Default is cross validated predictions from the training data.
        **axis_labels** Default True.
        **y_lim** Explicit limit for x and y axes. Default None.
        """
        if y is None:
            y = self.y
        if y_pred is None:
            if self.y_pred is None:
                self.get_predicted()
            y_pred = self.y_pred

        y = y.iloc[:,column]
        y_pred =  y_pred.iloc[:,column]
        y_min = min(y.min(), y_pred.min())
        y_max = max(self.y.max(), y_pred.max())
        axes.scatter(y_pred, y)
        axes.plot([y_min, y_max], [y_min, y_max])
        axes.set_xlabel(
                'Predicted ' +
                target if axis_labels is True else 'Predicted')
        axes.set_xlim(y_lim)
        axes.set_ylabel(
                'Observed ' +
                target if axis_labels is True else 'Observed')
        axes.set_ylim(y_lim)
        axes.axis('tight')

    def plot_observed_vs_predicted_old(
            self, axes, target, y=None, y_pred=None, axis_labels=True, y_lim=None, *args, **kwargs):
        """Plot observed target values from the training set against
        predicted target values computed using the training covariate
        values and the trained estimator.
        """
        if y is None:
            y = self.y
        if y_pred is None:
            y_pred = self.grid.predict(self.X)
        y_min = min(y.min(), y_pred.min())
        y_max = max(self.y.max(), y_pred.max())
        axes.scatter(y_pred, y)
        axes.plot([y_min, y_max], [y_min, y_max])
        axes.set_xlabel(
            'Predicted ' +
            target if axis_labels is True else 'Predicted')
        axes.set_xlim(y_lim)
        axes.set_ylabel(
            'Observed ' +
            target if axis_labels is True else 'Observed')
        axes.set_ylim(y_lim)
        axes.axis('tight')

class EvaluationClassification(_Unified):
    """Evaluate a classification estimator using double-loop cross-
    validation over k-folds. We treat each training-test split from the
    k-folds as an independent data set; we use the training split as the
    input to an inner cross-validation of the estimator, and get the best
    prediction score over the test split.
    """

    _CROSS_VALIDATION_DEFAULT = CrossValidationClassification

    def __init__(self,
                 xy_generator,
                 kfolds,
                 isCvobj = False,
                 nopreproc=False,
                 estimator=_CLASSIFICATION_ESTIMATOR_DEFAULT,
                 params=_CLASSIFICATION_PARAMS_DEFAULT,
                 fit_params=None,
                 ordinals=None,
                 pspec=None,
                 *args,
                 **kwargs):
        """Evaluate a classification estimator using double-loop cross-
        validation over k-folds.

        xy_generator: A function that takes in a dataframe of
                selected records, and that returns a pair of dataframes
                containing the X and y matrices
        kfolds:     A predefined k-fold split of the data
        estimator:  An instance of an classifier estimator class
                that implements `fit` and `predict`; by default,
                `sklearn.linear_model.LogisticRegression`
        params: A dictionary or list of dictionaries of candidate
                parameter values, as described in
                `sklearn.grid_search.GridSearchCV`; by default, a set of `C`
                values for `sklearn.linear_model.LogisticRegression`
        ordinals:   A map of class labels to ordinal values; default
                `None`

        The user may pass through any legal arguments to
        `sklearn.grid_search.GridSearchCV`.
        """
        if estimator is None:
            estimator = _CLASSIFICATION_ESTIMATOR_DEFAULT
        if params is None:
            params = _CLASSIFICATION_PARAMS_DEFAULT
        generator = None
        if ordinals is not None:
            def generator(df):
                X, y = xy_generator(df)
                return X, y.apply(lambda v: ordinals[v])
        else:
            generator = xy_generator
        self.ordinals = ordinals
        # Classifiers based on linear regression estimators (e.g.
        # `sklearn.linear_model.RidgeClassifier` do not use probabilities
        # to classify from covariates.
        if hasattr(estimator, 'predict_proba'):
            self.log_loss_test_path = []
        self.metrics_df = None
        self.prfs_df = None
        super(
            EvaluationClassification,
            self).__init__(
            generator,
            kfolds,
            estimator=estimator,
            isCvobj=False,
            isRegression=False,
            params=params,
            fit_params=fit_params,
            pspec=pspec,
            nopreproc=nopreproc,
            *args,
            **kwargs)

    def _update_score_path(
            self, scorer, model, X_training, X_test, y_training, y_test):
        super(
            EvaluationClassification,
            self)._update_score_path(
            scorer,
            model,
            X_training,
            X_test,
            y_training,
            y_test)
        if hasattr(model, 'predict_proba'):
            self.log_loss_test_path.append(log_loss(model, X_test, y_test))

    def _update_metrics(self):
        if len(self.y.shape)>1:
            keep = pd.Series(
                        self.y.iloc[:,0]).notnull().values & pd.Series(
                        self.y_pred.iloc[:,0]).notnull().values
        else:
            keep = pd.Series(
            self.y).notnull().values & pd.Series(
            self.y_pred).notnull().values
        columns = ['Model', 'Covariates', 'Error', 'LogLoss']
        if self.ordinals is not None:
            columns += ['MAE', 'MSE']

        # If the estimator is a preprocessing wrapper around a base
        # estimator, want to use the underlying name.
        if hasattr(self.estimator, 'underlyingName'):
            cv_metrics = [self.estimator.underlyingName()]
        else:
            cv_metrics = [self.estimator.__class__.__name__]

        cv_metrics.append(self.X.columns.tolist())
        self.error = 1.0 - \
            metrics.accuracy_score(self.y[keep], self.y_pred[keep])
        cv_metrics.append(self.error)
        self.logloss = self.get_log_loss()
        cv_metrics.append(self.logloss)
        if self.ordinals is not None:
            self.mae = metrics.mean_absolute_error(
                self.y[keep],
                self.y_pred[keep])
            self.mse = metrics.mean_squared_error(
                self.y[keep],
                self.y_pred[keep])
            cv_metrics.append(self.mae)
            cv_metrics.append(self.mse)
        self.metrics_df = DataFrame([cv_metrics], columns=columns)

    def get_prfs(self):
        """Get a precision, recall, F-score, and support report for a
        classifier trained on the supplied training data.
        """
        if self.prfs_df is None:
            p, r, f1, s = metrics.precision_recall_fscore_support(
                self.y, self.y_pred, average=None)
            s_percent = s / float(sum(s))
            labels = utils.multiclass.unique_labels(self.y, self.y_pred)
            if self.ordinals is not None:
                # Recover the labels if we classified on ordinal values
                ordinals_reverse = {}
                for k, v in self.ordinals.items():
                    ordinals_reverse[v] = k
                rows = zip([ordinals_reverse[l]
                            for l in labels], labels, p, r, f1, s, s_percent)
                self.prfs_df = DataFrame(
                    rows,
                    columns=[
                        'label',
                        'ordinal',
                        'precision',
                        'recall',
                        'f1',
                        'support',
                        'support_fraction'])
            else:
                rows = zip(labels, p, r, f1, s, s_percent)
                self.prfs_df = DataFrame(
                    rows,
                    columns=[
                        'label',
                        'precision',
                        'recall',
                        'f1',
                        'support',
                        'support_fraction'])
        return self.prfs_df

    def get_log_loss(self):
        """Get the log-loss score for the estimator. We compute the score
        by returning the average of the log-loss scores for each training
        and test split, normalized by the size of the test set.
        """
        if hasattr(self.estimator, 'predict_proba'):
            return sum([v[0] * float(v[1]) for v in zip(self.log_loss_test_path,
                                                        self.test_size_test_path)]) / sum(self.test_size_test_path)
        else:
            return float('nan')

class EvaluationClassificationBaseline(object):

    def __init__(self, target, kfolds, ordinals=None):
        y = kfolds.get_data(lambda df: df[target])
        y_ordinals = None
        if ordinals is not None:
            ordinals_reverse = {}
            for k, v in ordinals.items():
                ordinals_reverse[v] = k
            y_ordinals = y.map(lambda v: ordinals[v])
        # Support will contain a list of (label, count) tuples, with the most
        # common label first.

        support = sorted([(label, y.values.tolist().count(label)) for label in set(
            y.values.tolist())], key=(lambda v: v[1]), reverse=True)

        rows = []
        columns = ['Model', 'Covariates', 'Error', 'LogLoss']
        if ordinals is not None:
            columns += ['MAE', 'MSE']
        #
        # Mode model: Always guess the most common class
        #
        mode_class = support[0]
        mode_metrics = [
            'ModeClass [\'%s\']' %
            (mode_class[0]) if ordinals is None else 'ModeClass [\'%s\', ordinal %d]' %
            (mode_class[0],
             ordinals[
                mode_class[0]])]
        mode_metrics.append([])
        mode_metrics.append(1.0 - (float(mode_class[1]) / len(y)))
        # Fake a prediction probability vector equivalent to the support
        # fractions for each class. We MUST make sure that the entries
        # correspond to the class order that will be used by the log_loss
        # method (i.e., the order created by
        # sklearn.preprocessing.LabelBinarizer
        lb = sklearn.preprocessing.LabelBinarizer()
        lb.fit_transform(y)
        y_pred_prob = [float(dict(support)[l]) / len(y) for l in lb.classes_]
        mode_metrics.append(
            metrics.log_loss(y, [y_pred_prob for _ in range(0, len(y))]))
        self.mode_y_pred = [mode_class[0] for _ in range(0, len(y))]
        if ordinals is not None:
            mode_y_pred = [ordinals[mode_class[0]] for _ in range(0, len(y))]
            mode_metrics.append(
                metrics.mean_absolute_error(
                    y_ordinals,
                    mode_y_pred))
            mode_metrics.append(
                metrics.mean_squared_error(
                    y_ordinals,
                    mode_y_pred))
        rows.append(mode_metrics)
        if ordinals is not None:
            #
            # MeanClass model: Always guess the mean class. Only good for
            # classification problems with ordinal class representations
            y_mean = y_ordinals.mean()
            mean_class = ordinals_reverse[np.round(y_mean)]
            mean_metrics = [
                'MeanClass [\'%s\', ordinal %.2f]' %
                (mean_class, y_mean)]
            mean_metrics.append([])
            mean_metrics.append(
                1.0 - (float(dict(support)[mean_class]) / len(y)))
            mean_metrics.append(None)
            self.mean_y_pred = [y_mean for _ in range(0, len(y))]
            mean_metrics.append(
                metrics.mean_absolute_error(
                    y_ordinals,
                    self.mean_y_pred))
            mean_metrics.append(
                metrics.mean_squared_error(
                    y_ordinals,
                    self.mean_y_pred))
            rows.append(mean_metrics)
            #
            # MedianClass model: Always guess the median class. Only good for
            # classification problems with ordinal class representations
            #
            y_median = y_ordinals.median()
            median_class = ordinals_reverse[np.round(y_median)]
            median_metrics = [
                'MedianClass [\'%s\', ordinal %d]' %
                (median_class, y_median)]
            median_metrics.append([])
            median_metrics.append(
                1.0 - (float(dict(support)[median_class]) / len(y)))
            median_metrics.append(None)
            self.median_y_pred = [y_median for _ in range(0, len(y))]
            median_metrics.append(
                metrics.mean_absolute_error(
                    y_ordinals,
                    self.median_y_pred))
            median_metrics.append(
                metrics.mean_squared_error(
                    y_ordinals,
                    self.median_y_pred))
            rows.append(median_metrics)
        self.metrics_df = DataFrame(rows, columns=columns)

class EvaluationRegression(_Unified):
    """Evaluate a linear regression estimator using double-loop cross-
    validation over k-folds. We treat each training-test split from the
    k-folds as an independent data set; we use the training split as the
    input to an inner cross-validation of the estimator, and get the best
    prediction score over the test split.
    """

    _CROSS_VALIDATION_DEFAULT = CrossValidationRegression

    def __init__(self,
                 xy_generator,
                 kfolds,
                 isCvobj = False,
                 estimator=None,
                 nopreproc=False,
                 params=_REGRESSION_PARAMS_DEFAULT,
                 fit_params=None,
                 idx_covariates=None,
                 ordinals=None,
                 pspec=None,
                 VERSION=None,
                 TYPE=None,
                 COVARIATES=None,
                 PREDICTED_COVARIATES=None,
                 HOLDOUT=False,
                 *args,
                 **kwargs):
        """Evaluate a linear regression estimator using double-loop
        cross-validation over k-folds.

        xy_generator: A function that takes in a dataframe of
                selected records, and that returns a pair of dataframes
                containing the X and y matrices
        kfolds:     A predefined k-fold split of the data
        estimator:  An instance of an classifier estimator class
                that implements `fit` and `predict`; by default,
                datastack.ml.algorithms.linreg.Ridge, which should be
                essentially a superset of sklearn.linear_model.Ridge
        params: A dictionary or list of dictionaries of candidate
                parameter values, as described in
                `sklearn.grid_search.GridSearchCV`

        The user may pass through any legal arguments to
        `sklearn.grid_search.GridSearchCV`.

        We don't instantiate the default estimator until we get to the base
        class, because that is where we can properly apply the idx_covariates
        material.
        """
        if params is None:
            params = _REGRESSION_PARAMS_DEFAULT
        generator = None
        if ordinals is not None:
            def generator(df):
                X, y = xy_generator(df)
                return X, y.apply(lambda v: ordinals[v])
        else:
            generator = xy_generator
        self.ordinals = ordinals
        self.metrics_df = None
        self.VERSION = VERSION
        self.TYPE = TYPE
        self.COVARIATES=COVARIATES
        self.PREDICTED_COVARIATES=PREDICTED_COVARIATES
        self.HOLDOUT = HOLDOUT
        super(
            EvaluationRegression,
            self).__init__(
            generator,
            kfolds,
            estimator=estimator,
            nopreproc=nopreproc,
            params=params,
            fit_params=fit_params,
            idx_covariates=idx_covariates,
            isCvobj=False,
            dtype=float,
            pspec=pspec,
            VERSION=VERSION,
            TYPE=TYPE,
            COVARIATES=COVARIATES,
            PREDICTED_COVARIATES=PREDICTED_COVARIATES,
            HOLDOUT=HOLDOUT,
            *args,
            **kwargs)

    def _modName (self):
        if hasattr(self.estimator, 'underlyingName'):
            return self.estimator.underlyingName()

        return self.estimator.__class__.__name__

    def _add_covariates(self,keep):
        '''
        for face add back covariates
        '''
        print "adding covariates"
        femb_act = np.zeros((1,1))
        femb_pred = np.zeros((1,1))

        if self.VERSION is not None:
            if self.VERSION == "V8NoCov":
                femb_pred = face_model.add_covariates(self.y_pred.ix[keep,:],self.cvfold.ix[keep,:],self.covariates.ix[keep,:],VERSION=self.VERSION,TYPE=self.TYPE,COVARIATES=None,HOLDOUT=self.HOLDOUT)
            else:
                if self.PREDICTED_COVARIATES is None:
                    femb_pred = face_model.add_covariates(self.y_pred.ix[keep,:],self.cvfold.ix[keep,:],self.covariates.ix[keep,:],VERSION=self.VERSION,TYPE=self.TYPE,COVARIATES=self.COVARIATES,HOLDOUT=self.HOLDOUT)
                else:
                    print "pred covs: {}".format(self.PREDICTED_COVARIATES)
                    femb_pred = face_model.add_covariates(self.y_pred.ix[keep,:],self.cvfold.ix[keep,:],self.pred_covariates.ix[keep,:],VERSION=self.VERSION,TYPE=self.TYPE,COVARIATES=self.PREDICTED_COVARIATES,HOLDOUT=self.HOLDOUT)
            # directory where actual images are stored    
            data_path = "/data/notebooks/Faces_v789"
            if self.TYPE=='Position':
                embedding = "ImputedPosition"
            else:
                if self.TYPE =='Color':
                    embedding = 'Albedo'
            # load the real face for each subject
            femb_act = np.zeros(femb_pred.shape)
                    
            pkl_name=self.VERSION+"_"+self.TYPE+'.pkl'
            if self.PREDICTED_COVARIATES is None:
                pkl_name_pred = self.VERSION+"_"+self.TYPE+"_"+str(self.y_pred.shape[1])+"PCs_pred.pkl"
            else:
                pkl_name_pred = self.VERSION+"_"+self.TYPE+"_"+str(self.y_pred.shape[1])+"PCs_pred_predcov.pkl"

            #get order by fold
            idx_order=[]
            orig_order=np.array(list(self.cvfold.ix[keep,:].index))
            orig_fold =self.cvfold.ix[keep,:]
            for fld in range(1,11,1):
                idx_order.extend(list(orig_fold.loc[list(orig_fold['fold']==fld),:].index))
                    
            if not os.path.exists(pkl_name):
                indx=0
                for subject_id in list(self.IDs.ix[idx_order,'subject_id']):
                    fn = os.path.join(data_path, "%s_%s.npy" % (subject_id,embedding))
                            
                    data = np.load(fn)
                    femb_act[indx]= face_model.symmetrizeFace(data, embedding == 'ImputedPosition').flatten(order='F')
                            
                    indx+=1
                femb_act.dump(pkl_name)
            else:
                print "Reading actual values from pkl: {}".format(pkl_name)
                femb_act=np.load(pkl_name)
                    
            femb_act=femb_act.astype(np.float64, copy=False)

            if not os.path.exists(pkl_name_pred):
                print "Saving predicted into pkl: {}".format(pkl_name_pred)

                femb_pred.dump(pkl_name_pred)
            else:
                for i in range(1,101,1):
                    if self.PREDICTED_COVARIATES is None:
                        pkl_name_pred = self.VERSION+"_"+self.TYPE+"_"+str(self.y_pred.shape[1])+"PCs_pred_"+str(i)+".pkl"
                    else:
                        pkl_name_pred = self.VERSION+"_"+self.TYPE+"_"+str(self.y_pred.shape[1])+"PCs_pred_predcov_"+str(i)+".pkl"

                    if not os.path.exists(pkl_name_pred):
                        break
                print "Saving predicted into pkl: {}".format(pkl_name_pred)
                femb_pred.dump(pkl_name_pred)

            
        
        return femb_act,femb_pred

    def _update_metrics(self):
        if len(self.y.shape)>1:
            if self.y_pred.empty:
                keep = pd.Series(
                        self.y.iloc[:,0]).notnull().values & pd.Series(
                        self.y_pred).notnull().values
            else:
                keep = pd.Series(
                        self.y.iloc[:,0]).notnull().values & pd.Series(
                        self.y_pred.iloc[:,0]).notnull().values
            columns = ['Model', 'Covariates','SELECT10','R2']
            cv_metrics = [self._modName()]
            cv_metrics.append(self.X.columns.tolist())
            mae = []
            mse =[]
            select10 = []

            # for the face, add back covariates switch to pixel matching
            if self.VERSION is None or self.TYPE is None:
                r2 = r2_score(self.y.loc[keep,:], self.y_pred.loc[keep,:])
                tot_select10 = rank_all(self.y.ix[keep,:], self.y_pred.ix[keep,:])
                cv_metrics.append(tot_select10)
                cv_metrics.append(np.mean(r2))
            else:
                femb_act,femb_pred=self._add_covariates(keep)
                
                tot_select10 = rank_all(femb_act,femb_pred) 
                r2 = r2_score(self.y.loc[keep,:], self.y_pred.loc[keep,:])
                r2_all = np.mean(metrics.r2_score(femb_act, femb_pred,multioutput='variance_weighted'))
                
                cv_metrics.append(tot_select10)
                cv_metrics.append(r2_all)

            for idx in range(0,self.y.shape[1]):
                columns = columns+[str(idx)+':R2',str(idx)+':MAE',str(idx)+':MSE'] #,str(idx)+':SELECT10']
                mae.append(metrics.mean_absolute_error(self.y.ix[keep,idx], self.y_pred.ix[keep,idx]))
                mse.append(metrics.mean_squared_error(self.y.ix[keep,idx], self.y_pred.ix[keep,idx]))
                cv_metrics.append(r2[idx])
                cv_metrics.append(mae[idx])
                cv_metrics.append(mse[idx])
            self.metrics_df = DataFrame([cv_metrics], columns=columns)
        else:
            keep = pd.Series(
                        self.y).notnull().values & pd.Series(
                        self.y_pred).notnull().values
            columns = ['Model', 'Covariates', 'R2']
            if self.ordinals is not None:
                columns += ['Error']
            columns += ['MAE', 'MSE']
            cv_metrics = [self._modName()]
            cv_metrics.append(self.X.columns.tolist())
            self.r2 = r2_score(self.y.loc[keep], self.y_pred.loc[keep,'y'])
            cv_metrics.append(self.r2)
            if self.ordinals is not None:
                # XXX this assumes 0-based, increase-by-one ordinals!
                cv_compare = [
                    a[0] == a[1] for a in zip(
                        self.y[keep,:].round(),
                        self.y_pred[keep,:].round())]
                self.error = float(cv_compare.count(False)) / len(cv_compare)
                cv_metrics.append(self.error)

            self.mae = metrics.mean_absolute_error(self.y.loc[keep], self.y_pred.loc[keep,'y'])
            self.mse = metrics.mean_squared_error(self.y.loc[keep], self.y_pred.loc[keep,'y'])

            cv_metrics.append(self.mae)
            cv_metrics.append(self.mse)

            self.metrics_df = DataFrame([cv_metrics], columns=columns)

class EvaluationRegressionBaseline(object):

    def __init__(self, target, kfolds, ordinals=None, ppdict=None):
        y = kfolds.get_data(lambda df: df[target])

        # Skip the standardization step to avoid scaling issues later.
        if not ppdict is None and target in ppdict:
            pp = ppdict[target]
            pp2 = pp.clone()
            pp2.stand = False
            pp2.fitPreproc(y)
            y = pp2.runPreproc(y)

        if len(y.shape)>1:
            #y = y.iloc[:,0]
            rows = []
            columns = ['Model', 'Covariates','SELECT10','R2']
            mean_metrics = []
            mean_metrics += ['Mean']
            mean_metrics.append([])
            median_metrics = []
            select10 = []
            median_metrics += ['Median']
            median_metrics.append([])
            # did not calculate those for now
            mean_metrics.append(0)
            median_metrics.append(0)
            mean_metrics.append(0)
            median_metrics.append(0)
            for idx in range(0,y.shape[1]):
                columns +=[str(idx)+':R2', str(idx)+':MAE',str(idx)+':MSE']
                #
                # MeanClass model: Always guess the mean.
                #
                y_mean = y.ix[:,idx].mean()
                mean_y_pred = pd.Series([y_mean for _ in range(0, y.shape[0])])

                mean_metrics.append(r2_score(y.ix[:,idx], mean_y_pred))
                mean_metrics.append(metrics.mean_absolute_error(y.ix[:,idx], mean_y_pred))
                mean_metrics.append(metrics.mean_squared_error(y.ix[:,idx], mean_y_pred))

                #
                # MedianClass model: Always guess the median.
                #
                y_median = y.ix[:,idx].median()
                median_y_pred = pd.Series([y_median for _ in range(0, y.shape[0])])
                median_metrics.append(r2_score(y.ix[:,idx], median_y_pred))
                median_metrics.append(metrics.mean_absolute_error(y.ix[:,idx], median_y_pred))
                median_metrics.append(metrics.mean_squared_error(y.ix[:,idx], median_y_pred))

            rows.append(mean_metrics)
            rows.append(median_metrics)
            self.metrics_df = DataFrame(rows, columns=columns)
        else:
            if ordinals is not None:
                ordinals_reverse = {}
                for k, v in ordinals.items():
                    ordinals_reverse[v] = k
                y = y.map(lambda v: ordinals[v])
            rows = []
            columns = ['Model', 'Covariates', 'R2']
            if ordinals is not None:
                columns += ['Error']
            columns += ['MAE', 'MSE']

            #
            # MeanClass model: Always guess the mean.
            #
            y = y.convert_objects(convert_numeric=True)
            y_mean = y.mean()
            self.mean_y_pred = pd.Series([y_mean for _ in range(0, len(y))])
            mean_metrics = []
            if ordinals is not None:
                mean_class = ordinals_reverse[np.round(y_mean)]
                mean_metrics += ['MeanClass [\'%s\', ordinal %.2f]' %
                             (mean_class, y_mean)]
            else:
                mean_metrics += ['Mean [%s]' % ('{:.3f}'.format(y_mean))]

            mean_metrics.append([])
            mean_metrics.append(r2_score(y, self.mean_y_pred))
            if ordinals is not None:
                mean_compare = [a == round(y_mean) for a in y.round()]
                mean_metrics.append(
                    float(mean_compare.count(False)) /
                    len(mean_compare))

            mean_metrics.append(metrics.mean_absolute_error(y, self.mean_y_pred))
            mean_metrics.append(metrics.mean_squared_error(y, self.mean_y_pred))
            self.mean_metrics = mean_metrics
            rows.append(mean_metrics)

            #
            # MedianClass model: Always guess the median.
            #
            y_median = y.median()
            self.median_y_pred = pd.Series([y_median for _ in range(0, len(y))])
            median_metrics = []
            if ordinals is not None:
                median_class = ordinals_reverse[np.round(y_mean)]
                median_metrics += ['MedianClass [\'%s\', ordinal %.2f]' %
                               (median_class, y_median)]
            else:
                median_metrics += ['Median [%s]' % ('{:.3f}'.format(y_median))]
            median_metrics.append([])
            self.r2 = r2_score(y, self.median_y_pred)
            median_metrics.append(self.r2)
            if ordinals is not None:
                median_compare = [a == round(y_median) for a in y.round()]
                median_metrics.append(
                    float(
                        median_compare.count(False)) /
                    len(median_compare))

            median_metrics.append(metrics.mean_absolute_error(y, self.median_y_pred))
            median_metrics.append(metrics.mean_squared_error(y, self.median_y_pred))
            self.median_metrics = median_metrics
            rows.append(median_metrics)
            self.metrics_df = DataFrame(rows, columns=columns)
