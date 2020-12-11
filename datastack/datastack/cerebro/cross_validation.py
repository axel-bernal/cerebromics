'''
Classes to implement a double cross-validated grid search for training
scikit-learn-style linear estimators.

Created on Mar 14, 2016

@author: twong
'''

import datastack.settings as settings
import hashlib
import json
import logging
import numpy as np
import pandas as pd
import six
import sklearn.metrics as metrics
import warnings

from abc import ABCMeta, abstractmethod
from collections import namedtuple
from joblib import Parallel, delayed
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.cross_validation import PredefinedSplit, check_cv
from sklearn.grid_search import GridSearchCV, ParameterGrid
from sklearn.utils import safe_indexing

#
# This pickle helper block deals with Python's inability to pickle instance
# methods of classes.
#
# Author: Steven Bethard
# http://bytes.com/topic/python/answers/552476-why-cant-you-pickle-instancemethods

import copy_reg
import types


_logger = logging.getLogger(__name__)


def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    cls_name = ''
    if func_name.startswith('__') and not func_name.endswith('__'):
        cls_name = cls.__name__.lstrip('_')
    if cls_name:
        func_name = '_' + cls_name + func_name
    return _unpickle_method, (func_name, obj, cls)


def _unpickle_method(func_name, obj, cls):
    # Author: Steven Bethard
    # http://bytes.com/topic/python/answers/552476-why-cant-you-pickle-instancemethods
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)

# This call to copy_reg.pickle allows you to pass methods as the first arg to
# mp.Pool methods. If you comment out this line, `pool.map(self.foo, ...)` results in
# PicklingError: Can't pickle <type 'instancemethod'>: attribute lookup
# __builtin__.instancemethod failed

copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)

#
# End this pickle helper block.
#


def _hash(key, n_folds, salt=None):
    """Compute the MD5 hash on a subject key, and use the hash value to
    assign the subject to a fold.

    Returns:
        A fold assignment in the range [0, n_folds).
    """
    return long(hashlib.md5(str(key) + ('' if salt is None else salt)).hexdigest(), 16) % n_folds


def _verify_keys_in_df(df, keys=None):
    """Verify that all of a set of keys is in a dataframe.

    Args:
        df: A dataframe
        keys: A string key or a list-like of string keys
    Returns:
        A list-like of the input keys
    """
    if keys is None:
        keys = []
    if not isinstance(keys, list):
        keys = [keys]
    for k in keys:
        if k not in df.columns:
            raise KeyError('Key %s not in dataframe columns' % (k))
    return keys


def _gather_together(
    df,
    idx,
    together=None,
    keep_together_columns=None,
):
    """Gather all of the related subjects from one or more keep-together
    columns containing the subjects.

    Args:
        df: A dataframe containing subject data.
        idx: The dataframe index key that specifies the row containing
            the subjects. Note that the index key value may not necessarily
            correspond to a human-readable subject name.
        together: A pre-existing list to fill with related subjects; if
        `None`, create a new list.
        keep_together_columns: The columns in the dataframe that specify
            lists of subjects related to the subject referred to by idx_key;
            by default, 'None' (no-op).
    Returns:
        A list of related subjects gathered from row in the dataframe referred
            to by the idx_key.
    """
    if together is None:
        together = []
    together_raw = df.loc[idx, keep_together_columns].fillna('[]')
    for col in keep_together_columns:
        try:
            together += json.loads(together_raw[col])
        except TypeError:
            raise TypeError('Expected keep-together column to contain a JSON list: Record %s' % str(idx))
    return sorted(together)


def _verify_together(
    df,
    key,
    together=None,
    index_column=None,
    keep_together_columns=None,
):
    """For a key in a dataframe that is included in a
    keep-together relationship, check that every related
    key specifies all of the other related keys in a set
    of related keys; for example, if `A`, `B`, and `C`
    are related, then `A` must list `B` and `C`, `B` must
    list `A` and `C`, and `C` must list `A` and `B`.

    Args:
        df: A dataframe containing subject data.
        key: A subject key from the index_column, or from the dataframe index
            itself
        together: A list of subjects that are related to the subject specified
            by `subject`; by default, `None` (no-op)
        index_column: The column in the dataframe that specifies keys
            to hash subjects, or the dataframe index itself if `None`.
        keep_together_columns: The columns in the dataframe that specify
            lists of subjects related to the subject referred to by idx_key;
            by default, 'None' (no-op).
    Returns:
        `True` if the complete keep-together list key specifies all of the
        other related keys in a set of related keys.
    """
    if together is None or len(together) <= 0:
        return True
    # Create a set containing myself and my relatives.
    related_keys = [key] + together
    # Now, for each of my relatives, make sure that the
    # set of themselves and their relatives is identical
    # to mine.
    index = df[index_column] if index_column is not None else df.index
    for i in df[index.str.match('|'.join(together))].index:
        r = df.at[i, index_column] if index_column is not None else i
        related_related_keys = [r]
        related_related_keys = _gather_together(
            df,
            i,
            related_related_keys,
            keep_together_columns
        )
        if (set(related_keys) != set(related_related_keys)):
            return False
    return True


class HashedKfolds(PredefinedSplit):
    """Create k-folds by hashing a key column in a dataframe into folds
    suitable for use in `sklearn`-like cross validation loops. Iterating
    over the folds returns k training/test splits as a tuple of lists;
    the first list contains training set integer indices, and the second
    list contains test set integer indices. The indices are offsets into
    a dataframe in the k-fold object that has holdout subjects removes,
    which is containined in the `df` member variable. Do NOT use the
    indices to index into the original dataframe passed to the object
    initializer.

    Variables:
        df: A dataframe containing training subject data.
        df_holdout: A dataframe containing holdout subject data.
        index_column: The column in the dataframe that specifies keys
            to hash subjects, or the datafram index itself if `None`.
        n_training: The number of training folds.
        n_holdout: The number of holdout folds.
        keep_in_training: The columns in the dataframe that constrain
            subjects to be in training folds.
        keep_in_holdout: The columns in the dataframe that constrain
            subjects to be in holdout folds.
        keep_together: The columns in the dataframe that constrain related
            subjects to be in the same folds
        ids: An indexed series containing fold IDs for all subjects.
    """

    def __init__(
        self,
        df,
        n_training=10,
        n_holdout=2,
        index_column=None,
        keep_in_training_columns=None,
        keep_in_holdout_columns=None,
        keep_together_columns=None,
        inplace=False,
        drop_metadata_columns=False,
        sort_columns=True,
        salt=None,
    ):
        """Generate a new set of folds on subject data in a dataframe.
        The generator will use the values in a given key column as keys
        to hash the subjects into training folds and holdout folds. In
        general, one does not use subjects in the holdout to train or
        test exploratory machine learning models.

        :param df: A dataframe containing subject data.
        :param n_training: The number of training folds; by default, 10.
        :param n_holdout: The number of holdout folds; by default, 2.
        :param index_column: The column in the dataframe that specifies keys
            to hash subjects.
            If `None`, use the labels in the dataframe index.
        :param keep_in_training_columns: A list of columns with
            boolean values; by default, `None` (no constraints). If any
            of the values in the columns is true for a row, then include
            the subject in that row in the training set (and never put it
            in the holdout set).
        :param keep_in_holdout_columns: A list of columns with
            boolean values; by default, `None` (no constraints). If any
            of the values in the columns is true for a row, then include
            the subject in that row in the holdout set (and never put it
            in the training set).
        :param keep_together_columns: The column in the dataframe that specifies
            lists of subjects related to the current subject; by
            default, `None` (i.e., do not consider relations). If
            used, every related subject must specify all of the other
            related subjects in a set of related subjects; for
            example, if `A`, `B`, and `C` are related, then `A` must
            list `B` and `C`, `B` must list `A` and `C`, and `C` must
            list `A` and `B`.
        :param inplace: If `True`, performing column drop operations
            affects the original dataframe; by default, `False`.

            **Note**: The `sort_columns` option, if `True`, always sorts in
            place.
        :param drop_metadata_columns: If `True`, drop the metadata columns
            from the dataframe stored with the k-folds after creating the
            folds; by default, `False`.
        :param sort_columns: A flag to force sorting of columns in the dataframe;
            by default, `True`. If you set this to `False`, you had better
            know what you are doing, otherwise you will likely end up down
            the line with non-reusable regression models if downstream
            users hand off predictor columns in a different order. So
            basically, set it to `True`.
        :param salt: A text salt to seed the function used hash subjects;
            by default, `None`
        """
        if df is None:
            raise ValueError('Missing dataframe parameter')
        self._keep_together = _verify_keys_in_df(df, keep_together_columns)
        self._keep_in_training = _verify_keys_in_df(df, keep_in_training_columns)
        self._keep_in_holdout = _verify_keys_in_df(df, keep_in_holdout_columns)
        if inplace:
            self._df = df
        else:
            self._df = df.copy()
        if n_training < 1:
            raise ValueError('Invalid training set size: Expected at least one fold, got %d' % (self.n_training))
        self.n_training = n_training
        if n_holdout < 0:
            raise ValueError('Invalid training set size: Expected at least zero folds, got %d' % (self.n_holdout))
        self.n_holdout = n_holdout
        self.in_training = lambda v: (v >= 0) & (v < self.n_training)
        self.in_holdout = lambda v: (v < 0) | (v >= self.n_training)
        self.index_column = index_column
        # Create a list of the hash keys that we will run through the
        # hash-and-modulo function to assign subjects to k-folds. The basic
        # hash key is the value in the key column, but this may change later
        # based on the keep- constraints.
        #
        # Note that the hash keys may change based on keep-together, but the
        # subject keys will not.
        subject_keys = None
        if self.index_column is not None:
            try:
                subject_keys = self._df[self.index_column]
            except KeyError:
                raise KeyError('Missing key column \'%s\' in dataframe' % (self.index_column))
        else:
            subject_keys = pd.Series(self._df.index, index=self._df.index)
        hash_keys = subject_keys.copy()
        hash_keys.name = 'ids'
        # Get a boolean pandas series of subjects to include in the training
        # set. The pandas index of the series, if set, will be the same
        training_mask = None
        if len(self.keep_in_training) > 0:
            training_mask = self._df[self.keep_in_training].any(axis=1)
        # Get a boolean pandas series of subjects to include in the holdout
        # set. The pandas index of the series, if set, will be the same.
        holdout_mask = None
        if len(self.keep_in_holdout) > 0:
            holdout_mask = self._df[self.keep_in_holdout].any(axis=1)
        # Now work with subjects that we need to keep together
        if len(self.keep_together) > 0:
            # To extract list of kept-together subjects, we only need to look
            # at rows with at least one related subject entry.
            tidx = self._df[self.keep_together].any(axis=1)
            together_raw = self._df[self.keep_together][tidx].fillna('[]')
            # Careful: i is the pandas dataframe index, which may or may not be
            # semantically meaningful
            for i in together_raw.index:
                together = _gather_together(together_raw, i, keep_together_columns=self.keep_together)
                if not _verify_together(
                        self._df,
                        subject_keys[i],
                        together=together,
                        index_column=self.index_column,
                        keep_together_columns=self.keep_together):
                    raise ValueError(
                        'Keep-together columns inconsistent: Record %s index %s' %
                        (str(i), subject_keys[i]))
                # Yuck. Once we have the together list, we need to flag
                # subjects that we must keep together, to respect
                # keep-in-training and keep-in-holdout. Note that if there's a
                # conflict where the subject is kept in training and while a
                # kept-together subject is kept in holdout (or vice versa)
                # we'll throw an exception when we check for kept-in overlaps.
                if training_mask is not None and training_mask[i]:
                    training_mask[subject_keys.isin(together)] = True
                if holdout_mask is not None and holdout_mask[i]:
                    holdout_mask[subject_keys.isin(together)] = True
                hash_keys[i] = sorted([hash_keys[i]] + together)[0]
        # Make sure that there isn't any overlap between the keep-in-training
        # and keep-in-holdout constraints.
        if training_mask is not None and holdout_mask is not None:
            if (training_mask & holdout_mask).any():
                overlap = set(subject_keys[training_mask & holdout_mask].tolist())
                raise ValueError('Keep-in-training and keep-in-holdout constraints overlap: %s' % (' '.join(overlap)))
        self._n_folds = self.n_training + self.n_holdout
        self._hashes = hash_keys.apply(lambda hash_key: _hash(hash_key, self._n_folds))
        # Apply keep-in-training and -holdout masks
        if training_mask is not None:
            self._hashes[training_mask] = hash_keys[training_mask].apply(lambda h: _hash(h, self.n_training))
        # Get a boolean array of subjects to include in the holdout set.
        if (holdout_mask is not None):
            if self.n_holdout > 0:
                self._hashes[holdout_mask] = hash_keys[holdout_mask].apply(lambda h: _hash(h, self.n_holdout) + self.n_training)
            else:
                self._hashes[holdout_mask] = -1
        # Extreme paranoia - make sure that the hash function doesn't
        # scramble the indexes.
        if (self._df.index == self._hashes.index).all() != True:
            raise Exception('Dataframe and k-fold hashing indexes out of sync')
        # Now hand off k-fold assignments (minus the holdout set) to the parent
        # class
        assignments_minus_holdouts = self._hashes[self.in_training(self._hashes)]
        # Save anything that might alter the original dataframe until the very
        # end.
        #
        # After multiple bugs caused by randomly ordered columns, we now
        # forcibly sort the columns
        if sort_columns:
            self._df.sort_index(axis=1, inplace=True)
        else:
            _logger.warning('Leaving dataframe columns unsorted; you have been warned!')
        if drop_metadata_columns:
            for c in self._keep_together + self._keep_in_training + self._keep_in_holdout:
                del self._df[c]
        super(HashedKfolds, self).__init__(assignments_minus_holdouts)

    def __len__(self):
        """Returns the total number of folds, excluding the holdout folds.

        Returns:
            The total number of training folds
        """
        return (self.n_training)

    def __repr__(self):
        """Returns a human-readable representation of the k-fold
        collection.
        """
        return '%s.%s(n_subjects=%i,n_training=%i,n_holdout=%i)' % (
            self.__class__.__module__,
            self.__class__.__name__,
            self.ids.shape[0],
            self.n_training,
            self.n_holdout,
        )

    @property
    def keep_in_training(self):
        return self._keep_in_training

    @property
    def keep_in_holdout(self):
        return self._keep_in_holdout

    @property
    def keep_together(self):
        return self._keep_together

    @property
    def df(self):
        """A dataframe containing only training data.
        """
        return self._df.loc[self.in_training(self._hashes)]

    @property
    def df_holdout(self):
        """A dataframe containing only holdout data.
        """
        return self._df.loc[self.in_holdout(self._hashes)]

    @property
    def ids(self):
        """Returns a list-like of the k-fold assignment for each subject.
        """
        return self._hashes

    def get_fold(self, fold_id):
        """Returns the set of subjects in a given fold.

        Args:
            fold_id: The integer ID of the desired fold.

        Returns:
            The set of subjects in the desired fold as pandas dataframe index
            keys into the original dataframe.
        """
        return self._hashes[self._hashes == fold_id].index


class PrelabeledKfolds(PredefinedSplit):

    def __init__(self, df, label_column, label_filter=None, drop_label_column=False, sort_columns=True):
        """Generate a new set of folds on subject data in a dataframe. Each
        fold contains the set of subjects with same label as identified by
        the labels in the named label column. One potential use of this
        type of k-fold is when defining folds by subject study.

        Note: One could imagine wrapping
        :class:`sklearn.cross_validation.LabelKFold` instead, except that
        the ``sklearn`` version does not deal with unlabeled subjects.

        :param df: A dataframe containing subject data.
        :param label_column: The column in the dataframe that specifies
            the labels.
        :param label_filter: If not `None`, a list of labels to include,
            while excluding all other labels; by default, `None` (include
            all labels).
        :param drop_label_column: If `True`, drop the label column from
            the dataframe stored with the k-folds after creating the folds;
            by default, `False`.
        :param sort_columns: A flag to force sorting of columns in the dataframe;
            by default, `True`. If you set this to `False`, you had better
            know what you are doing, otherwise you will likely end up down
            the line with non-reusable regression models if downstream
            users hand off predictor columns in a different order. So
            basically, set it to `True`.
        """
        if df is None:
            raise ValueError('Missing dataframe parameter')
        self._df = df
        if label_column is None:
            raise ValueError('Missing label key parameter')
        self._label_column = label_column
        if label_filter is None:
            self._label_filter = []
        else:
            self._label_filter = list(set(label_filter))
        labels_included = set(self._df[self._label_column].dropna())
        if len(self._label_filter) > 0:
            labels_included = labels_included & set(self._label_filter)
            self._df = self._df.dropna(subset=[self._label_column])
            self._df = self._df[self._df[self._label_column].str.match('^' + '$|^'.join(labels_included) + '$')]
        else:
            self._df = df
        self._len = len(labels_included)
        # -1 assigns unlabeled subjects to the training split. See
        # sklearn.cross_validation.PredefinedSplit
        self._ids = pd.Series(data=-1, index=self._df.index, dtype=int, name='ids')
        self._ids_to_labels = {}
        self._labels_to_ids = {}
        _id = 0
        for s in sorted(list(labels_included)):
            # Remember that _ids and _df have identical indexes
            self._ids[self._df[self._label_column] == s] = _id
            self._ids_to_labels[_id] = s
            self._labels_to_ids[s] = _id
            _id += 1
        # Save anything that might alter the original dataframe until the very
        # end.
        if sort_columns:
            self._df.sort_index(axis=1, inplace=True)
        else:
            _logger.warning('Leaving dataframe columns unsorted; you have been warned!')
        # Note - after this point, do not try to reference the label
        # column in the dataframe.
        if drop_label_column:
            for c in [self._label_column]:
                del self._df[c]
        super(PrelabeledKfolds, self).__init__(self._ids)

    def __len__(self):
        """Returns the total number of folds, which is equivalent to the
        total number of selected labels

        :return: the total number of folds
        """
        return self._len

    def __repr__(self):
        """Returns a human-readable representation of the k-fold
        collection.
        """
        return '{}.{}(n_subjects={},labels={})'.format(self.__class__.__module__, self.__class__.__name__, self._df.shape[0], self.labels)

    @property
    def df(self):
        """A dataframe containing the data for the selected labels
        """
        return self._df

    @property
    def ids(self):
        """Returns a list-like of the k-fold assignment for each subject.
        """
        return self._ids

    @property
    def labels(self):
        """Returns a list of the included labels
        """
        return sorted(self._labels_to_ids.keys())

    def get_fold(self, fold_id):
        """Returns the set of subjects in a given fold.

        :param fold_id: The integer ID of the desired fold, or a label
            corresponding to a fold.

        :return: the set of subjects in the desired fold as pandas
            dataframe index keys into the original dataframe.
        """
        if isinstance(fold_id, basestring):
            fold_id = self._labels_to_ids[fold_id]
        if fold_id >= self._len:
            raise IndexError
        return self._ids[self._ids == fold_id].index

    def get_fold_label(self, fold_id):
        """Return the label corresponding to a fold ID

        :param fold_id: The integer ID of the desired fold.

        :return: the label corresponding to a fold ID
        """
        return self._ids_to_labels[fold_id]


class GridScore(namedtuple('GridScore', ['estimator', 'params', 'score', 'split_id', 'test_index', 'y_test', 'y_test_pred', 'y_test_pred_proba'])):
    """A named tuple containing the scores from a cross-validated grid
    search.

    :var estimator: The estimator trained with the best hyperparameter
        values found when training on the training data and scoring on the
        test data.
    :var params: The best hyperparameter values.
    :var score: The best predicted score found by scoring the estimator
        with the test data.
    :var split_id: The split ID. For most k-fold generators, this ID will
        correspond to the fold that was used as the test fold.
    :var test_index: The integer index values used to select the test
        data from the cross-validation generator. The index values are row
        offsets into the X and y data structures used to train the
        estimator.
    :var y_test: The observed target values from the test data.
    :var y_test_pred: The predicted target values computed using the
        estimator and the test data.
    :var y_test_pred_proba: The predicted target value probabilities
        (for estimators that are classifiers).
    """

    # Setting __slots__ prevents extra memory allocation for this subclass'
    # attribute dictionary, which we don't need anyway because we're not
    # adding new attributes.
    __slots__ = ()

    def __repr__(self, *args, **kwargs):
        return '(split={0}, params={1}, score={2:f}, y_pred.shape={3})'.format(self.split_id, self.params, self.score, self.y_test_pred.shape)


def _safe_indexing(X, y, train, test):
    X_train, X_test = safe_indexing(X, train), safe_indexing(X, test)
    if X_train.isnull().any().any() or X_test.isnull().any().any():
        warnings.warn('Test split contains NaN values')
    if y is not None:
        y_train, y_test = safe_indexing(y, train), safe_indexing(y, test)
        if y_train.isnull().any().any() or y_test.isnull().any().any():
            warnings.warn('Test split contains NaN values')
    else:
        y_train, y_test = None, None
    return X_train, X_test, y_train, y_test, train, test


class CrossValidationInnerBase(six.with_metaclass(ABCMeta)):
    """A base class that encapsulates the inner loop of a cross-validation
    run.
    """

    @abstractmethod
    def run(self, grid_search, cv, X, y, **kwargs):
        """Run the inner loop of a cross-validation algorithm.

        :param grid_search: The enclosing cross-validation grid search
            object; expected to be a subclass of
            :class:`sklearn.grid_search.GridSearchCV`.
        :param cv: A cross-validation generator, as defined for
            :class:`sklearn.grid_search.GridSearchCV`. cv **must** have a
            well-defined length accessible through the ``len()`` function.
        :param X: An array of the training covariate values for all members
            of all folds in a set of k-folds; must correspond to the
            train/test split specified by the train and test indexes.
        :param y: An array or vector of training observation values for all
            members of all folds in a set of k-folds; must correspond
            to the train/test split specified by the train and test
            indexes.
        """
        if grid_search.n_jobs > 1 and not settings.DATASTACK_MULTIPROCESSING_OK:
            warnings.warn('On OS X, a POSIX violation in multiprocessing results in the interpreter hanging inside sklearn', RuntimeWarning)
            warnings.warn('"Bad interaction of multiprocessing and third-party libraries"', RuntimeWarning)
            warnings.warn('https://pythonhosted.org/joblib/parallel.html', RuntimeWarning)


class CrossValidationInnerBasic(CrossValidationInnerBase):
    """A class that encapsulates the inner loop of a basic cross-validation
    run.
    """

    def run(self, grid_search, cv, X, y, with_legacy_grid_score=False, **kwargs):
        """Run the inner loop of a cross-validation algorithm.

        :param grid_search: The enclosing cross-validation grid search
            object; expected to be a subclass of
            :class:`sklearn.grid_search.GridSearchCV`.
        :param cv: A cross-validation generator, as defined for
            :class:`sklearn.grid_search.GridSearchCV`
        :param X: An array of the training covariate values for all members
            of all folds in a set of k-folds; must correspond to the
            train/test split specified by the train and test indexes.
        :param y: An array or vector of training observation values for all
            members of all folds in a set of k-folds; must correspond
            to the train/test split specified by the train and test
            indexes.
        :param with_legacy_grid_score: Return the same score values that
            :class:`sklearn.grid_search.GridSearchCV` would return in
            :data:`sklearn.grid_search.GridSearchCV.best_score_`
            field. Our backward-compatibility inner loop converges on the
            same best set of hyperparameter values as
            :class:`sklearn.grid_search.GridSearchCV`, but normally returns
            a different score because of the way it retrains estimators
            with the best set for each train/test split.
        """
        super(CrossValidationInnerBasic, self).run(grid_search, cv, X, y)
        raw_scores = Parallel(n_jobs=grid_search.n_jobs, verbose=grid_search.verbose, pre_dispatch=grid_search.pre_dispatch)(
            delayed(self._fit_and_score_basic_loop)(
                grid_search.estimator,
                grid_search.param_grid,
                grid_search.scoring,
                X,
                y,
                split_id,
                train,
                test,
                fit_params=grid_search.fit_params,
                iid=grid_search.iid,
                verbose=grid_search.verbose,
            )
            for split_id, (train, test) in zip(range(len(cv)), cv)
        )
        # The Cerebro outer CV loop is train/test split oriented, whereas
        # the sklearn grd search loop is parameter-set oriented. So, to get
        # a sane set of scores, we have to compute the average score across
        # all test splits for each parameter set.
        #
        # First, flatten out the raw score list - it will be a list of
        # lists.
        flattened_raw_scores = [grid_score for grid_score_list in raw_scores for grid_score in grid_score_list]
        param_scores = []
        # Compute a composite score for each possible parameter set
        for params in ParameterGrid(grid_search.param_grid):
            score = 0.0
            subjects = 0
            test_split_scores = [g for g in flattened_raw_scores if g.params == params]
            # Now, for each test split, compute the average score. If the
            # subject data is identically distributed across the training,
            # splits, compute an average weighted by subjects per training
            # split.
            for test_fold_score in test_split_scores:
                if grid_search.iid:
                    score += (test_fold_score.score * len(test_fold_score.test_index))
                    subjects += len(test_fold_score.test_index)
                else:
                    score += test_fold_score.score
            if grid_search.iid:
                score /= float(subjects)
            else:
                score /= float(len(test_split_scores))
            param_scores.append((params, score))
        # Sort and select the hyperparameter set yielding the highest
        # score.
        _best_params = sorted(param_scores, key=lambda param_score: param_score[1], reverse=True)[0]
        _logger.debug('Best parameters: {}'.format(_best_params[0]))
        # To maintain semantic compatibility with the grid scores produced
        # by the five-fold inner loop, retrain an estimator using the
        # best hyperparameter set for each train/test split, and record the
        # score for each test split.
        _grid_scores = []
        for split_id, (train, test) in zip(range(len(cv)), cv):
            X_train, X_test, y_train, y_test, train, test = _safe_indexing(X, y, train, test)
            _estimator, _score, y_test_pred, y_test_pred_proba = self._fit_and_score_basic(
                grid_search.estimator,
                _best_params[0],
                grid_search.scoring,
                X_train,
                X_test,
                y_train,
                y_test,
                grid_search.fit_params,
                grid_search.verbose
            )
            if with_legacy_grid_score:
                _score = _best_params[1]
            _grid_scores.append(GridScore(_estimator, _best_params[0], _score, split_id, test, y_test, y_test_pred, y_test_pred_proba))

        return _grid_scores

    def _fit_and_score_basic(self, estimator, params, scoring, X_train, X_test, y_train, y_test, fit_params, verbose=False):
        """Fit an estimator with a training split, and score it with a test
        split.
        """
        _estimator = clone(estimator)
        _estimator.set_params(**params)
        _estimator.fit(X_train, y=y_train, **fit_params)
        if scoring is not None:
            _score = scoring(_estimator, X_test, y_test)
        else:
            _score = _estimator.score(X_test, y_test)
        if verbose > 2:
            print '_fit_and_score_basic: params {} score {}'.format(params, _score)
        y_test_pred = _estimator.predict(X_test)
        y_test_pred_proba = None
        if is_classifier(_estimator):
            y_test_pred_proba = _estimator.predict_proba(X_test)
        return _estimator, _score, y_test_pred, y_test_pred_proba

    def _fit_and_score_basic_loop(
        self,
        estimator,
        param_grid,
        scoring,
        X,
        y,
        split_id,
        train,
        test,
        fit_params,
        iid,
        verbose,
    ):
        """A helper method to run the inner cross-validation loop. Note that
        the train and test integer indexes are row offsets into the X and y
        data structures, and do not correspond to any pandas indexes.
        """
        X_train, X_test, y_train, y_test, train, test = _safe_indexing(X, y, train, test)
        _param_scores = []
        # For every possible parameter combination, fit using the training
        # split, then score using the test split.
        for _params in ParameterGrid(param_grid):
            _estimator, _score, y_test_pred, y_test_pred_proba = self._fit_and_score_basic(
                estimator,
                _params,
                scoring,
                X_train,
                X_test,
                y_train,
                y_test,
                fit_params,
                verbose
            )
            _param_scores.append(GridScore(_estimator, _params, _score, split_id, test, y_test, y_test_pred, y_test_pred_proba))
        # Return a list of scores, one for each possible parameter combo.
        return _param_scores


class CrossValidationInnerGridSearch(CrossValidationInnerBase):
    """A class that encapsulates the inner loop of a double-cross-
    validation run.
    """

    def _inner_cv_generator_default(self, X_train):
        return 5

    def __init__(self, inner_cv_generator=None):
        """Create the inner loop of a double-cross-validation run.

        :param cv_generator: A function that takes a dataframe of
            training covariate values, and returns a cross-validation fold
            specification, as defined for
            :class:`sklearn.grid_search.GridSearchCV`, by default, a
            function that returns the integer 5 (use five folds).

        **WARNING** This cannot be a lambda function - pickle (and our
        pickle helpers) cannot pickle lambda functions.
        """
        if inner_cv_generator is None:
            inner_cv_generator = self._inner_cv_generator_default
        self.inner_cv_generator = inner_cv_generator
        super(CrossValidationInnerGridSearch, self).__init__()

    def run(self, grid_search, cv, X, y, **kwargs):
        super(CrossValidationInnerGridSearch, self).run(grid_search, cv, X, y, **kwargs)
        grid_scores = Parallel(n_jobs=grid_search.n_jobs, verbose=grid_search.verbose, pre_dispatch=grid_search.pre_dispatch)(
            delayed(self._fit_and_score_grid_search)(
                grid_search.estimator,
                grid_search.param_grid,
                grid_search.scoring,
                self.inner_cv_generator,
                X,
                y,
                split_id,
                train,
                test,
                fit_params=grid_search.fit_params,
                iid=grid_search.iid,
                verbose=grid_search.verbose,
            )
            for split_id, (train, test) in zip(range(len(cv)), cv)
        )
        return grid_scores

    def _fit_and_score_grid_search(
        self,
        estimator,
        param_grid,
        scoring,
        cv_generator,
        X,
        y,
        split_id,
        train,
        test,
        fit_params,
        iid,
        verbose,
    ):
        """A helper method to run the inner cross-validation loop. Note that
        the train and test integer indexes are row offsets into the X and y
        data structures, and do not correspond to any pandas indexes.
        """
        X_train, X_test, y_train, y_test, train, test = _safe_indexing(X, y, train, test)
        # The inner grid search uses the training data to complete a cross-
        # validated grid search.
        grid = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring=scoring,
            fit_params=fit_params,
            n_jobs=1,
            iid=iid,
            refit=True,
            cv=cv_generator(X_train),
            verbose=verbose,
        )
        grid.fit(X_train, y=y_train, **fit_params)

        best_score = grid.score(X_test, y_test)
        y_test_pred = grid.best_estimator_.predict(X_test)
        y_test_pred_proba = None
        if is_classifier(grid.best_estimator_):
            y_test_pred_proba = grid.best_estimator_.predict_proba(X_test)

        return GridScore(grid.best_estimator_, grid.best_params_, best_score, split_id, test, y_test, y_test_pred, y_test_pred_proba)


class CrossValidation(GridSearchCV):

    def __init__(
        self,
        estimator,
        param_grid,
        scoring=None,
        fit_params=None,
        n_jobs=1,
        iid=True,
        cv=None,
        refit=True,
        verbose=0,
        pre_dispatch='2*n_jobs',
        inner_loop=CrossValidationInnerGridSearch(),
        **kwargs
    ):
        """Create a cross-validation loop that saves predicted target
        values computed from the test data in each train/test split. By
        default, for a k-fold cross-validated search,
        :class:`CrossValidation` will
        fit the estimator k times: each fit will train the estimator using
        (k-1) folds as training data, and predict target values from the
        trained best estimator using the remaining fold. In this way, we
        can get and save predicted target values for all of the input
        training data, unlike the scikit-learn grid search. We inherit any
        class methods not overriden here from
        :class:`sklearn.grid_search.GridSearchCV`.

        :class:`CrossValidation` by default uses an inner loop that itself
        uses a five-fold grid search to find the best hyperparameters for
        the estimator when fitted with each train/test split. Users that
        need the legacy scikit-learn grid search inner loop, but that also
        need to save predicted target values, can substitute the default
        inner loop with a re-implemented basic loop; see the `inner_loop`
        parameter.

        :param estimator: An estimator that implements the scikit-learn
            estimator interface.
        :param param_grid: Dictionary with parameters names (string) as
            keys and lists of parameter settings to try as values, or a
            list of such dictionaries, in which case the grids spanned by
            each dictionary in the list are explored. This enables
            searching over any sequence of parameter settings.
        :param scoring: A string (see model evaluation documentation) or a
            scorer callable object / function with signature
            `scorer(estimator, X, y)`. If `None`, use the score method of
            the estimator.

            **WARNING**: The scoring function must be such that greater
            scores are better, otherwise :func:`best_score_`,
            :func:`best_estimator`, and :func:`best_params_` will refer to
            the *worst* fitting estimator.
        :param fit_params: Parameters to pass to the fit method in the
            estimator.
        :param n_jobs: The number of jobs to run in parallel; by default,
            one.
        :param iid: If `True`, we assume that the data is identically
            distributed across the folds, and we minimize the total loss
            per sample, and not the mean loss across the folds; by default,
            `True`.
        :param cv: A cross-validation generator, as defined for
            :class:`sklearn.grid_search.GridSearchCV`.
        :param refit: If `True`, refit the best estimator with the entire
            dataset; by default, `True`.
        :param verbose: Controls the verbosity: the higher, the more
            messages. This argument is passed through to
            :class:`sklearn.grid_search.GridSearchCV`.
        :param inner_loop: Selects the inner loop implementation to use;
            by default, an instance of
            :class:`CrossValidationInnerGridSearch`. To get a
            legacy scikit-learn grid search inner loop, use an instance of
            :class:`CrossValidationInnerBasic` instead.
        :type inner_loop: concrete subclass of
            :class:`CrossValidationInnerBase`
        """
        super(CrossValidation, self).__init__(
            estimator,
            param_grid,
            scoring=scoring,
            fit_params=fit_params,
            n_jobs=n_jobs,
            iid=iid,
            refit=refit,
            cv=cv,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            **kwargs
        )
        self._inner_loop = inner_loop

    @property
    def grid_scores_(self):
        """A list of grid score tuples, with one tuple for each
        train/test split specified by the input cross-validation generator
        `cv`. For the tuple field definitions, see
        :class:`datastack.cerebro.cross_validation.GridScore`.
        """
        return self._grid_scores

    @property
    def best_estimator_(self):
        """An estimator that uses the set of parameters that gave
        the best score (or smallest loss, if specified) across all of the
        inner grid searches, and that is fit on all of the data from the
        original train/test splits. This is the underlying estimator used
        in calls to `predict`. Only available if the user set the `refit`
        instance parameter to `True`.
        """
        return self._best_estimator

    @property
    def best_params_(self):
        """The set of parameters that gave the best score (or smallest
        loss, if specified) across all of the inner grid searches.
        """
        return sorted(self.grid_scores_, key=lambda x: x.score, reverse=True)[0].params

    @property
    def best_score_(self):
        """The best score (or smallest loss, if specified)
        across all of the inner grid searches
        """
        return sorted(self.grid_scores_, key=lambda x: x.score, reverse=True)[0].score

    @property
    def y(self):
        """The observed target values
        """
        return self._y

    @property
    def y_pred(self):
        """The predicted target values from the trained best estimator for
        each train/test split, using the test split (i.e., the held-out
        fold on the inner cross-validated grid search) as the input
        predictor/covariate values to the estimator.
        """
        if self._y_pred is None:
            _value_table = [(g.test_index, g.y_test_pred) for g in self.grid_scores_]
            _value_template = _value_table[0][1]
            _value_types = list(set([type(v) for _, v in _value_table]))
            if len(_value_types) != 1:
                raise Exception('Predicted y values saved in heterogeneous data containers')
            _value_length = sum([len(index) for index, _ in _value_table])
            # Constructing the shape is tricky when dealing with
            # dataframes vs. series vs. arrays...
            _value_shape = list(_value_template.shape)
            _value_shape[0] = _value_length
            try:
                _value_dtype = _value_template.dtype
            except AttributeError:
                # Maybe it's a dataframe? Get the most accommodating dtype
                _value_dtype = _value_template.values.dtype
            _value_pred = np.empty(tuple(_value_shape), dtype=_value_dtype)
            _value_pred_index = None
            # Hack time: If the estimator stored its predictions in an
            # indexed container (i.e., a dataframe or series), then we
            # construct an indexed container to hold all of the
            # predictions.
            if hasattr(_value_template, 'index'):
                _value_pred_index = np.empty((_value_length, ), dtype=_value_template.index.dtype)
            for index, v in _value_table:
                _value_pred[index] = v
                if hasattr(_value_template, 'index'):
                    _value_pred_index[index] = v.index
            if _value_types[0] == pd.DataFrame:
                self._y_pred = pd.DataFrame(data=_value_pred, index=_value_pred_index, columns=_value_template.columns)
            elif _value_types[0] == pd.Series:
                self._y_pred = pd.Series(data=_value_pred, index=self.y.index, name=_value_template.name)
            else:
                self._y_pred = _value_pred
        return self._y_pred

    @property
    def y_pred_proba(self):
        """The predicted target value probabilities from the trained best
        estimator for each train/test split, using the test split (i.e.,
        the held-out fold on the inner cross-validated grid search) as the
        input predictor/covariate values to the estimator. Only available
        for estimators that are classifiers.
        """
        if self._y_pred_proba is None:
            _value_table = [(g.test_index, g.y_test_pred_proba) for g in self.grid_scores_]
            _value_template = _value_table[0][1]
            _value_types = list(set([type(v) for _, v in _value_table]))
            if len(_value_types) != 1:
                raise Exception('Predicted y values saved in heterogeneous data containers')
            _value_length = sum([len(index) for index, _ in _value_table])
            _value_width = set([v.shape[1] for _, v in _value_table])
            if len(_value_width) != 1:
                warnings.warn('Subjects in some k-folds did not cover all classes', RuntimeWarning)
                return None
            _value_pred = np.empty((_value_length, _value_width.pop()))
            _value_pred_index = None
            # Hack time: If the estimator stored its predictions in an
            # indexed container (i.e., a dataframe or series), then we
            # construct an indexed container to hold all of the
            # predictions.
            if hasattr(_value_template, 'index'):
                _value_pred_index = np.empty((_value_length, ), dtype=_value_template.index.dtype)
            for index, v in _value_table:
                _value_pred[index] = v
                if hasattr(_value_template, 'index'):
                    _value_pred_index[index] = v.index
            if _value_types[0] == pd.DataFrame:
                self._y_pred_proba = pd.DataFrame(data=_value_pred, index=_value_pred_index)
            else:
                self._y_pred_proba = _value_pred
        return self._y_pred_proba

    def fit(self, X, y, **inner_loop_kwargs):
        """Runs an inner cross-validated grid search on all input train/
        test splits over the estimator parameter space to train the input
        linear estimator. Note that if the input cross-validation generator
        is a predefined split (i.e., a scikit-learn `PredefinedSplit`
        instance), the input X and y data must correspond to the integer
        indexes provided in the train/test splits.

        Note that this implementation borrows heavily from the fit and _fit
        methods in scikit-learn `GridSearchCV` and `BaseSearchCV` classes.

        :param X: An array-like of the training predictor/covariate values.
        :param y: An array-like or vector of the training target values.

        :return: a reference to this double cross-validated grid search
            instance.
        """
        if y is not None:
            if y.shape[0] != X.shape[0]:
                raise ValueError('Got mismatched count of observations to covariates: Expected {:d}, got {:d}'.format(X.shape[0], y.shape[0]))
        cv = check_cv(self.cv, X, y, classifier=is_classifier(self.estimator))

        self._grid_scores = self._inner_loop.run(self, cv, X, y, **inner_loop_kwargs)

        # Reset observed and predicted target data
        if y is not None:
            self._y = y.copy()
        else:
            self._y = None
        self._y_pred = None
        self._y_pred_proba = None

        if self.refit:
            self._best_estimator = clone(self.estimator).set_params(**self.best_params_)
            self._best_estimator.fit(X, y=y, **self.fit_params)
        else:
            if hasattr(self, '_best_estimator'):
                del self._best_estimator

        return self

    @staticmethod
    def get_metrics_names(estimator, scoring=None):
        _scoring_name = None
        if scoring is not None:
            try:
                _scoring_name = scoring[0]
            except TypeError:
                _scoring_name = str(scoring)
        try:
            if is_classifier(estimator) == True:
                _metrics_names = (['Accuracy'] if _scoring_name is None else [_scoring_name]) + ['LogLoss']
            elif is_regressor(estimator) == True:
                _metrics_names = (['R2'] if _scoring_name is None else [_scoring_name]) + ['MAE', 'MSE']
        except AttributeError:
            raise TypeError('Unable to determine whether {} is a classifier or regressor'.format(estimator))
        return _metrics_names

    @staticmethod
    def compute_metrics(cv_estimator, estimator_name='', covariates_name='', scoring=None):
        """Compute the predicted goodness-of-fit metrics for a trained
        cross-validated estimator.

        **IMPORTANT NOTE**: Assuming that the passed CV estimator includes
        the predicted target values computed on the test predictor/
        covariate values from each training/test split, `compute_metrics`
        in effect returns weighted goodness-of-fit metrics by applying
        scoring functions to all non-held-out (see
        :class:`datastack.cerebro.cross_validation.HashedKfolds`)
        observed and predicted target values. These metrics will in general
        differ from the score returned in the `best_score_` parameter of
        :class:`sklearn.grid_search.GridSearchCV`-derived CV estimators,
        which return the score from a single training/test split.

        :param cv_estimator: A cross-validated estimator.
        :type cv_estimator:
            :class:`datastack.cerebro.cross_validation.CrossValidation`
        :param estimator_name: A pretty-printable name for the
            estimator; by default, an empty string.
        :param covariates_name: A pretty-printable name for the
            covariates used when training the estimator; by default,
            an empty string.
        :param scoring: A string or a scorer callable object / function
            with signature `scorer(estimator, X, y)`. If `None`, use
            the score method of the estimator. See
            :func:`sklearn.metrics.make_scorer`.
        """

        class _identity_estimator(object):
            """Dummy estimator used to pass `X` values straight through. Useful
            for calling scorers created by :func:`sklearn.metrics.make_scorer`
            that expect the observed predictor values instead of previously
            derived predicted target values.
            """

            def predict(self, X):
                return X

        _score_row = [estimator_name, covariates_name]
        # If we used an explicit scoring function as opposed
        # to the standard for the estimator, we need to
        # call the scoring function instead to get the
        # estimated goodness-of-fit metric. The problem is that
        # a scoring function generated by
        # sklearn.metrics.make_scorer() takes a trained
        # estimator and generates predictions from observed
        # test X, which is not what we want. Instead, then,
        # we pass in a dummy estimator that simply returns its
        # input as output, and pass in the y_pred values from
        # the cross-validation loop.
        if is_classifier(cv_estimator.estimator):
            if scoring is not None:
                _score_row.append(scoring(_identity_estimator(), cv_estimator.y_pred, cv_estimator.y))
            else:
                _score_row.append(metrics.accuracy_score(cv_estimator.y, cv_estimator.y_pred))
            # If the prediction probabilities are wrapped as an indexed
            # container, unwrap first otherwise log_loss crashes...
            if hasattr(cv_estimator.y_pred_proba, 'index'):
                _pred_proba = cv_estimator.y_pred_proba.values
            else:
                _pred_proba = cv_estimator.y_pred_proba
            _score_row.append(metrics.log_loss(cv_estimator.y, _pred_proba))
        elif is_regressor(cv_estimator.estimator):
            # See the big comment in the is_classifier() block.
            if scoring is not None:
                _score_row.append(scoring(_identity_estimator(), cv_estimator.y_pred, cv_estimator.y))
            else:
                _score_row.append(metrics.r2_score(cv_estimator.y, cv_estimator.y_pred))
            _score_row.append(metrics.mean_absolute_error(cv_estimator.y, cv_estimator.y_pred))
            _score_row.append(metrics.mean_squared_error(cv_estimator.y, cv_estimator.y_pred))
        else:
            raise ValueError('Unable to get type of cross-validated estimator')
        return _score_row
