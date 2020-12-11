#!/usr/bin/env python
'''
Tests for classes that standardize `scikit-learn`-style cross-validation
subject splits on HLI data.

Created on Oct 30, 2015

@author: twong
'''

import datastack.settings as settings
import json
import logging
import md5
import os.path
import pandas as pd
import random
import unittest

import datastack.cerebro.cross_validation as cv

_BASENAME = os.path.splitext(os.path.realpath(__file__))[0]
_TEST_DATA = _BASENAME + '.data'
_TEST_RELATED_DATA = _BASENAME + '_related.data'

_INDEX_COLUMN = 'ds.index.sample_key'
_KEEP_TOGETHER_COLUMN = 'related.to'
_KEEP_TOGETHER_COLUMN_A = 'related.to.a'
_KEEP_TOGETHER_COLUMN_B = 'related.to.b'

_KEEP_BOOLEAN_COLUMN_A = 'member.a'
_KEEP_BOOLEAN_COLUMN_B = 'member.b'
_KEEP_BOOLEAN_COLUMN_C = 'member.c'
_KEEP_BOOLEAN_COLUMN_D = 'member.d'
_KEEP_BOOLEAN_COLUMN_E = 'member.e'
_KEEP_BOOLEAN_COLUMNS = [
    _KEEP_BOOLEAN_COLUMN_A,
    _KEEP_BOOLEAN_COLUMN_B,
    _KEEP_BOOLEAN_COLUMN_C,
    _KEEP_BOOLEAN_COLUMN_D,
    _KEEP_BOOLEAN_COLUMN_E,
]

_N_TRAINING_GOLDEN = 10
_N_HOLDOUT_GOLDEN = 2

_SUBJECT_KFOLD_GOLDEN_COLUMN = 'fold.golden'


class Test_hash(unittest.TestCase):
    """Test the hash function used to HLI subjects to k-folds.
    """

    def testHash(self):
        random.seed(0)
        for salt in [None] + [chr(ord('a') + i) for i in range(0, 26)]:
            for _ in range(0, 100):
                n_folds = int(random.randint(5, 10))
                key = '%d' % (int(random.random()))
                m = md5.new()
                m.update(key + ('' if salt is None else salt))
                self.assertEqual(int(m.hexdigest(), 16) %
                                 n_folds, cv._hash(key, n_folds, salt=salt))


class Test_verify_keys_in_df(unittest.TestCase):
    """Test the hash function used to verify that a set of keys is in a dataframe (and normalize the keys into a list)
    """

    @classmethod
    def setUpClass(cls):
        super(Test_verify_keys_in_df, cls).setUpClass()
        cls._test_df = pd.read_pickle(_TEST_RELATED_DATA)

    def testVerifySingleKeyInDf(self):
        keys = cv._verify_keys_in_df(self._test_df, _INDEX_COLUMN)
        self.assertIs(type(keys), list)
        self.assertEqual(1, len(keys))
        self.assertTrue(_INDEX_COLUMN in keys)

    def testVerifyMultipleKeysInDf(self):
        keys = cv._verify_keys_in_df(
            self._test_df, [
                _INDEX_COLUMN, _KEEP_TOGETHER_COLUMN])
        self.assertIs(type(keys), list)
        self.assertEqual(2, len(keys))
        self.assertTrue(_INDEX_COLUMN in keys)
        self.assertTrue(_KEEP_TOGETHER_COLUMN in keys)

    def testVerifyKeyNotInDf(self):
        with self.assertRaises(KeyError):
            cv._verify_keys_in_df(
                self._test_df, [
                    _INDEX_COLUMN + _KEEP_TOGETHER_COLUMN])


class TestHashedKfold(unittest.TestCase):
    """Test the hashed k-fold generator that assigns HLI subjects to k-folds.
    """

    def _check_unrelated(self, kf):
        """Check that subjects that do not have related subjects are not
        affected by constraint operations on related subjects, into-training,
        or into-holdout.
        """
        for i in range(0, kf.n_training + kf.n_holdout):
            df = kf._df.loc[kf.get_fold(i)]
            # Get a boolean selector of the subjects that have no related
            # subjects
            unrelated = df[_KEEP_TOGETHER_COLUMN].isnull()
            self.assertTrue((df[unrelated][_SUBJECT_KFOLD_GOLDEN_COLUMN] == i).all())

    @classmethod
    def setUpClass(cls):
        super(TestHashedKfold, cls).setUpClass()
        cls._test_df = pd.read_pickle(_TEST_DATA)
        cls._related_df = pd.read_pickle(_TEST_RELATED_DATA)

    def testEmptyDataframe(self):
        """Test that passing in an empty dataframe yields a well-formed data structure (albeit with no subjects)
        """
        empty_df = pd.DataFrame(columns=[_INDEX_COLUMN])
        kf = cv.HashedKfolds(df=empty_df, index_column=_INDEX_COLUMN, inplace=True)
        self.assertEqual(0, kf.ids.shape[0])
        self.assertEqual(10, kf.n_training)
        self.assertEqual(2, kf.n_holdout)
        self.assertEqual([], kf.keep_in_training)
        self.assertEqual([], kf.keep_in_holdout)
        self.assertEqual([], kf.keep_together)
        self.assertEqual(empty_df.columns, kf.df.columns)

    def testTraining(self):
        """Test that training subjects appear in all training/test splits
        """
        kf = cv.HashedKfolds(df=self._test_df, index_column=_INDEX_COLUMN)
        for training, test in kf:
            training_keys = kf.df.iloc[training].index | kf.df.iloc[test].index
            for i in kf.df.index:
                self.assertTrue(i in training_keys)

    def testHoldout(self):
        """Test that holdout subjects do not appear in any training/test split
        """
        kf = cv.HashedKfolds(df=self._test_df, index_column=_INDEX_COLUMN)
        for training, test in kf:
            training_keys = kf.df.iloc[training].index | kf.df.iloc[test].index
            for i in kf.df_holdout.index:
                self.assertTrue(i not in training_keys)

    def testIndexColumn(self):
        """Test that the index column name whatever we specify `qc.sample_key`
        """
        kf = cv.HashedKfolds(df=self._test_df, index_column=_INDEX_COLUMN)
        self.assertEqual(_INDEX_COLUMN, kf.index_column)

    def testLen(self):
        """Test that the `len` operator returns only the number of training folds.
        """
        kf = cv.HashedKfolds(
            df=self._test_df,
            index_column=_INDEX_COLUMN,
            n_training=10,
            n_holdout=2)
        self.assertEqual(_N_TRAINING_GOLDEN, len(kf))

    def testWithHg19(self):
        """Test that the generator returns the same folds for the same subjects as the original k-fold generator.
        """
        kf = cv.HashedKfolds(df=self._test_df, index_column=_INDEX_COLUMN)
        for i in range(0, kf.n_training + kf.n_holdout):
            self.assertTrue(
                (kf._df.loc[
                    kf.get_fold(i)][_SUBJECT_KFOLD_GOLDEN_COLUMN] == i).all())

    def testWithKeepTogetherColumn(self):
        """Test that the generator returns the same folds for the same related subjects as the original k-fold generator.
        """
        kf = cv.HashedKfolds(
            df=self._related_df,
            index_column=_INDEX_COLUMN,
            keep_together_columns=_KEEP_TOGETHER_COLUMN)
        for i in range(0, kf.n_training + kf.n_holdout):
            self.assertTrue(
                (kf._df.loc[
                    kf.get_fold(i)][_SUBJECT_KFOLD_GOLDEN_COLUMN] == i).all())

    def testWithKeepTogetherColumnAsList(self):
        """Test that the generator returns the same folds for the same related subjects as the original k-fold generator, with the related-to columns specified as a list of multiple columns
        """
        kf = cv.HashedKfolds(
            df=self._related_df,
            index_column=_INDEX_COLUMN,
            keep_together_columns=[_KEEP_TOGETHER_COLUMN_A, _KEEP_TOGETHER_COLUMN_B])
        for i in range(0, kf.n_training + kf.n_holdout):
            self.assertTrue(
                (kf._df.loc[
                    kf.get_fold(i)][_SUBJECT_KFOLD_GOLDEN_COLUMN] == i).all())

    def testWithKeepTogetherColumnAndUnrelatedData(self):
        """Test that the generator returns subjects with no related subjects hashed into the same folds as before.
        """
        # Create a single data frame that combines subjects with related
        # subjects and subjects with no related subjects
        df = self._test_df.join(
            self._related_df[
                [_KEEP_TOGETHER_COLUMN]],
            how='outer')
        kf = cv.HashedKfolds(df=df, index_column=_INDEX_COLUMN, keep_together_columns=_KEEP_TOGETHER_COLUMN)
        self._check_unrelated(kf)

    @unittest.expectedFailure
    def testWithKeepTogetherColumnAndNewRelatedSubject(self):
        """Test that the generator returns the same folds as the original k-fold generator, even after we add new related subjects. Currently known not to work.
        """
        df = self._related_df.copy()
        # Try every 1-9 ordinal to see if the fold assignment for related
        # subjects changes
        for i in range(0, 9):
            new_subject = str(i) + df.ix[0, _INDEX_COLUMN]
            new_related_subjects = [new_subject] + \
                json.loads(df.iloc[0][_KEEP_TOGETHER_COLUMN])
            df.ix[0, _KEEP_TOGETHER_COLUMN] = json.dumps(
                new_related_subjects)
            kf = cv.HashedKfolds(
                df=df,
                index_column=_INDEX_COLUMN,
                keep_together_columns=_KEEP_TOGETHER_COLUMN)
            for i in range(0, kf.n_training + kf.n_holdout):
                self.assertTrue(
                    (kf._df.loc[kf.get_fold(i)][
                        _SUBJECT_KFOLD_GOLDEN_COLUMN] == i).all(),
                    'Adding new subjects to related subject set changes the fold assignment')

    def testWithInTrainingSubjects(self):
        """Test that the generator constrains subjects into training sets, where the constraint comes from a single column in a dataframe.
        """
        df = self._test_df.join(
            self._related_df[[_KEEP_TOGETHER_COLUMN] + _KEEP_BOOLEAN_COLUMNS],
            how='outer')
        kf = cv.HashedKfolds(
            df=df,
            index_column=_INDEX_COLUMN,
            keep_in_training_columns=_KEEP_BOOLEAN_COLUMN_A)
        index = kf._df[_KEEP_BOOLEAN_COLUMN_A].fillna(False)
        self.assertTrue((kf.ids[index] < kf.n_training).all())
        self._check_unrelated(kf)

    def testWithInTrainingSubjectsAsList(self):
        """Test that the generator constrains subjects into training sets, where the constraint comes from a multiple columns in a dataframe.
        """
        df = self._test_df.join(
            self._related_df[[_KEEP_TOGETHER_COLUMN] + _KEEP_BOOLEAN_COLUMNS],
            how='outer')
        kf = cv.HashedKfolds(
            df=df,
            index_column=_INDEX_COLUMN,
            keep_in_training_columns=[_KEEP_BOOLEAN_COLUMN_B, _KEEP_BOOLEAN_COLUMN_C])
        index = (kf._df[_KEEP_BOOLEAN_COLUMN_B] |
                 kf._df[_KEEP_BOOLEAN_COLUMN_C]).fillna(False)
        self.assertTrue((kf.ids[index] < kf.n_training).all())
        self._check_unrelated(kf)

    def testWithInHoldoutSubjects(self):
        """Test that the generator constrains subjects into holdout sets, where the constraint comes from a single column in a dataframe.
        """
        df = self._test_df.join(
            self._related_df[[_KEEP_TOGETHER_COLUMN] + _KEEP_BOOLEAN_COLUMNS],
            how='outer')
        kf = cv.HashedKfolds(
            df=df,
            index_column=_INDEX_COLUMN,
            keep_in_holdout_columns=_KEEP_BOOLEAN_COLUMN_A)
        index = kf._df[_KEEP_BOOLEAN_COLUMN_A].fillna(False)
        self.assertTrue((kf.ids[index] >= kf.n_training).all())
        self.assertTrue((kf.ids[index] < kf._n_folds).all())
        self._check_unrelated(kf)

    def testWithInHoldoutSubjectsAndZeroHoldoutFolds(self):
        """Test that the generator constrains subjects *out* of the training folds if the user specifies a holdout column.
        """
        df = self._test_df.join(self._related_df[[_KEEP_TOGETHER_COLUMN] + _KEEP_BOOLEAN_COLUMNS], how='outer')
        kf = cv.HashedKfolds(df=df, n_holdout=0, index_column=_INDEX_COLUMN, keep_in_holdout_columns=_KEEP_BOOLEAN_COLUMN_A)
        index = kf._df[_KEEP_BOOLEAN_COLUMN_A].fillna(False)
        self.assertTrue((kf.ids[index] < 0).all())
        self.assertTrue(kf.in_holdout(kf.ids[index]).all())
        self.assertTrue((kf.ids[index] < kf._n_folds).all())
        # Note that we don't call _check_unrelated() since some IDs will
        # now be less than zero.

    def testWithInHoldoutSubjectsAsList(self):
        """Test that the generator constrains subjects into holdout sets, where the constraint comes from a multiple columns in a dataframe.
        """
        df = self._test_df.join(
            self._related_df[[_KEEP_TOGETHER_COLUMN] + _KEEP_BOOLEAN_COLUMNS],
            how='outer')
        kf = cv.HashedKfolds(
            df=df,
            index_column=_INDEX_COLUMN,
            keep_in_holdout_columns=[_KEEP_BOOLEAN_COLUMN_B, _KEEP_BOOLEAN_COLUMN_C])
        index = (kf._df[_KEEP_BOOLEAN_COLUMN_B] |
                 kf._df[_KEEP_BOOLEAN_COLUMN_C]).fillna(False)
        self.assertTrue((kf.ids[index] >= kf.n_training).all())
        self.assertTrue((kf.ids[index] < kf._n_folds).all())
        self._check_unrelated(kf)

    def testWithInHoldoutSubjectsAndKeepTogetherColumn(self):
        """Test that the generator (a) keeps related subjects in the same fold, and (b) ensures that if a subject is constrained into the holdout set, then its related subjects are also constrained into the holdout set.
                """
        df = self._test_df.join(self._related_df[[_KEEP_TOGETHER_COLUMN] + _KEEP_BOOLEAN_COLUMNS], how='outer')
        # Gather all holdout subjects AND their related subjects and make sure
        # they are all in the holdout folds.
        together = []
        for t in df[_KEEP_TOGETHER_COLUMN][~df[_KEEP_TOGETHER_COLUMN].isnull()]:
            together += json.loads(t)
        index = df[_KEEP_BOOLEAN_COLUMN_D].fillna(False) | df[_INDEX_COLUMN].isin(together)
        kf = cv.HashedKfolds(
            df=df,
            index_column=_INDEX_COLUMN,
            keep_in_holdout_columns=_KEEP_BOOLEAN_COLUMN_D,
            keep_together_columns=_KEEP_TOGETHER_COLUMN)
        self.assertEqual(set([_KEEP_TOGETHER_COLUMN, _KEEP_BOOLEAN_COLUMN_D]), set(kf.df.columns) & set([_KEEP_TOGETHER_COLUMN, _KEEP_BOOLEAN_COLUMN_D]))
        self.assertEqual(index.shape[0], df.shape[0])
        self.assertTrue((kf.ids[index] >= kf.n_training).all())
        self.assertTrue((kf.ids[index] < kf._n_folds).all())
        self._check_unrelated(kf)

    def testWithInHoldoutSubjectsAndKeepTogetherColumnAndDroppingMetadata(self):
        """Test that the generator (a) keeps related subjects in the same fold, (b) ensures that if a subject is constrained into the holdout set, then its related subjects are also constrained into the holdout set, and (c) that we drop metadata columns after processing.
                """
        df = self._test_df.join(self._related_df[[_KEEP_TOGETHER_COLUMN] + _KEEP_BOOLEAN_COLUMNS], how='outer')
        # Gather all holdout subjects AND their related subjects and make sure
        # they are all in the holdout folds.
        together = []
        for t in df[_KEEP_TOGETHER_COLUMN][~df[_KEEP_TOGETHER_COLUMN].isnull()]:
            together += json.loads(t)
        index = df[_KEEP_BOOLEAN_COLUMN_D].fillna(False) | df[_INDEX_COLUMN].isin(together)
        kf = cv.HashedKfolds(
            df=df,
            index_column=_INDEX_COLUMN,
            keep_in_holdout_columns=_KEEP_BOOLEAN_COLUMN_D,
            keep_together_columns=_KEEP_TOGETHER_COLUMN,
            drop_metadata_columns=True)
        self.assertEqual(0, len(set(kf.df.columns) & set([_KEEP_TOGETHER_COLUMN, _KEEP_BOOLEAN_COLUMN_D])))
        self.assertEqual(index.shape[0], df.shape[0])
        self.assertTrue((kf.ids[index] >= kf.n_training).all())
        self.assertTrue((kf.ids[index] < kf._n_folds).all())

    def testWithInTrainingAndInHoldout(self):
        """Test that the generator constrains subjects into training and holdout sets in one call.
        """
        df = self._test_df.join(
            self._related_df[[_KEEP_TOGETHER_COLUMN] + _KEEP_BOOLEAN_COLUMNS],
            how='outer')
        kf = cv.HashedKfolds(
            df=df,
            index_column=_INDEX_COLUMN,
            keep_in_training_columns=_KEEP_BOOLEAN_COLUMN_B,
            keep_in_holdout_columns=_KEEP_BOOLEAN_COLUMN_C)
        index = kf._df[_KEEP_BOOLEAN_COLUMN_B].fillna(False)
        self.assertTrue((kf.ids[index] < kf.n_training).all())
        index = kf._df[_KEEP_BOOLEAN_COLUMN_C].fillna(False)
        self.assertTrue((kf.ids[index] >= kf.n_training).all())
        self.assertTrue((kf.ids[index] < kf._n_folds).all())
        self._check_unrelated(kf)

    def testWithDataframeIndex(self):
        """Test that the generator returns the same folds for the same related subjects as the original k-fold generator.
        """
        # Make a copy so as not to break the class-wide dataframe
        df = self._related_df.copy()
        df.index = pd.Index(list(df[_INDEX_COLUMN]))
        kf = cv.HashedKfolds(df=df, keep_together_columns=_KEEP_TOGETHER_COLUMN)
        for i in range(0, kf.n_training + kf.n_holdout):
            self.assertTrue(
                (kf._df.loc[
                    kf.get_fold(i)][_SUBJECT_KFOLD_GOLDEN_COLUMN] == i).all())

    def testWithOverlappingInTrainingAndInHoldout(self):
        """Test that the generator rejects overlapping constraints.
        """
        df = self._test_df.join(
            self._related_df[_KEEP_BOOLEAN_COLUMNS],
            how='outer')
        with self.assertRaises(ValueError):
            cv.HashedKfolds(
                df=df,
                index_column=_INDEX_COLUMN,
                keep_in_training_columns=_KEEP_BOOLEAN_COLUMN_A,
                keep_in_holdout_columns=_KEEP_BOOLEAN_COLUMN_B)

    def testWithOverlappingInTrainingAndInHoldoutAndKeepTogetherColumn(self):
        """Test that the generator rejects TRANSITIVE overlapping constraints caused through related subjects.
        """
        df = self._test_df.join(
            self._related_df[[_KEEP_TOGETHER_COLUMN] + _KEEP_BOOLEAN_COLUMNS],
            how='outer')
        with self.assertRaises(ValueError):
            cv.HashedKfolds(
                df=df,
                index_column=_INDEX_COLUMN,
                keep_in_training_columns=_KEEP_BOOLEAN_COLUMN_D,
                keep_in_holdout_columns=_KEEP_BOOLEAN_COLUMN_E,
                keep_together_columns=_KEEP_TOGETHER_COLUMN)

    def testWithMissingIndexColumn(self):
        """Test that the k-fold generator stops if the source of hash keys is missing.
        """
        with self.assertRaises(KeyError):
            cv.HashedKfolds(
                df=self._test_df.drop([_INDEX_COLUMN], axis=1),
                index_column=_INDEX_COLUMN,
                keep_in_training_columns=[])

    def testWithMissingKeepTogetherColumn(self):
        """Test that the k-fold generator stops if the source of related-to is missing (if related-to is specified).
        """
        with self.assertRaises(KeyError):
            cv.HashedKfolds(
                df=self._related_df.drop([_KEEP_TOGETHER_COLUMN], axis=1),
                index_column=_INDEX_COLUMN,
                keep_together_columns=_KEEP_TOGETHER_COLUMN)

    def testWithFewerSubjects(self):
        """Test that k-fold assignment do not change if we use a subset of the subjects from a previous set of k-folds.
        """
        for i in range(10):
            kf = cv.HashedKfolds(df=self._test_df, index_column=_INDEX_COLUMN)
            small_df = self._test_df.sample(frac=0.5, random_state=i)
            small_kf = cv.HashedKfolds(df=small_df, index_column=_INDEX_COLUMN)
            for k in small_kf._df.index:
                self.assertEqual(kf.ids[k], small_kf.ids[k])


class TestHashedKfoldWithSampleName(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestHashedKfoldWithSampleName, cls).setUpClass()
        cls._test_df = pd.read_pickle(_TEST_DATA)
        cls._test_df[settings.ROSETTA_SUBJECT_NAME_KEY] = cls._test_df[_INDEX_COLUMN].apply(lambda k: k.split('_')[1])
        del cls._test_df[_INDEX_COLUMN]

    def testTraining(self):
        """Test that training subjects appear in all training/test splits
        """
        kf = cv.HashedKfolds(df=self._test_df, index_column=settings.ROSETTA_SUBJECT_NAME_KEY)
        for training, test in kf:
            training_keys = kf.df.iloc[training].index | kf.df.iloc[test].index
            for i in kf.df.index:
                self.assertTrue(i in training_keys)

    def testHoldout(self):
        """Test that holdout subjects do not appear in any training/test split
        """
        kf = cv.HashedKfolds(df=self._test_df, index_column=settings.ROSETTA_SUBJECT_NAME_KEY)
        for training, test in kf:
            training_keys = kf.df.iloc[training].index | kf.df.iloc[test].index
            for i in kf.df_holdout.index:
                self.assertTrue(i not in training_keys)


if __name__ == '__main__':
    logging.basicConfig()
    unittest.main()
