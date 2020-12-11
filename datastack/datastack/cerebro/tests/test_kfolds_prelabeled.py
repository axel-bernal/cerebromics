#!/usr/bin/env python
'''
Test using labels on subjects to define fold membership for k-folds.

Created on Jun 20, 2016

@author: twong
'''
import logging
import os
import pandas as pd
import unittest

from datastack.cerebro.cross_validation import CrossValidation, PrelabeledKfolds
from sklearn.linear_model import Ridge

_BASENAME = os.path.splitext(os.path.realpath(__file__))[0]
_TEST_DATA = _BASENAME + '.data'

_TEST_LABEL_COLUMN = 'ds.index.ProjectID'
_TEST_STUDIES = dict([(u'Ideker', 1), (u'Loomba', 18), (u'Spector', 236), (u'NMDP', 49), (u"O'Connor", 71),
                      (u'Haddad', 10), (u'Genentech', 257), (u'FACE', 106), (u'Celgene', 4), (u'Cleveland clinic', 100)])
_TEST_STUDY_FILTER = ['NMDP', 'FACE', 'Loomba']
_logger = logging.getLogger(__name__)


class TestPrelabeledKfolds(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestPrelabeledKfolds, cls).setUpClass()
        cls._test_df = pd.read_pickle(_TEST_DATA)

    def testPrelabeledKfolds(self):
        """Test that the prelabeled k-folds have ID values in the sorted order of the labels.
        """
        kf = PrelabeledKfolds(df=self._test_df, label_column=_TEST_LABEL_COLUMN)
        self.assertEqual(len(_TEST_STUDIES), len(kf), 'Got wrong number of folds: Expected {}, got {}'.format(len(_TEST_STUDIES), len(kf)))
        _id = 0
        for s in sorted(_TEST_STUDIES.keys()):
            _filter = kf.get_fold(s)
            self.assertEqual(
                _TEST_STUDIES[s],
                len(_filter),
                'Got wrong number of subject in the fold for {}: Expected {}, got {}'.format(s, _TEST_STUDIES[s], len(_filter))
            )
            self.assertTrue((kf.ids.loc[_filter] == _id).all())
            _id += 1

    def testFoldMembership(self):
        """Test that each fold contains subjects with the same label.
        """
        kf = PrelabeledKfolds(df=self._test_df, label_column=_TEST_LABEL_COLUMN)
        for i in range(len(kf)):
            self.assertEqual(1, len(set(kf.df.loc[kf.get_fold(i), _TEST_LABEL_COLUMN])))

    def testUnlabeledSubjectsInTrainingSplit(self):
        """Test that unlabeled subjects only ever go into the training split.
        """
        unlabeled_subjects = set(self._test_df[self._test_df[_TEST_LABEL_COLUMN].isnull()].index)
        kf = PrelabeledKfolds(df=self._test_df, label_column=_TEST_LABEL_COLUMN)
        for train, _ in kf:
            self.assertTrue(set(kf.df.iloc[train].index) & unlabeled_subjects == unlabeled_subjects)

    def testPrelabeledKfoldsDropColumn(self):
        """Test that inplace changes modify the original dataframe
        """
        _df = self._test_df.copy()
        kf = PrelabeledKfolds(df=_df, label_column=_TEST_LABEL_COLUMN, drop_label_column=True)
        # Next check implicitly ensures that deleting the label column
        # doesn't break k-fold member functions
        self.assertEqual(set(_TEST_STUDIES), set(kf.labels))
        self.assertTrue(_TEST_LABEL_COLUMN not in kf.df.columns)
        self.assertTrue(_TEST_LABEL_COLUMN not in _df)

    def testPrelabeledKfoldsWithFilter(self):
        """Test that restricting the label values used to make folds still produces well-formed k-folds.
        """
        _TEST_STUDY_FILTER = ['NMDP', 'FACE', 'Loomba']
        kf = PrelabeledKfolds(df=self._test_df, label_column=_TEST_LABEL_COLUMN, label_filter=_TEST_STUDY_FILTER)
        self.assertEqual(len(_TEST_STUDY_FILTER), len(kf), 'Got wrong number of folds: Expected {}, got {}'.format(len(_TEST_STUDY_FILTER), len(kf)))
        self.assertEqual(
            set(_TEST_STUDY_FILTER),
            set(kf.df[_TEST_LABEL_COLUMN]),
            'Got wrong labels in folds: Expected {}, got {}'.format(set(_TEST_STUDY_FILTER), set(kf.df[_TEST_LABEL_COLUMN]))
        )
        _id = 0
        for s in sorted(_TEST_STUDY_FILTER):
            _filter = kf.get_fold(s)
            self.assertEqual(
                _TEST_STUDIES[s],
                len(_filter),
                'Got wrong number of subject in the fold for {}: Expected {}, got {}'.format(s, _TEST_STUDIES[s], len(_filter))
            )
            self.assertTrue((kf.ids.loc[_filter] == _id).all())
            _id += 1
        for i in range(len(kf)):
            self.assertEqual(1, len(set(kf.df.loc[kf.get_fold(i)][_TEST_LABEL_COLUMN])))
        # This last test verifies that no unlabeled subjects made it into a
        # fold.
        for train, _ in kf:
            self.assertFalse(kf.df.loc[kf.df.index[train], _TEST_LABEL_COLUMN].isnull().any())


class TestPrelabeledKfoldsWithCrossValidation(unittest.TestCase):

    _TEST_RIDGE_PARAM_GRID = {
        "alpha": [
            0.001,
            0.01,
            0.1,
            1.01,
            10.0
        ]
    }

    @classmethod
    def setUpClass(cls):
        super(TestPrelabeledKfoldsWithCrossValidation, cls).setUpClass()
        cls._test_df = pd.read_pickle(_TEST_DATA)
        cls._test_kf = PrelabeledKfolds(df=cls._test_df, label_column=_TEST_LABEL_COLUMN)
        cls._model = CrossValidation(
            Ridge(),
            cls._TEST_RIDGE_PARAM_GRID,
            cv=cls._test_kf,
        )
        cls._model.fit(cls._test_df[['kf.age']], cls._test_df['kf.age_p'])

    def testPrelabeledKfoldsLabels(self):
        """Test that the split IDs in a cross-validated grid search with prelabeled k-folds correspond to the left-out fold ID
        """
        for g in self._model.grid_scores_:
            labels = list(set(self._test_kf.df.iloc[g.test_index][_TEST_LABEL_COLUMN]))
            self.assertEqual(1, len(labels))
            self.assertEqual(labels[0], self._test_kf.get_fold_label(g.split_id))


if __name__ == "__main__":
    unittest.main()
