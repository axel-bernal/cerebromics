#!/usr/bin/env python
'''
Test that each subject gets assigned to the same fold regardless of the
number of other subjects under consideration.

Created on Apr 14, 2016

@author: twong
'''

import logging
import os
import pandas as pd
import unittest

from datastack.cerebro.cross_validation import HashedKfolds

_BASENAME = os.path.splitext(os.path.realpath(__file__))[0]
_TEST_DATA = _BASENAME + '.data'

_logger = logging.getLogger(__name__)


class TestKfoldSubsets(unittest.TestCase):
    """Test that each subject gets assigned to the same fold regardless of the number of other subjects under consideration.
    """
    @classmethod
    def setUpClass(cls):
        super(TestKfoldSubsets, cls).setUpClass()
        cls._test_df = pd.read_pickle(_TEST_DATA)
        # Get all males and make k-folds just for the males
        male_df = cls._test_df[cls._test_df['facepheno.Sex'] == 'Male']
        cls._male_kf = HashedKfolds(
            male_df,
            n_holdout=0,
            keep_in_holdout_columns=['dynamic.FACE.consented.v1.value'],
            keep_together_columns=['dynamic.FACE.related.v1.value'],
        )
        # Get the membership of all train/test splits for just males. Any
        # other train/test splits on some subset of males should produce
        # train/test splits that are subsets of these train/test splits.
        cls._male_train_sets = []
        cls._male_test_sets = []
        for train, test in cls._male_kf:
            cls._male_train_sets.append(cls._male_kf.df.iloc[train].index)
            cls._male_test_sets.append(cls._male_kf.df.iloc[test].index)
        # Do the same thing for females.
        female_df = cls._test_df[cls._test_df['facepheno.Sex'] == 'Female']
        cls._female_kf = HashedKfolds(
            female_df,
            n_holdout=0,
            keep_in_holdout_columns=['dynamic.FACE.consented.v1.value'],
            keep_together_columns=['dynamic.FACE.related.v1.value'],
        )
        cls._female_train_sets = []
        cls._female_test_sets = []
        for train, test in cls._female_kf:
            cls._female_train_sets.append(cls._female_kf.df.iloc[train].index)
            cls._female_test_sets.append(cls._female_kf.df.iloc[test].index)

    def testSubsets(self):
        """Test that each subject gets assigned to the same fold regardless of the number of other subjects under consideration.
        """
        df = self._test_df.sample(frac=0.5)
        kf = HashedKfolds(
            df,
            n_holdout=0,
            keep_in_holdout_columns=['dynamic.FACE.consented.v1.value'],
            keep_together_columns=['dynamic.FACE.related.v1.value'],
        )
        # Check that the male in the subset are assigned to the same
        # folds as with the male-only subset.
        male = df[df['facepheno.Sex'] == 'Male'].index
        self.assertTrue((kf.ids[male] == self._male_kf.ids[male]).all())
        # Likewise for the female.
        female = df[df['facepheno.Sex'] == 'Female'].index
        self.assertTrue((kf.ids[female] == self._female_kf.ids[female]).all())
        # Check that the male/female assignments to train/test splits
        # remains the same.
        splits = zip(kf, self._male_train_sets, self._male_test_sets, self._female_train_sets, self._female_test_sets)
        # For each train/test split, check each male and female subject
        # gets assigned to the same training and test split over the all
        # splits regardless of the number of other subjects used to create
        # the splits.
        for (sample_train, sample_test), male_train_golden, male_test_golden, female_train_golden, female_test_golden in splits:
            _train_df = kf.df.iloc[sample_train]
            _test_df = kf.df.iloc[sample_test]
            male_train = _train_df[_train_df['facepheno.Sex'] == 'Male'].index
            female_train = _train_df[_train_df['facepheno.Sex'] == 'Female'].index
            male_test = _test_df[_test_df['facepheno.Sex'] == 'Male'].index
            female_test = _test_df[_test_df['facepheno.Sex'] == 'Female'].index
            self.assertTrue(set(male_train) <= set(male_train_golden))
            self.assertTrue(set(female_train) <= set(female_train_golden))
            self.assertTrue(set(male_test) <= set(male_test_golden))
            self.assertTrue(set(female_test) <= set(female_test_golden))


def load_tests(loader, tests, pattern):
    """Helper function that the Python unittest framework automagically calls
    to create a test suite in a module. The advantage of this automagic
    approach over running the tests within a loop in a single
    `unittest.TestCase` subclass is that the unittest (or TeamCity messages)
    runner will treat each added test as a separate test case, presenting a
    discrete output message for each namespace-version combination.
    """
    _logger.debug('Creating test cases...')
    test_cases = unittest.TestSuite()
    # And now the magic: add N tests
    for _ in range(10):
        test_cases.addTest(TestKfoldSubsets('testSubsets'))
    return test_cases

if __name__ == "__main__":
    unittest.main()
