#!/usr/bin/env python
'''
Tests to ensure that we constraint consented and related subjects to be in
certain k-folds.

Created on Feb 05, 2016

@author: pgarst twong
'''

import datastack.cerebro.cross_validation as kfolds
import datastack.common.settings as settings
import logging
import unittest

from datastack.dbs.rdb import RosettaDBMongo

_NAMESPACES = ['hg19', 'hg38_noEBV']
_CONSENTED_COLUMN = 'dynamic.FACE.consented.v1.value'
_RELATED_COLUMN = 'dynamic.FACE.related.v1.value'


_CONSENTED_SUBJECTS = ['176501039', '187525153', '187522714']
_RELATED_SUBJECTS = ['187525200', '187525176', '176447815']

_logger = logging.getLogger(__name__)
_rdb = None


def _check_consented(kfolds):
    """Test that a trio of known-consented subjects appear in the holdout set
    """
    ids = kfolds.ids[
        kfolds._df[
            settings.ROSETTA_INDEX_KEY].str.match(
            '|'.join(['.*_%s$' % (s) for s in _CONSENTED_SUBJECTS]))]
    return ((ids >= kfolds.n_training) & (
        ids < (kfolds.n_training + kfolds.n_holdout))).all()


def _check_related(kfolds):
    """Test that a trio of known-related subjects appear in the same kfold
    """
    ids = kfolds.ids[
        kfolds._df[settings.ROSETTA_INDEX_KEY].str.match('|'.join(['.*_%s$' % (s) for s in _RELATED_SUBJECTS]))]
    return len(ids.unique()) == 1


class TestConsentedRelatedRosetta(unittest.TestCase):

    def __init__(self, method, namespace, version):
        super(TestConsentedRelatedRosetta, self).__init__(method)
        self.namespace = namespace
        self.version = version

    def run_test_consented(self):
        r_name = 'Rosetta %s v.%s' % (self.namespace, self.version)
        _rdb.initialize(namespace=self.namespace, version=self.version)
        if len(_rdb.find_keys(_CONSENTED_COLUMN)) <= 0:
            _logger.warning(
                'Column \'%s\' not in %s' %
                (_CONSENTED_COLUMN, r_name))
            return
        keys = [
            settings.ROSETTA_INDEX_KEY,
            _CONSENTED_COLUMN,
        ]
        filters = {}
        filters['ds.index.ProjectID'] = 'FACE'
        df = _rdb.query(keys=keys, filters=filters)
        kfolds = kfolds.HashedKfolds(
            df=df,
            index_column=settings.ROSETTA_INDEX_KEY,
            keep_in_holdout_columns=_CONSENTED_COLUMN)
        self.assertTrue(
            _check_consented(kfolds),
            'Consented subjects are NOT in holdout folds in %s' %
            (r_name))

    def run_test_related(self):
        r_name = 'Rosetta %s v.%s' % (self.namespace, self.version)
        _rdb.initialize(namespace=self.namespace, version=self.version)
        if len(_rdb.find_keys(_RELATED_COLUMN)) <= 0:
            _logger.warning(
                'Column \'%s\' not in %s' %
                (_RELATED_COLUMN, r_name))
            return
        keys = [
            settings.ROSETTA_INDEX_KEY,
            _RELATED_COLUMN,
        ]
        filters = {}
        filters['ds.index.ProjectID'] = 'FACE'
        df = _rdb.query(keys=keys, filters=filters)
        # If we're using subject keys instead of names as the index key,
        # recover the name from the key - related names specify names, not
        # keys
        if settings.ROSETTA_INDEX_KEY == 'ds.index.sample_key':
            df['_sample_name'] = df[settings.ROSETTA_INDEX_KEY].apply(lambda k: k.split('_')[1])
        kfolds = kfolds.HashedKfolds(df=df, index_column='_sample_name', keep_together_columns=_RELATED_COLUMN)
        self.assertTrue(
            _check_related(kfolds),
            'Related subjects are NOT in the same fold in %s' %
            (r_name))

# If the user declares the DATASTACK_TEST_STANDALONE in the environment,
# we skip this set of tests. Useful for testing on non-networked test
# hosts.


@unittest.skipIf(settings.DATASTACK_TEST_STANDALONE == True, 'Running tests in stand-alone mode')
def load_tests(loader, tests, pattern):
    """Helper function that the Python unittest framework automagically calls
    to create a test suite in a module. We use it to create a test suite to
    test consented and related subject column processing across multiple
    namespaces and Rosetta versions. The advantage of this automagic approach
    over running the tests within a loop in a single `unittest.TestCase`
    subclass is that the unittest (or TeamCity messages) runner will treat
    each added test as a separate test case, presenting a discrete output
    message for each namespace-version combination.
    """
    global _rdb
    _logger.debug('Connecting to Rosetta...')
    # Hacky, but opening multiple RDB connections on a single host seems to
    # bog Python down.
    _rdb = RosettaDBMongo()
    _logger.debug('Creating test cases...')
    test_cases = unittest.TestSuite()
    # And now the magic: add a consent test and a related test for each
    # accesible namespace-version combination, yielding a total of
    # N(namespaces) x N(versions) x 2 tests.
    for n in _NAMESPACES:
        for v in _rdb.get_versions(namespace=n):
            test_cases.addTest(
                TestConsentedRelatedRosetta(
                    'run_test_consented',
                    n,
                    v))
            test_cases.addTest(
                TestConsentedRelatedRosetta(
                    'run_test_related',
                    n,
                    v))
    return test_cases

if __name__ == '__main__':
    logging.basicConfig()
    unittest.main()
