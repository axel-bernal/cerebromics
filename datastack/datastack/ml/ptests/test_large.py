# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 18:01:20 2015

Copyright (C) 2015 Human Longevity Inc. Written by Peter Garst.
"""

import datastack.ml.baseregress as br
import datastack.ml.ptests.conftest as conftest
import datastack.settings as settings
import logging
import unittest

_logger = logging.getLogger(__name__)


@unittest.skipIf(settings.DATASTACK_TEST_STANDALONE == True, 'Running tests in stand-alone mode')
class TestLarge(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.age = conftest.age(None)
        cls.height = conftest.height(None)

    def test_large(self):
        """Test that we can use them all rows from Rosetta in regression models.
        """
        targets = {'height': 'pheno.height'}
        base = br.BaseRegress(targets)
        base.addData(self.age)
        base.addData(self.height)
        base.covariates['age'] = ['pheno.age']
        base.covariates['gender'] = ['pheno.gender']
        self.assertTrue(base.run(run_keys=['age', 'gender'], with_bootstrap=False))
        nsamp = base.metrics_df.iloc[0]['Samples']
        _logger.info('Found %d samples' % (nsamp))
        self.assertGreater(nsamp, 3000)

if __name__ == '__main__':
    logging.basicConfig()
    unittest.main()
