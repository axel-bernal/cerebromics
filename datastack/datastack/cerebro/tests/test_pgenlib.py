#!/usr/bin/env python
'''
Test Pgenlib Python API.

Created on Nov 8, 2016

@author: cchang
'''
import logging
import numpy as np
import os
import random
import unittest

import pgenlib

_BASENAME = os.path.splitext(os.path.realpath(__file__))[0]
_TEST_PGEN = _BASENAME + '.pgen'

class TestPgenlib(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super(TestPgenlib, cls).setUpClass()
        cls._pgr = pgenlib.PgenReader(_TEST_PGEN)

    def testCounts(self):
        """Test that PgenReader.count() and PgenReader.read() are consistent.
        """
        self._pgr.change_sample_subset()
        sample_ct = self._pgr.get_raw_sample_ct()
        variant_ct = self._pgr.get_variant_ct()
        genobuf = np.empty(sample_ct, np.int8)
        countbuf = np.empty(4, np.uint32)
        countbuf2 = np.empty(4, np.uint32)
        for iters in xrange(10):
            vidx = random.randrange(variant_ct)
            self._pgr.read(vidx, genobuf)
            for geno_idx in xrange(4):
                countbuf[geno_idx] = 0
            for sample_idx in xrange(sample_ct):
                cur_geno = genobuf[sample_idx]
                if cur_geno == -9:
                    cur_geno = 3
                countbuf[cur_geno] += 1
            self._pgr.count(vidx, countbuf2)
            for geno_idx in xrange(4):
                self.assertEqual(countbuf[geno_idx], countbuf2[geno_idx], 'Inconsistent genotype counts: Explicit loop yielded {}, count function yielded {}'.format(countbuf[geno_idx], countbuf2[geno_idx]))


if __name__ == "__main__":
    unittest.main()
