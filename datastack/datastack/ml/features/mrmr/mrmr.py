# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 14:16:46 2016

Copyright (C) 2015 Human Longevity Inc. Written by Peter Garst.
"""

import sys
import os
import logging

class MrMr (object):
    
    verbose = False
    
    def __init__ (self):
        # May need to add the path to the compiled mrmr library
        path = os.path.dirname(os.path.realpath(__file__)) + '/build'
        sys.path.append(path)
        import _mrmrlib
        reload(_mrmrlib)     
        self.mrptr = _mrmrlib.newmr()
    
    def selectCols (self, X, y, ndim, core):
        """
        Given a data frame select ndim columns to use as features using the
        wrapped mrmr c++ library, in addition to the core features if any.
        Return a list of columns to use as features.
        
        To help with testing, as a special case if ndim is 0 and core is None
        we return all the columns.
        """
        if (ndim <= 0) and (core is None):
            return list(X.columns)
        if ndim <= 0:
            return core
        
        if core is None:
            core = []
            
        val = X.values
        yval = y.values
        if len(y.shape) > 1:
            yval = yval[:, 0]
        cols = ['y'] + list(X.columns)
        # cols = list(X.columns)
        import _mrmrlib
        if self.verbose:
            print 'Start data'
            sys.stdout.flush()
        self.data = _mrmrlib.data(val, yval, cols, core)
        if self.verbose:
            print 'Start select'
            sys.stdout.flush()
        fcols = _mrmrlib.select(self.mrptr, self.data, ndim)
        if self.verbose:
            print 'End select', fcols
            sys.stdout.flush()
            
        # logging.info('Selected: ' + str(len(X)) + '   ' + fcols[22])
        
        return  fcols
  
# For debugging
if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv('dtestout7.csv')
    
    y = df[['dynamic.FACE.neyecolor.v1_visit1.l']]
    xcols = list(df.columns)
    xcols = [x for x in xcols if not 'neyecolor' in x]
    X = df[xcols]
    mrmr = MrMr()
    cols = mrmr.selectCols(X, y, 5, None)
    print cols