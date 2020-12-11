# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 08:20:04 2016

Copyright (C) 2016 Human Longevity Inc. Written by Peter Garst.
"""

import datastack.ml.baseregress as baseregress
import collections

def test_skin1 (skin1):       
    target_dict=collections.OrderedDict()
    for i in ['l','a','b']:
        target_dict['skin_color_{}'.format(i)]='dynamic.FACE.skin.v3.{}'.format(i)
    
    base = baseregress.BaseRegress(target_dict, multi=True)
    base.addData(skin1)
    
    fmt = {}
    fmt['MAE'] = 2
    fmt['MSE'] = 2
    fmt['R2'] =2
    
    base.run()
    base.dataframe.to_csv('skin.csv', index=False)
    base.display(fmtdict=fmt)
    
def test_skin2 (skin1):       
    target_dict=collections.OrderedDict()
    for i in ['l','a','b']:
        target_dict['skin_color_{}'.format(i)]='dynamic.FACE.skin.v3.{}'.format(i)
    
    base = baseregress.BaseRegress(target_dict, multi=False)
    base.addData(skin1)
    
    fmt = {}
    fmt['MAE'] = 2
    fmt['MSE'] = 2
    fmt['R2'] =2
    
    base.run()
    base.dataframe.to_csv('skin.csv', index=False)
    base.display(fmtdict=fmt)

if __name__ == "__main__":
    """
    This version you can invoke from an IDE for debugging purposes.
    """
    import conftest
    df = conftest.skin1(None)
    test_skin1(df)
    test_skin2(df)
