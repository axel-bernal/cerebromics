# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 12:48:57 2016

Copyright (C) 2015 Human Longevity Inc. Written by Peter Garst.
"""

import datastack.ml.DepthColorPCs as depth
import conftest

def test_multi1 (multi1):
    m=50
    genPCs = ["genomics.kinship.pc."+str(x) for x in xrange(1,m+1,1)]
    modelPCs = depth.DepthColorPCs(data=multi1)
    baseColor = modelPCs.predictColorPCs(covPCs=genPCs)
    baseDepth = modelPCs.predictDepthPCs(covPCs=genPCs)
    modelColorCV=baseColor.cv[('AGE + SNPs', 'multi', 0)]
    modelDepthCV=baseDepth.cv[('AGE + BMI', 'multi', 0)]  

if __name__ == "__main__":
    """
    This version you can invoke from an IDE for debugging purposes.
    """        
    df = conftest.multi1(None)
    test_multi1(df)
