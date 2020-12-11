# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 10:52:52 2015

Copyright (C) 2015 Human Longevity Inc. Written by Peter Garst.

Run an estimator repeatedly on different random splits of the data in
order to get mean and median error performance.
Used to find training curves.
"""

import numpy as np
import sklearn.metrics as metrics
from sklearn.base import clone

def split (frame, frac=0.9):
    """
    Get random test and train frames.
    random.sample doesn't work, try something else.
    """
    samps = list(range(len(frame)))
    np.random.shuffle(samps)
    trsize = int(len(frame) * frac)
    use = samps[:trsize]
    other = samps[trsize:]
    trainf = frame.iloc[use]
    testf = frame.iloc[other]
    return trainf, testf
                       
def shuffleRegress (est, feat, targ, frame, frac=0.9, ntries = 100):
    """
    Get a shuffle estimate of accuracy.
    How many iterations to run? Aim for a mean of 100 tests on each
    sample point - for example, if the test set is 10%, then do 1000
    iterations.
    
    est is an estimator. You might run cross validation and get the best one
    as a template.
    
    feat is the set of features to use.
    
    targ is the regression target.
    
    frame is a dataframe that includes all the features and the target.
    
    frac is proportion of the data frame to use as training data.
    
    ntries determines how many random splits to do. We aim to do this 
    many tests with a given point in the test set.
    
    Return mean and median error, the standard deviation, and the training
    set size. The return value is a dictionary containing this list for r2,
    mae and mse.
    """
    np.random.seed(1)
    nboot = int(ntries / (1.0 - frac))   
    r2list = []
    maelist = []
    mselist = []
    
    # f1=open('testfile', 'w')
    
    # Drop missing values - impute before calling if you want
    allcol = list(feat)
    allcol.append(targ)
    working = frame[allcol]
    working = working.dropna()
    
    for n in range(nboot):
        trainf, testf = split(working, frac)
        test = clone(est)
        test.fit(trainf[feat], trainf[targ])
        ytrue = testf[targ]
        ypred = test.predict(testf[feat])
        r2list.append(metrics.r2_score(ytrue, ypred))
        maelist.append(metrics.mean_absolute_error(ytrue, ypred))
        mselist.append(metrics.mean_squared_error(ytrue, ypred))
 
    trsize = int(len(working) * frac)
    result = {}
    # print >>f1, 'Shuffle feat', feat, len(working)
    # print >>f1, '  list', r2list
    result['r2'] = [np.mean(r2list), np.median(r2list), np.std(r2list), trsize]
    result['mae'] = [np.mean(maelist), np.median(maelist), np.std(maelist), trsize]
    result['mse'] = [np.mean(mselist), np.median(mselist), np.std(mselist), trsize]
  
    return result
       
# For debugging
if __name__ == "__main__":
    from sklearn import linear_model
    import datastack.dbs.rdb as rosetta
    
    rdb = rosetta.RosettaDBMongo(host="rosetta.hli.io")
    rdb.initialize(namespace='hg19')
    
    est = linear_model.Ridge()
    feat = ['facepheno.height', 'dynamic.FACE.pheno.v1.bmi']
    targ = 'facepheno.hand.strength.right.m1'
    frame = rdb.query(feat + [targ], with_nans=False)
    print shuffleRegress(est, feat, targ, frame)