# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 14:05:39 2016

Copyright (C) 2016 Human Longevity Inc. Written by Peter Garst.

A utility to keep x and y, test and training organized.
They can be preprocessed or not. 
"""

class DataSet (object):
    
    def __init__ (self, xtr, xtst, ytr, ytst):
        self.X_training = xtr
        self.X_test = xtst
        self.y_training = ytr
        self.y_test = ytst