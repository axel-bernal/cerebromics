# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 23:16:04 2016

Copyright (C) 2016 Human Longevity Inc. Written by Peter Garst.
"""

import pytest
import pandas as pd
import os

@pytest.fixture(scope="session")
def frame1(request):
    path = os.path.dirname(os.path.realpath(__file__)) + '/resources/frame1.csv'
    df = pd.read_csv(path)
    return df
    
@pytest.fixture(scope="session")
def multi1(request):
    path = os.path.dirname(os.path.realpath(__file__)) + '/resources/multi1.csv'
    df = pd.read_csv(path)
    return df
        
@pytest.fixture(scope="session")
def skin1(request):
    path = os.path.dirname(os.path.realpath(__file__)) + '/resources/skin.csv'
    df = pd.read_csv(path)
    return df
              
@pytest.fixture(scope="session")
def age(request):
    path = os.path.dirname(os.path.realpath(__file__)) + '/resources/age.csv'
    df = pd.read_csv(path)
    return df
           
@pytest.fixture(scope="session")
def height(request):
    path = os.path.dirname(os.path.realpath(__file__)) + '/resources/height.csv'
    df = pd.read_csv(path)
    return df