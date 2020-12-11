from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import re
import os
from itertools import cycle



f = open("/home/ec2-user/compute/Data/HLI/plink-OC-NAFLD/OC-Loomba.sorted.fam",'r')
lastFam=None
j=0
for i,l in enumerate(f):
    parts = l.split();
    if (i==0):
        lastFam=parts[0];
        print(l,end="")
        j+=1
    else:
        if (lastFam == parts[0]):
            continue # go to the next line
        else:
            print(l,end="")
            lastFam=parts[0]
            j+=1

