
from distutils.core import setup, Extension

module1 = Extension('mrmr', 
                    include_dirs = ['/usr/local/include',
                    '/home/pgarst/tools/anaconda/pkgs/numpy-1.10.1-py27_0/lib/python2.7/site-packages/numpy/core/include'],
                    sources = ['mrmrmodule.cpp', 'mrdata.cpp'])

setup (name = 'MrMr',
       version = '1.0',
       description = 'This is a wrapper for mrmr feature selection.',
       ext_modules = [module1])
