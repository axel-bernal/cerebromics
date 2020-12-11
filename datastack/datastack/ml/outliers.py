# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 16:38:45 2015

Copyright Human Longevity Inc. 2015. Authored by M. Cyrus Maher and Peter Garst
"""

import numpy as np
import pandas as pd


class Outliers (object):
    """
    This object supports outlier cleaning.
    It is integrated into preprocess to provide cleaning as data is read
    from the database, or it can be instantiated and called to provide cleaning
    on a data frame directly.
    Cyrus has lots of ideas for improving this.
    """

    def __init__ (self, logger = None, filter_info = None, stddevs = 6, **kwargs):
        """
        Instantiate the object with specified logging facilities.
        """
        self.logger = logger
        self.filter_info = filter_info
        self.outlier_stddevs = stddevs
        self.flagged = {}

    def __repr__(self):
        return '%s(outlier_stddevs=%i)' % (
            self.__class__.__name__,
            self.outlier_stddevs
        )

    def add_logger (self, logger):
        self.logger = logger

    def trim_vals(self, vec, stdevs=4, **kwargs):
        """
        Use robust mean and standard deviation to find outliers
        """

        if len(vec) > 10:  # arbitrary minimum number of observations
            # Get inner 95% of distribution
            quantiles = vec.quantile([.05, .95])
            dist_trimmed = vec[(vec >= quantiles.iloc[0]) & (vec <= quantiles.iloc[1])]
        else:
            dist_trimmed = vec

        # Calculate mean, upper and lower limits
        stdev_trim = dist_trimmed.std()
        upperlimit = dist_trimmed.mean() + stdev_trim * stdevs
        lowerlimit = dist_trimmed.mean() - stdev_trim * stdevs

        # Trim values
        over = (vec > upperlimit)
        under = (vec < lowerlimit)
        return over, under, upperlimit, lowerlimit

    def checkGroups (self, data, clcol, cols, trim = True):
        """
        Do outlier detection and trimming independently on the points
        belonging to each class.
        In place only for now.
        """
        plist = []
        classes = np.unique(data[clcol])
        for cl in classes:
            datac = data[data[clcol] == cl]
            self.check(datac, datac, cols, trim)
            plist.append(datac)

        return pd.concat(plist)

    def check (self, data, dataout, cols, trim = True):
        """
        Key routine for finding outliers.

        data is the input data and dataout is the trimmed data.
        I assume they are data frames. They may be the same.
        If we are just warning and not trimming, the output frame
        can be null.

        cols is the set of columns to trim.
        If trim is true, we trim the outliers; otherwise, we just warn.
        """
        for cc in cols:
            over, under, upperlimit, lowerlimit = self.trim_vals(data[cc], stdevs=self.outlier_stddevs)

            outliers = (over | under)
            n_outliers = outliers.sum()
            self.flagged[cc] = outliers[outliers].index

            if n_outliers > 0:
                if not self.logger is None:
                    self.logger.warning("%s outliers found for %s" % (n_outliers, cc))
                if trim:
                    dataout.loc[over.values, cc] = upperlimit
                    dataout.loc[under.values, cc] = lowerlimit
                    if not self.filter_info is None:
                        self.filter_info.ix["Outliers (trim)", cc] += n_outliers
                else:
                    if not self.filter_info is None:
                        self.filter_info.ix["Outliers (warn)", cc] += n_outliers
