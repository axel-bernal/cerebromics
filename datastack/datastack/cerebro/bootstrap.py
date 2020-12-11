'''
Functions to perform bootstrap aggregation on classifier/regression
predictions compared to some baseline predictions.

Created on Nov 12, 2015

@author: och, twong
'''

import numpy as np
import random
import pandas as pd


def bootstrap(baseline_scores, test_scores, iterations=1000,
              smaller_scores_better=True):
    """Bootstrap aggregate a set of test scores corresponding to
    classification/regression predictions against a set of baseline
    scores to see if the test performs better than the baseline. Score
    values should be some for of numeric value; examples include {0,1}
    values for correctly/incorrectly predicted, or MSE values per
    prediction.

    Args:
        baseline_scores: An iterable collection of baseline scores.
        test_scores: An iteratble collection of test scores.
        iterations: The number of bootstrap experiments to try; by
            default, 1000.
        smaller_scores_better: A flat to indicate if smaller score values
            are better; by default, `True`

    Returns:
        A tuple of the star rating, the fraction of experiments that
        showed the test was better than the baseline, and of the fraction
        that showed that the test was worse.
    """
    if len(test_scores) != len(baseline_scores):
        raise IndexError('Test and baseline score vectors have different lengths: test %d baseline %d' %
                         (len(test_scores), len(baseline_scores)))
    if iterations <= 0:
        raise ValueError('Iteration count must be greater than 0')
    delta_scores = test_scores - baseline_scores
    test_better = 0
    test_worse = 0
    for _ in range(0, iterations):
        aggregate = 0
        for _ in range(0, len(test_scores)):
            aggregate += delta_scores[random.randrange(len(test_scores))]
        if ((smaller_scores_better is True and aggregate < 0)
                or (smaller_scores_better is False and aggregate > 0)):
            test_better += 1
        elif ((smaller_scores_better is True and aggregate > 0)
                or (smaller_scores_better is False and aggregate < 0)):
            test_worse += 1
        else:
            pass
    stars = 0
    if test_better > 990:
        stars = '**'
    elif test_worse > 990:
        stars = '!!'
    elif test_better > 950:
        stars = '*'
    elif test_worse > 950:
        stars = '!'
    else:
        stars = ''
    return stars, float(test_better) / iterations, float(test_worse) / iterations

class Bootstrap:
    """
    Currently this is supported for one dimensional output, not for the 
    multivariate case. This is just a kludge to get the system released in a state
    which doesn't crash and doesn't lose any functionality.
    We should add the multivariate case back in.
    """
    
    def __init__(self, observed, baseline, test, scorer=None, smaller_scores_better=True):
        self.observed = observed
        self.valid = True
        self.baseline = self.makeOned(baseline)
        self.test = self.makeOned(test)
        if self.valid and (scorer is not None):
            self.rescore(scorer, smaller_scores_better=smaller_scores_better)
            
    def makeOned (self, val):
        """
        Currently only support one dimension.
        Change this for multivariate.
        """
        if isinstance(val, pd.DataFrame):
            sh = val.shape
            if sh[1] > 1:
                self.valid = False
                return None
            val = pd.Series(val.values[:,0])
        return val

    def rescore(self, scorer, smaller_scores_better=True):
        if not self.valid:
            return
        self.scorer = scorer
        self.smaller_scores_better = smaller_scores_better
        self.baseline_scores = np.array([scorer(pair[0], pair[1]) for pair in zip(self.observed, self.baseline)])
        self.test_scores = np.array([scorer(pair[0], pair[1]) for pair in zip(self.observed, self.test)])

    def bootstrap(self, iterations=1000):
        if not self.valid:
            return '', 0.5, 0.5
        return bootstrap(self.baseline_scores, self.test_scores, iterations=iterations, smaller_scores_better=self.smaller_scores_better)
