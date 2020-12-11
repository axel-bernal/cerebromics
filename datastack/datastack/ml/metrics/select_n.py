import numpy as np
import scipy.stats as st
from scipy.stats import rankdata
import scipy.stats as st


def get_ranks_ties(distances):
    [n,m] = distances.shape
    ranks_min = np.zeros(n)
    ranks_max = np.zeros(n)
    for i in xrange(n):
        d_ = distances[i]
        ranks_min[i] = rankdata(d_,method='min')[i]
        ranks_max[i] = rankdata(d_,method='max')[i]
    return ranks_min, ranks_max

def get_ranks(distances):
    [n,m] = distances.shape
    ranks = np.zeros(n)
    for i in xrange(n):
        d_ = distances[i]
        ranks[i] = rankdata(d_,method='ordinal')[i]
        #ranks_max[i] = rankdata(d_,method='max')[i]
    return ranks


def get_ranks_0based(distances):
    i_sort = distances.argsort(axis=1)
    id_ = np.arange(distances.shape[0])
    ranks = np.nonzero(i_sort == id_[:,np.newaxis])[1]
    return ranks


def compute_distances(t_y, p_y, distance="cosine"):
    t_y = np.array(t_y)
    p_y = np.array(p_y)

    # fast implementation of select@10:
    assert t_y.shape[0] == p_y.shape[0], "assuming t_y matrix, p_y matrices have same 1st dims"
    assert t_y.shape[1] == p_y.shape[1], "assuming t_y matrix, p_y matrices have same 2nd dims"
    p_y_ = p_y #- p_y.mean(1)[:,np.newaxis] # the cosine does not zero mean
    t_y_ = t_y #- t_y.mean(1)[:,np.newaxis] # the cosine does not zero mean
    p_variances = (p_y_*p_y_).sum(1)
    t_variances = (t_y_*t_y_).sum(1)
    prods = p_y.dot(t_y_.T)
    if distance == "cosine":

        distances = 1.0 - prods / np.sqrt(np.array(t_variances[np.newaxis,:] * p_variances[:,np.newaxis],np.dtype(np.float64)))
    elif distance == "Euclidean":
        distances = np.sqrt( np.array((t_variances[np.newaxis,:] + p_variances[:,np.newaxis]) - 2.0 * prods ,np.dtype(np.float64)))
    elif distance == "Manhattan":
        distances = np.zeros((t_y.shape[0],t_y.shape[0]))
        for i in xrange(t_y.shape[0]):
            distances[i] = np.absolute(p_y[i][np.newaxis,:] - t_y[:,:]).sum(1)
    else:
        raise NotImplementedError(distance)
    return distances


def select_at_n_dist(distances, n=10, d=0, replacement=False, average=True):
    """
    for the use case of ranking one prediction against n true people,
    distances is a squared matrix of distances from predicted (rows) and true (columns).
    The matches are on the diagonal.
    """
    ranks = get_ranks(distances)
    return select_at_n_ranks(ranks=ranks, n=n, d=d, replacement=replacement, average=average)


def select_at_n_ranks(ranks, n=10, d=0, replacement=False, average=True):
    select_ = np.zeros(ranks.shape[0])
    if replacement:
        for i in xrange(ranks.shape[0]):
            select_[i] = st.binom.cdf(d, n=n-1, p=(ranks[i]-1.0)/(ranks.shape[0]-1.0))# n draws, p probability of success. allow d samples to have smaller rank in lineup
    else:        
        for i in xrange(ranks.shape[0]):
            select_[i] = st.hypergeom.cdf(d, M=ranks.shape[0]-1, n=(ranks[i]-1), N=n-1)# N draws, M total number objects, n Type I objects
    if average:
        return select_.mean()
    else:
        return select_

def select_at_n_ranks_ties(ranks_min, ranks_max, n=10, d=0, replacement=False, average=True):
    select_ = np.zeros(ranks_min.shape[0])
    M = ranks_min.shape[0] - 1
    if replacement:
        for i in xrange(ranks_min.shape[0]):
            ranks = np.arange(ranks_min[i],ranks_max[i]+1)
            res = np.zeros_like(ranks)
            for ir, r in enumerate(ranks):
                res[ir] = st.binom.cdf(d, n=n-1, p=(r-1.0)/M)# n draws, p probability of success. allow d samples to have smaller rank in lineup
            select_[i] = res.mean()
    else:        
        for i in xrange(ranks_min.shape[0]):
            ranks = np.arange(ranks_min[i],ranks_max[i]+1)
            res = np.zeros_like(ranks)
            for ir, r in enumerate(ranks):
                res[ir] = st.hypergeom.cdf(d, M=M, n=r-1, N=n-1)# N draws, M total number objects, n Type I objects
            select_[i] = res.mean()
    if average:
        return select_.mean()
    else:
        return select_


def select_at_n(t_y, p_y, distance="cosine", n=10, d=0, replacement=False, average=True):
    t_y = np.array(t_y)
    p_y = np.array(p_y)

    # fast implementation of select@10:
    assert t_y.shape[0] == p_y.shape[0], "assuming t_y matrix, p_y matrices have same 1st dims"
    assert t_y.shape[1] == p_y.shape[1], "assuming t_y matrix, p_y matrices have same 2nd dims"

    distances = compute_distances(t_y, p_y, distance=distance)
    ranks = get_ranks(distances)
    return select_at_n_ranks(ranks=ranks, n=n, d=d, replacement=replacement, average=average)


def rank_all(t_y, p_y, distance="cosine"):
    distances=compute_distances(t_y, p_y, distance=distance)
    ranks = get_ranks_0based(distances)
    # orig wrong version:
    # select_10_fast = np.power(1.0-(ranks-1.0)/len(p_y), 9.0)
    # corrected version:
    select_10_fast = np.power(1.0-ranks/(len(p_y)-1.0), 9.0)
    return select_10_fast.mean()


def rank_all_wrong(t_y, p_y, distance="cosine"):
    """
    old version as implemented before.

    Yields wildly inflated results for small test sets.
    The corrected version of sampling without replacement is implemented in rank_all.
    """
    distances=compute_distances(t_y, p_y, distance=distance)
    ranks = get_ranks_0based(distances)
    # orig wrong version:
    select_10_fast = np.power(1.0-(ranks-1.0)/len(p_y), 9.0)
    # corrected version:
    # select_10_fast = np.power(1.0-ranks/(len(p_y)-1.0), 9.0)
    return select_10_fast.mean()

def similarities_from_probabilities(probs, transpose=True):
    """
    probs is a df with columns "pred_ind" and "obs_ind" and "Prob"

    Usage: 
    
    sims = select_n.similarities_from_probabilities(probs=probs, transpose=True)
    select_at_n = select_n.select_at_n_dist(distances=-sims, n=10, d=0, replacement=False, average=True)
    """
    inds = probs["pred_ind"].unique()
    assert (inds == probs["obs_ind"].unique()).all()
    sims = np.zeros((len(inds), len(inds)))
    for iobs, idx_pred in enumerate(inds):
        for ipred, idx_obs in enumerate(inds):
            sims[iobs, ipred] = probs["Prob"].values[iobs * len(inds) + ipred]
    if transpose:
        sims = sims.T
    return probs


if __name__ == '__main__':
    import datastack.ml.cross_validation as cross_validation

    n = 10  # select@n
    d = 1   # number of samples allowed to have smaller rank
    for N in xrange(20, 100, 10):
        X = np.random.randn(N,2)
        Y = X + np.random.randn(N,2)

        ere = rank_all_wrong(X,Y, distance="Euclidean")
        correct = rank_all(X,Y, distance="Euclidean")

        s_h = select_at_n(X,Y, n=n, d=d, distance="Euclidean")
        #s_h_ = cross_validation.rank_all(X,Y, n=n, distance="Euclidean")
        #assert s_h == s_h_, "missmatch"
        s_b = select_at_n(X,Y, n=n, d=d, distance="Euclidean", replacement=True)
        #s_b_ = cross_validation.rank_all(X,Y, n=n, distance="Euclidean", replacement=True)
        #assert s_b == s_b_, "missmatch"

        print "--------\nN = %i, select@%i, allow %i smaller ranks:" % (N,n,d)
        print "hypergeo: %.4e, binom: %.4e, correct: %.4e old_wrong_version: %.4e" % (s_h, s_b, correct, ere)
