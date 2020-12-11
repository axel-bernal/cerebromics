import cython
cimport cython
cimport numpy as np
import numpy as np
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


@cython.boundscheck(False) # turn of bounds-checking for entire function
def aggregate(np.ndarray[DTYPE_t, ndim=2] probs, np.ndarray[DTYPE_t, ndim=3]  X, np.ndarray[DTYPE_t, ndim=2]  Y):
	assert probs.dtype == DTYPE and X.dtype == DTYPE and Y.dtype == DTYPE
	cdef Py_ssize_t N = probs.shape[0]
	cdef Py_ssize_t M = probs.shape[1]
	cdef Py_ssize_t D = X.shape[2]
	cdef np.ndarray[DTYPE_t, ndim=2] X_marginal = np.zeros((N,D), dtype=DTYPE)
	cdef Py_ssize_t n,m,d

	for n in range(N):	
		X_marginal[n] = np.dot(probs[n], X[n])
	cdef np.ndarray[DTYPE_t, ndim=1] gradient = np.zeros((D), dtype=DTYPE)
	
	for n in range(M):	
		for m in range(N):
			for d in range(D):
					gradient[d] += (X_marginal[n,d] - X[n,m,d]) * Y[n,m]
	return gradient


