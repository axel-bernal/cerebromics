#ifndef _MRMR_H_
#define _MRMR_H_

#include <Python.h>
#include <numpy/arrayobject.h>

extern char *stralloc (char *str);

// Tests

void testf (long *values, int rows, int cols);
void testg (int val);

#include "DataTable.h"

class RelInfo {
public:
	RelInfo (float *mival, int *index, DataTable *data, int nfeat);
	int getNfeat ();
	int getNvar ();
	int *getIndex ();
	DataTable *getData ();
	float *mival;

private:
	DataTable	*data;
	int			*index;
	int			nfeat;
};

class VarChoice {
public:
	VarChoice (RelInfo *rel, bool domid, FILE *db);
	char **selectCols (int ndim);
	void dumpCols (int ndim);

private:
	long	*midInd;
	float	*score;
	RelInfo	*relinf;
};

/*
 * This class encapsulates the algorithm.
 * Everything else is io, error checking, etc.
 */
class MrMr {
public:
	MrMr ();
	~MrMr ();

	int getNfeat ();
	char **selectCols (DataTable *dtable, int ndim, int *nret);

	RelInfo *getRel ();
	VarChoice *getMid ();
	VarChoice *getMiq ();

private:
	PyArrayObject *array;
	void tryMid ();
	void tryMiq ();
	int *convInt (float *arr, int len);

	RelInfo	*relChoice;
	VarChoice	*midChoice;
	VarChoice	*miqChoice;

	int nfeatures;
};

#endif
