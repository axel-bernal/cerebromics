#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "DataTable.h"

/*
 * This is roughly a container for a python data frame, with some additional
 * data manipulation facilities.
 */

DataTable::DataTable() {

	data = NULL;
	data2d = NULL;
	nsample = 0;
	nvar = 0;
	colnames = NULL;
	corecol = NULL;
	corelen = 0;
	b_zscore = 0;    // initialze the data as not being z-scored
	b_discetize = 0; // initialze the data as continous
    }
	
DataTable::~DataTable() {

	if (data) {
		delete[] data;
		data = NULL;
	}
	if (data2d) {
		destroyData2d();
	}
	nsample = nvar = 0;
	return;
    }

void DataTable::dumpTab () {

	printf("Dump data table:\n");
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			printf(" %g", data2d[i][j]);
		}
		printf("\n");
	}
}
	
int DataTable::getIndex (char *col) {

	for (int i = 0; i < nvar; i++) {
		if (!strcmp(col, colnames[i]))
			return i;
	}

	return -1;
}

int DataTable::buildData2d() {
	if (!data) {
		fprintf(stderr, "The data is not ready yet: data point is NULL");
	}

	if (data2d)
		destroyData2d();
	if (nsample <= 0 || nvar <= 0) {
		fprintf(stderr, "The data is not ready yet: nsample=%ld nvar=%ld", nsample, nvar);
		return 0;
	}
	data2d = new float* [nsample];
	if (!data2d) {
		fprintf(stderr, "Line %d: Fail to allocate memory.\n", __LINE__);
		return 0;
	} else {
		for (long i = 0; i < nsample; i++) {
			data2d[i] = data + i * nvar;
		}
	}
	return 1;
}
	
void DataTable::destroyData2d() {
	if (data2d) {
		delete[] data2d;
		data2d = NULL;
	}
}
		
void DataTable::setColumns (char **cols, int len) {

	colnames	= cols;
}
		
void DataTable::setCore (char **cols, int len) {

	corecol	= cols;
	corelen = len;
}

char **DataTable::getColumns () {
	return colnames;
}

char **DataTable::getCoreColumns () {
	return corecol;
}

int DataTable::getCoreLen () {
	return corelen;
}

void DataTable::zscore(long indExcludeColumn, int b_discretize) {
	if (!data2d)
		buildData2d();

	if (!b_discretize)
		return; // in this case, just generate the 2D data array

	for (long j = 0; j < nvar; j++) {
		if (j == indExcludeColumn) {
			continue; // this is useful to exclude the first column, which will be the target classification
					  // variable
		}
		double cursum = 0;
		double curmean = 0;
		double curstd = 0;
		for (long i = 0; i < nsample; i++)
			cursum += data2d[i][j];
		curmean = cursum / nsample;
		cursum = 0;
		register double tmpf;
		for (long i = 0; i < nsample; i++) {
			tmpf = data2d[i][j] - curmean;
			cursum += tmpf * tmpf;
		}
		curstd = (nsample == 1) ? 0 : sqrt(cursum / (nsample - 1)); // nsample -1 is an unbiased version for
																	// Gaussian
		for (long i = 0; i < nsample; i++) {
			data2d[i][j] = (data2d[i][j] - curmean) / curstd;
		}
	}
	b_zscore = 1;
    }

void DataTable::discretize(double threshold, int b_discretize) {
	long indExcludeColumn = 0; // exclude the first column
	if (b_zscore == 0) {
		zscore(indExcludeColumn, b_discretize); // exclude the first column
	}

	if (!b_discretize)
		return; // in this case, just generate the 2D array

	for (long j = 0; j < nvar; j++) {
		if (j == indExcludeColumn)
			continue;
		register double tmpf;
		for (long i = 0; i < nsample; i++) {
			tmpf = data2d[i][j];
			if (tmpf > threshold)
				tmpf = 1;
			else {
				if (tmpf < -threshold)
					tmpf = -1;
				else
					tmpf = 0;
			}
			data2d[i][j] = tmpf;
		}
	}
	b_discetize = 1;
    }
