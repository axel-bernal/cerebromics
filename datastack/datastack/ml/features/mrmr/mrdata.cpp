#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "DataTable.h"
#include "sort2.h"
#include "mrmr.h"

/*
 * First some general service routines.
 */

char *stralloc (char *str) {

	char	*nstr = (char *) malloc(strlen(str) + 1);
	strcpy(nstr, str);
	return nstr;
}

static double compute_mutualinfo (double* pab, long pabhei, long pabwid, bool debug)
{
    // check if parameters are correct
    if (!pab)
		return -1;

    double** pab2d = new double* [pabwid];
    for (long j = 0; j < pabwid; j++)
		pab2d[j] = pab + j * pabhei;

	if (debug) {
		for (long j = 0; j < pabwid; j++) {
			for (long i = 0; i < pabhei; i++) // should for p1
				printf("JP %2d %2d: %g\n", j, i, pab2d[j][i]);
		}
	}

    double* p1 = 0, *p2 = 0;
    long p1len = 0, p2len = 0;
    int b_findmarginalprob = 1;

    // generate marginal probability arrays
    if (b_findmarginalprob != 0) {
		p1len = pabhei;
		p2len = pabwid;
		p1 = new double[p1len];
		p2 = new double[p2len];

		for (long i = 0; i < p1len; i++)
			p1[i] = 0;
		for (long j = 0; j < p2len; j++)
			p2[j] = 0;

		for (long i = 0; i < p1len; i++)     // p1len = pabhei
			for (long j = 0; j < p2len; j++) // p2len = panwid
			{
			p1[i] += pab2d[j][i];
			p2[j] += pab2d[j][i];
			}
    }

    // calculate the mutual information
    double muInf = 0.0;
    for (long j = 0; j < pabwid; j++) // should for p2
    {
		for (long i = 0; i < pabhei; i++) // should for p1
		{
			if (pab2d[j][i] != 0 && p1[i] != 0 && p2[j] != 0) 
				muInf += pab2d[j][i] * log(pab2d[j][i] / p1[i] / p2[j]);
		}
    }

    muInf /= log(2);

    // free memory
    if (pab2d) {
		delete[] pab2d;
    }
    if (b_findmarginalprob != 0) {
		if (p1) {
			delete[] p1;
		}
		if (p2) {
			delete[] p2;
		}
    }

    return muInf;
}

static void copyvecdata (int* srcdata, long len, int* desdata, int& nstate) {
    if (!srcdata || !desdata)
		return;

    // note: originally I added 0.5 before rounding, however seems the negative numbers and
    //      positive numbers are all rounded towarded 0; hence int(-1+0.5)=0 and int(1+0.5)=1;
    //      This is unwanted because I need the above to be -1 and 1.
    // for this reason I just round with 0.5 adjustment for positive and negative differently

    // copy data
    int minn, maxx;
    if (srcdata[0] > 0)
		maxx = minn = int(srcdata[0] + 0.5);
    else
		maxx = minn = int(srcdata[0] - 0.5);

    int tmp;
    double tmp1;
    for (long i = 0; i < len; i++) {
		tmp1 = double(srcdata[i]);
		tmp = (tmp1 > 0) ? (int)(tmp1 + 0.5) : (int)(tmp1 - 0.5); // round to integers
		minn = (minn < tmp) ? minn : tmp;
		maxx = (maxx > tmp) ? maxx : tmp;
		desdata[i] = tmp;
    }

    // make the vector data begin from 0 (i.e. 1st state)
    for (long i = 0; i < len; i++) {
		desdata[i] -= minn;
    }

    // return the #state
    nstate = (maxx - minn + 1);
    return;
}

static double* compute_jointprob(int* img1, int* img2, long len, long maxstatenum, int& nstate1, int& nstate2, bool debug) {

    // get and check size information
    if (!img1 || !img2 || len < 0)
		return NULL;

    // int b_findstatenum = 1;
    //  int nstate1 = 0, nstate2 = 0;

    int b_returnprob = 1;

    // copy data into new INT type array (hence quantization) and then reange them begin from 0 (i.e. state1)
    int* vec1 = new int[len];
    int* vec2 = new int[len];

    if (!vec1 || !vec2)
		return NULL;

    int nrealstate1 = 0, nrealstate2 = 0;

    copyvecdata(img1, len, vec1, nrealstate1);
    copyvecdata(img2, len, vec2, nrealstate2);

	if (debug) 
		printf("joint: states %d %d\n", nrealstate1, nrealstate2);

    // update the #state when necessary
    nstate1 = (nstate1 < nrealstate1) ? nrealstate1 : nstate1;
    // printf("First vector #state = %i\n",nrealstate1);
    nstate2 = (nstate2 < nrealstate2) ? nrealstate2 : nstate2;
    // printf("Second vector #state = %i\n",nrealstate2);

    // generate the joint-distribution table
    double* hab = new double[nstate1 * nstate2];
    double** hab2d = new double* [nstate2];

    if (!hab || !hab2d)
		return NULL;

    for (long j = 0; j < nstate2; j++)
		hab2d[j] = hab + (long)j * nstate1;

    for (long i = 0; i < nstate1; i++)
	for (long j = 0; j < nstate2; j++)
	    hab2d[j][i] = 0;

    for (long i = 0; i < len; i++) {
		// old method -- slow
		//     indx = (long)(vec2[i]) * nstate1 + vec1[i];
		//     hab[indx] += 1;

		// new method -- fast
		hab2d[vec2[i]][vec1[i]] += 1;
    }

    // return the probabilities, otherwise return count numbers
    if (b_returnprob) {
		for (long i = 0; i < nstate1; i++)
			for (long j = 0; j < nstate2; j++) {
				hab2d[j][i] /= len;
			}
    }

    // finish
    if (hab2d) {
		delete[] hab2d;
		hab2d = 0;
    }
    if (vec1) {
		delete[] vec1;
		vec1 = 0;
    }
    if (vec2) {
		delete[] vec2;
		vec2 = 0;
    }

    return hab;
}

/**
 * Return -1 on error.
 */
static double calMutualInfo(DataTable* myData, long v1, long v2, bool debug) {

    double mi = -1; // initialized as an illegal value

    if (!myData || !myData->data || !myData->data2d)
		return mi;

    long nsample = myData->nsample;
    long nvar = myData->nvar;

    if (v1 >= nvar || v2 >= nvar || v1 < 0 || v2 < 0)
		return mi;

    // copy data
    int* v1data = new int[nsample];
    int* v2data = new int[nsample];

    if (!v1data || !v2data)
		return mi;

    for (long i = 0; i < nsample; i++) {
		v1data[i] = int(myData->data2d[i][v1]); // the float already been discretized, should be safe now
		v2data[i] = int(myData->data2d[i][v2]);
    }

    // compute mutual info
    long nstate = 3; // always true for DataTable, which was discretized as three states

    int nstate1 = 0, nstate2 = 0;
    double *pab = compute_jointprob(v1data, v2data, nsample, nstate, nstate1, nstate2, debug);
	if (pab == NULL)
		return mi;

	if (debug) {
		char **cols = myData->getColumns();
		printf("DEBUG %s %s\n", cols[v1], cols[v2]);
	}
    mi = compute_mutualinfo(pab, nstate1, nstate2, debug);

    // free memory and return
    if (v1data) {
		delete[] v1data;
		v1data = 0;
    }
    if (v2data) {
		delete[] v2data;
		v2data = 0;
    }
    if (pab) {
		delete[] pab;
		pab = 0;
    }

    return mi;
}

RelInfo::RelInfo (float *mival, int *index, DataTable *data, int nfeat) {

	this->data	= data;
	this->mival	= mival;
	this->index	= index;
	this->nfeat	= nfeat;
}

DataTable *RelInfo::getData () {

	return data;
}

int RelInfo::getNfeat () {

	return nfeat;
}

int *RelInfo::getIndex () {

	return index;
}

int RelInfo::getNvar () {

	return data->nvar;
}

VarChoice::VarChoice (RelInfo *rel, bool domid, FILE *db) {

	int	nfea	= rel->getNfeat();
	relinf	= rel;

	int	clen = rel->getData()->getCoreLen();
    midInd = new long[nfea + clen];
    score = new float[nfea + clen];

    // mRMR selection
    long poolFeaIndMin = 1;

	// For efficiency we only test covariates that have pretty good MI with
	// the target - check the number we want, plus a margin.
    long poolFeaIndMax = 500 + nfea;
	if (poolFeaIndMax >= rel->getNvar())
		poolFeaIndMax	= rel->getNvar() - 1;

	int	*poolInd	= rel->getIndex();

	// Mask showing still available variables.
	bool	*poolIndMask	= new bool[rel->getNvar()];
	for (int i = 0; i < rel->getNvar(); i++)
		poolIndMask[i]	= true;

	int used = 1;
	int lim	= nfea;
	if (clen == 0) {
		midInd[0] = poolInd[1];
		score[0] = rel->mival[midInd[0]];
		poolIndMask[midInd[0]] = false; // after selection, no longer consider this feature
		poolIndMask[0] = 0;
	} else {
		for (int k = 0; k < clen; k++) {
			char	*col = rel->getData()->getCoreColumns()[k];
			int	selectind	= rel->getData()->getIndex(col);
			midInd[k] = selectind;
			score[k] = rel->mival[selectind];
			poolIndMask[selectind] = false;
		}
		lim	= nfea + clen;
		used = clen;
	}

	// the first one, midInd[0] has been determined already
    for (long k = used; k < lim; k++) {
		double relevanceVal, redundancyVal, tmpscore, selectscore;
		long selectind;
		int b_firstSelected = 0;

		for (long i = poolFeaIndMin; i <= poolFeaIndMax; i++) {
			if (!poolIndMask[poolInd[i]])
				continue; // skip this feature as it was selected already

			relevanceVal = calMutualInfo(rel->getData(), 0, poolInd[i], false); // actually no necessary to re-compute it, this
																	   // value can be retrieved from mival vector
			redundancyVal = 0;
			for (long j = 0; j < k; j++)
				redundancyVal += calMutualInfo(rel->getData(), midInd[j], poolInd[i], false);
			redundancyVal /= k;
			if (db != NULL)
				fprintf(db, "  Try %d: mi %g  redun %g\n", poolInd[i], relevanceVal, redundancyVal);

			if (domid) 
				tmpscore = relevanceVal - redundancyVal;
			else
				tmpscore = relevanceVal / (redundancyVal + 0.0001);

			if (b_firstSelected == 0) {
				selectscore = tmpscore;
				selectind = poolInd[i];
				b_firstSelected = 1;
			} else {
				if (tmpscore > selectscore) { // update the best feature found and the score
					selectscore = tmpscore;
					selectind = poolInd[i];
				}
			}
		}

		if (db != NULL)
			fprintf(db, "SELECT %d score %g\n", selectind, selectscore);
		midInd[k] = selectind;
		score[k] = selectscore;
		poolIndMask[selectind] = false;
    }

    if (poolIndMask != NULL)
		delete[] poolIndMask;
	/*
	for (int i = 0; i < clen + 1; i++) {
		printf("   %2d.  %d ", i, midInd[i]); 
		fflush(stdout);
		printf("%s\n", relinf->getData()->getColumns()[midInd[i]]); 
		fflush(stdout);
	}
	*/
}

void VarChoice::dumpCols (int ndim) {

	char **cols	= relinf->getData()->getColumns();
	int	*indv	= relinf->getIndex();

	printf("MI values:\n");
	for (int i = 0; i < ndim; i++) {
		int	ind	= indv[i];
		printf("  %s %g\n", cols[ind], relinf->mival[ind]);
	}

	printf("Column selection:\n");
	for (int i = 0; i < ndim; i++) {
		printf("  %s %g\n", cols[midInd[i]], score[i]);
	}
}

char **VarChoice::selectCols (int ndim) {

	// dumpCols(ndim);
	char **result = (char **) malloc(ndim * sizeof(*result));
	for (int i = 0; i < ndim; i++) {
		result[i]	= relinf->getData()->getColumns()[midInd[i]];
	}

	return result;
}

MrMr::~MrMr () {
	
	if (relChoice != NULL)
		delete relChoice;
	if (midChoice != NULL)
		delete midChoice;
	if (miqChoice != NULL)
		delete miqChoice;
}

MrMr::MrMr () {
}

static void dumpv (FILE *f, float *mi, int nv, int *index, char **cols) {

	for (int i = 0; i < nv; i++) {
		char *cname = (char *) "";
		int	ind	= i;
		if ((i > 0) && (index != NULL)) {
			ind	= index[i];
		}
		if (i > 0)
			cname	= cols[ind];

		fprintf(f, "  %2d. %g  %s\n", i, mi[i], cname);
	}
}

char **MrMr::selectCols (DataTable *dtable, int ndim, int *nret) {

    // long poolUseFeaLen = myData->nvar - 1; //500;
    int maxf = 500;
    if (maxf > dtable->nvar - 1) // there is a target variable (the first one), that is why must remove one
		maxf = dtable->nvar - 1;

	FILE *dbf = NULL; // fopen("debug.txt", "w");

    if (ndim > maxf)
		ndim = maxf;
	
	nfeatures = ndim;
	int	totf = ndim + dtable->getCoreLen();
	if (totf > (dtable->nvar - 1))
		// Could just return all columns now
		totf	= dtable->nvar - 1;

    // determine the pool.
	// The poolInd array is float so that we can pass it to sort2.
    float* mival = new float[dtable->nvar];
    float* poolInd = new float[dtable->nvar];
    if (!mival || !poolInd) 
		return NULL;

	// the mival[0] is the entropy of target classification variable
	bool debug = false;
    for (long i = 0; i < dtable->nvar; i++) {
		mival[i] = -calMutualInfo(dtable, 0, i, debug); // set as negative for sorting purpose
		poolInd[i] = i;
		debug	= false;
    }

    float* NR_mival = mival;     // vector_phc(1,myData->nvar-1);
    float* NR_poolInd = poolInd; // vector_phc(1,myData->nvar-1);

	// This gives us the list of x columns, not y
	char **cols = dtable->getColumns();
	// fprintf(dbf, "Initial MI values:\n");
	// dumpv(dbf, mival, dtable->nvar, NULL, cols);

	// Sort2 is 1-based - comes from ancient numerical recipes fortran code 
    sort2(dtable->nvar, &NR_mival[-1], &NR_poolInd[-1]); // note that poolIndMask is not needed to be sorted, as everything in
                                                   // it is 1 up to this point

    for (long i = 0; i < dtable->nvar; i++)
		mival[i] = -mival[i];
	int	*relindex	= convInt(NR_poolInd, dtable->nvar);

	// fprintf(dbf, "Sorted MI values:\n");
	// dumpv(dbf, mival, dtable->nvar, relindex, cols);
	// fflush(dbf);

	relChoice	= new RelInfo(mival, relindex, dtable, ndim);

	midChoice	= new VarChoice(relChoice, true, NULL);
	miqChoice	= new VarChoice(relChoice, false, dbf);
	fclose(dbf);

	// Next pass we'll configure one of mid, miq
	*nret	= totf;
	return miqChoice->selectCols(totf);
}

int MrMr::getNfeat () {

	return nfeatures;
}

int *MrMr::convInt (float *arr, int len) {

	int *narr	= new int[len];
	for (int i = 0; i < len; i++)
		narr[i]	= int(arr[i]);
	
	return narr;
}

RelInfo *MrMr::getRel () {

	return relChoice;
}

VarChoice *MrMr::getMid () {

	return midChoice;
}

VarChoice *MrMr::getMiq () {

	return miqChoice;
}
