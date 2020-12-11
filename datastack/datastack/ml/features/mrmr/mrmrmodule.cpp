/*
 * This is the interface file - we define the python module and the
 * methods we support in python.
 * It is a pretty stupid, basically c interface, but swig doesn't 
 * support 2D arrays, regardless of what it claims.
 *
 * Copy all data into c++ heap.
 */

#include "mrmr.h"

/*
 * Select some set of columns from the available features, and
 * return a list of the names.
 * Arguments: Mrmr object, data object, target dimensions.
 */
static PyObject *select(PyObject *self, PyObject *args) {
	PyObject *mrp, *dtp;
	int	ndim;
     
    if (!PyArg_ParseTuple(args, "OOi", &mrp, &dtp, &ndim))
        return NULL;

	MrMr		*mr	= (MrMr *) PyLong_AsVoidPtr(mrp);
	DataTable	*dt	= (DataTable *) PyLong_AsVoidPtr(dtp);

	int	totf;
	char	**cols	= mr->selectCols(dt, ndim, &totf);

	PyObject *pylist = PyList_New(totf);
	for (int i=0; i < totf; i++) {
		PyObject *item = Py_BuildValue("s", cols[i]);
		PyList_SET_ITEM(pylist, i, item);
	}

	return pylist;
}

/*
 * Move the data into a c++ array.
 * We will address it as d[sample][var], comes in as rowsXcolumns
 * Not optimized at all.
 *
 * The underlying code puts y in the first column, so we need to
 * allow for it here.
 */
static void initDataLong (DataTable *dtable, PyArrayObject *X) {

	dtable->nsample	= X->dimensions[0];
	dtable->nvar = X->dimensions[1] + 1;	// Provide for y column

	int	sstride = (int ) X->strides[0];
	int	vstride = (int ) X->strides[1];

	char	*base	= X->data;
	dtable->data = new float[dtable->nsample * dtable->nvar];
	float	*ptr = dtable->data;
	for (int samp = 0; samp < dtable->nsample; samp++) {
		ptr++;	// Placeholder for y value.
		for (int var = 0; var < dtable->nvar - 1; var++)  {
			*ptr++	= (float ) (* ((long *) (base + samp*sstride + var*vstride)));
		}
	}
}

static void initDataDouble (DataTable *dtable, PyArrayObject *X) {

	dtable->nsample	= X->dimensions[0];
	dtable->nvar = X->dimensions[1] + 1;	// Provide for y column

	int	sstride = (int ) X->strides[0];
	int	vstride = (int ) X->strides[1];

	char	*base	= X->data;
	dtable->data = new float[dtable->nsample * dtable->nvar];
	float	*ptr = dtable->data;
	for (int samp = 0; samp < dtable->nsample; samp++) {
		ptr++;	// Placeholder for y value.
		for (int var = 0; var < dtable->nvar - 1; var++)  {
			*ptr++	= (float ) (* ((double *) (base + samp*sstride + var*vstride)));
		}
	}
}

static void initYLong (DataTable *dtable, PyArrayObject *y) {

	int	sstride = (int ) y->strides[0];
	char	*base	= y->data;
	for (int i = 0; i < dtable->nsample; i++) {
		long val	= (* ((long *) (base + i*sstride)));
		int	pos		= i * dtable->nvar;
		dtable->data[pos]	= (float ) val;
	}
}

static void initYDouble (DataTable *dtable, PyArrayObject *y) {

	int	sstride = (int ) y->strides[0];
	char	*base	= y->data;
	for (int i = 0; i < dtable->nsample; i++) {
		double val	= (* ((double *) (base + i*sstride)));
		int	pos		= i * dtable->nvar;
		dtable->data[pos]	= (float ) val;
	}
}

static void dumpArr (PyArrayObject *X, char *tag) {

	printf("Array %s dimension %d:\n", tag, X->nd);
	printf("    size ");
	for (int i = 0; i < X->nd; i++)
		printf(" %d", (int ) X->dimensions[i]);
	printf("\n");

	printf("    strides ");
	for (int i = 0; i < X->nd; i++)
		printf(" %d", (int ) X->strides[i]);
	printf("\n");

	printf("    type %d\n", (int ) X->descr->type_num);
	fflush(stdout);
}

static char **readList (PyListObject *clist, int *lenp) {

	int	len	= PyList_Size((PyObject *) clist);
	char	**cols = (char **) malloc(len * sizeof(*cols));
	for (int i = 0; i < len; i++) {
		PyObject	*strp	= PyList_GetItem((PyObject *) clist, i);
		cols[i]	= stralloc(PyString_AsString(strp));
	}

	*lenp = len;
	return cols;
}

/*
 * Instantiate a DataTable object.
 * This is a container for a data frame, with some extra abilities for 
 * quantization and such.
 * Arguments are:
 *    a 2d np array of data 
 *    a 1d y array
 *    a list of all columns
 *    a list (possibly empty) of core columns we always want
 */
static PyObject *data (PyObject *self, PyObject *args) {
    PyArrayObject *X, *y;
	PyListObject *clist;
	PyListObject *core;

    if (!PyArg_ParseTuple(args, "O!O!O!O!", &PyArray_Type, &X, &PyArray_Type, &y, &PyList_Type, &clist, &PyList_Type, &core))
        return NULL;
	
	DataTable	*dtable	= new DataTable();

	// The first job is to get X and y arrays as real c++ arrays in the dtable object
	switch (X->descr->type_num) {
	case PyArray_LONG:
		initDataLong(dtable, X);
		break;
	case PyArray_DOUBLE:
		initDataDouble(dtable, X);
		break;
	default:
		printf("Python type not handled: %d\n", X->descr->type_num);
		fflush(stdout);
	}

	switch (y->descr->type_num) {
	case PyArray_LONG:
		initYLong(dtable, y);
		break;
	case PyArray_DOUBLE:
		initYDouble(dtable, y);
		break;
	default:
		printf("Python type not handled: %d\n", y->descr->type_num);
		fflush(stdout);
	}

	// dumpArr(X, (char *) "X");
	// dumpArr(y, (char *) "y");

	// Now we get the list of column names for the X data.
	int len0, len1;
	char **cols = readList(clist, &len0);
	char **corecol = readList(core, &len1);
	dtable->setColumns(cols, len0);
	dtable->setCore(corecol, len1);
	dtable->buildData2d();

	// dtable->dumpTab();

	return PyLong_FromVoidPtr((void *) dtable);
}

/*
 * Instantiate an MrMr object.
 * At this point it knows nothing.
 * No arguments.
 */
static PyObject *newmr (PyObject *self, PyObject *args) {
	MrMr *mr	= new MrMr();
	return PyLong_FromVoidPtr((void *) mr);
}

static PyMethodDef MrmrMethods[] = {
    {"newmr",  newmr, METH_VARARGS, "Instantiate an MrMr object."},
    {"data",  data, METH_VARARGS, "Instantiate a data table."},
    {"select",  select, METH_VARARGS, "Run feature selection."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

// extern "C" PyMODINIT_FUNC init_mrmrlib(void)
// extern "C" void init_mrmrlib(void)
PyMODINIT_FUNC init_mrmrlib(void)
{
    (void) Py_InitModule("_mrmrlib", MrmrMethods);
	import_array();
}
