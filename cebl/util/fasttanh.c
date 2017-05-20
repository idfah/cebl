#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

#include <math.h>

#define par_thresh 128

static PyObject *fastTanh(PyObject *self, PyObject *args)
{
    PyObject *input;
    PyArrayObject *xArr, *yArr;
    long n;
    int dtype;

    if (!PyArg_ParseTuple(args, "O", &input))
        return NULL;

    xArr = (PyArrayObject*)PyArray_FROM_OF(input, NPY_ARRAY_IN_ARRAY);
    if (xArr == NULL)
        return NULL;

    dtype = PyArray_TYPE(xArr);

    if (!PyTypeNum_ISFLOAT(dtype)) {
        Py_DECREF(xArr);
        xArr = (PyArrayObject*)PyArray_FROM_OTF(input, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
        dtype = NPY_DOUBLE;
    }

    yArr = (PyArrayObject*)PyArray_NewLikeArray(xArr, NPY_CORDER, NULL, 0);

    //for (i = 0; i < PyArray_NDIM(xArr); ++i)
    //    n *= PyArray_DIM(xArr, i);
    n = PyArray_SIZE(xArr);

    if (dtype == NPY_FLOAT32) {
        float *x, *y;

        x = (float*)PyArray_DATA(xArr);
        y = (float*)PyArray_DATA(yArr);

        if (n < par_thresh) {
        //if (1) {
            const float *yn = y + n;
            while (y < yn) {
                *y = tanhf(*x);
                y++; x++;
            }
        } else {
            #pragma omp parallel for
            for (long i = 0; i < n; ++i)
                y[i] = tanhf(x[i]);
        }
    }
    else if (dtype == NPY_FLOAT128) {
        long double *x, *y;

        x = (long double*)PyArray_DATA(xArr);
        y = (long double*)PyArray_DATA(yArr);

        if (n < par_thresh) {
        //if (1) {
            const long double *yn = y + n;
            while (y < yn) {
                *y = tanhl(*x);
                y++; x++;
            }
        } else {
            #pragma omp parallel for
            for (long i = 0; i < n; ++i)
                y[i] = tanhl(x[i]);
        }
    }
    else {
        double *x, *y;

        x = (double*)PyArray_DATA(xArr);
        y = (double*)PyArray_DATA(yArr);

        if (n < par_thresh) {
        //if (1) {
            const double *yn = y + n;
            while (y < yn) {
                *y = tanh(*x);
                y++; x++;
            }
        } else {
            #pragma omp parallel for
            for (long i = 0; i < n; ++i)
                y[i] = tanh(x[i]);
        }
    }

    Py_DECREF(xArr);

    return PyArray_Return(yArr);
}

static PyMethodDef fasttanh_methods[] = {
    {"fastTanh", fastTanh, METH_VARARGS, "Fast Hyperbolic Tangent"},
    {NULL, NULL, 0, NULL}
};

void initfasttanh()
{
    (void)Py_InitModule("fasttanh", fasttanh_methods);
    import_array();
}
