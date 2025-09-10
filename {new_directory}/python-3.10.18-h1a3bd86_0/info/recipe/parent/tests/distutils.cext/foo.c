#include <Python.h>

static PyObject *
greet_name(PyObject *self, PyObject *args)
{
    const char *name;

    if (!PyArg_ParseTuple(args, "s", &name))
    {
        return NULL;
    }
    
    printf("Hello %s!\n", name);

    Py_RETURN_NONE;
}

static PyMethodDef GreetMethods[] = {
    {"greet", greet_name, METH_VARARGS, "Greet an entity."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef greet =
{
    PyModuleDef_HEAD_INIT,
    "greet",     /* name of module */
    "",          /* module documentation, may be NULL */
    -1,          /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    GreetMethods
};

PyMODINIT_FUNC PyInit_greet(void)
{
    return PyModule_Create(&greet);
}

