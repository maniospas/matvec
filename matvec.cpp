//#define DLL_EXPORT __declspec(dllimport) extern "C"
#define sizetype long long
#define valuetype double
#include <iostream>
#include <math.h>
#include <omp.h>
#include <vector>
#include "Python.h"
#include <map>
#include<unordered_map>

#if defined(_MSC_VER)
    //  Microsoft
    #define EXPORT __declspec(dllexport)
#elif defined(__GNUC__)
    //  GCC
    #define EXPORT __attribute__((visibility("default")))
    #define IMPORT
#else
    //  do nothing and hope for the best?
    #define EXPORT
    #define IMPORT
    #pragma warning Unknown dynamic link import/export semantics.
#endif

// TODO: integrate https://github.com/taoito/matvec-mpi

std::unordered_map<sizetype, std::vector<valuetype*>*> reusable;

valuetype* allocate_values(sizetype size) {
    valuetype* pending_values;
    if(reusable.find(size)!=reusable.end() && reusable[size]->size()>0) {
        pending_values = reusable[size]->back();
        reusable[size]->pop_back();
    }
    else
        pending_values = new valuetype[size];
    return pending_values;
}

class Vector {
    public:
        Py_buffer* release_buffer;
        valuetype* values;
        sizetype size;

        Vector(valuetype* _values, sizetype _size, bool copy=true,  Py_buffer* _release_buffer=NULL) {
            if(copy) {
                valuetype* pending_values = allocate_values(_size);
                sizetype i;
                #pragma omp parallel for shared(_size, pending_values, _values) private(i)
                for(i=0;i<_size;i++)
                    pending_values[i] = _values[i];
                values = pending_values;
            }
            else
                values = _values;
            release_buffer = _release_buffer;
            size = _size;
        }
        ~Vector() {
            if(release_buffer!=NULL) {
                PyBuffer_Release(release_buffer);
                delete release_buffer;
            }
            else {
                if(reusable.find(size)==reusable.end())
                    reusable[size] = new std::vector<valuetype*>();
                reusable[size]->push_back(values);
                //delete values;
            }
        }
};


class Matrix {
    public:
        sizetype* x;
        sizetype* y;
        valuetype* values;
        sizetype entries;
        sizetype size;
        //std::unordered_map<sizetype, std::vector<sizetype>*> row_indexes;
        //std::unordered_map<sizetype, std::vector<sizetype>*> col_indexes;

        Matrix(sizetype* _x, sizetype* _y, valuetype* _values, sizetype _entries, sizetype _size, bool copy=true) {
            if(copy) {
                sizetype* px = new sizetype[_entries];
                sizetype* py = new sizetype[_entries];
                valuetype* pv = new valuetype[_entries];
                sizetype i;
                #pragma omp parallel for shared(_entries, px, py, pv, _x, _y, _values) private(i)
                for(i=0;i<_entries;i++) {
                    px[i] = _x[i];
                    py[i] = _y[i];
                    pv[i] = _values[i];
                }
                x = px;
                y = py;
                values = pv;
            }
            else {
                x = _x;
                y = _y;
                values = _values;
            }
            entries = _entries;
            size = _size;
            /*sizetype i;
            //#pragma omp parallel for shared(_entries, px, py, pv, _x, _y, _values) private(i)
            for(i=0;i<_entries;i++) {
                if(row_indexes.find(_x[i]) != row_indexes.end())
                    row_indexes[_x[i]] = new std::vector<sizetype>();
                row_indexes[_x[i]]->push_back(i);
            }*/
        }
        ~Matrix() {
            delete x;
            delete y;
            delete values;
        }
};

extern "C" EXPORT void* multiply(void* _matrix, void* _vector) {
    Matrix* matrix = (Matrix*)_matrix;
    Vector* vector = (Vector*)_vector;
    sizetype size = vector->size;
    sizetype* x = matrix->x;
    sizetype* y = matrix->y;
    valuetype* mv = matrix->values;
    valuetype* vv = vector->values;
    valuetype* ret = allocate_values(size);//new valuetype[size]();
    sizetype i;
    #pragma omp parallel for shared(size, ret) private(i)
    for(i=0;i<size;i++)
        ret[i] = 0;
    sizetype entries = matrix->entries;
    #pragma omp parallel for shared(entries, ret, x, y, mv, vv) private(i)
    for(i=0;i<entries;i++) {
        valuetype val = mv[i]*vv[y[i]];
        #pragma omp atomic
        ret[x[i]] += val;
    }
    /*
    sizetype row;
    #pragma omp parallel for shared(entries, ret, x, y, mv, vv, matrix) private(row)
    for(row=0;row<size;row++) {
        std::vector<sizetype>* adjacent = matrix->row_indexes[row];
        std::cout<<row<" row\n";
        for(int j=0;j<adjacent->size();i++) {
            i = adjacent->at(j);
            ret[row] += mv[i]*vv[y[i]];
        }
    }*/

    return new Vector(ret, size, false);
}


extern "C" EXPORT void* rmultiply(void* _matrix, void* _vector) {
    Matrix* matrix = (Matrix*)_matrix;
    Vector* vector = (Vector*)_vector;
    sizetype size = vector->size;
    sizetype* x = matrix->x;
    sizetype* y = matrix->y;
    valuetype* mv = matrix->values;
    valuetype* vv = vector->values;
    valuetype* ret = allocate_values(size);//new valuetype[size]();
    sizetype i;
    #pragma omp parallel for shared(size, ret) private(i)
    for(i=0;i<size;i++)
        ret[i] = 0;
    sizetype entries = matrix->entries;
    #pragma omp parallel for shared(entries, ret, x, y, mv, vv) private(i)
    for(i=0;i<entries;i++)
        ret[y[i]] += mv[i]*vv[x[i]];
    return new Vector(ret, size, false);
}


extern "C" EXPORT void* mask(void* _a, void* _b) {
    Vector* a = (Vector*)_a;
    Vector* b = (Vector*)_b;
    valuetype* av = a->values;
    valuetype* bv = b->values;
    sizetype size = a->size;
    sizetype i;
    sizetype nonzeroes = 0;
    #pragma omp parallel for shared(size, av, bv) private(i) reduction(+:nonzeroes)
    for(i=0;i<size;i++)
        if(bv[i])
            nonzeroes += 1;
    valuetype* ret = allocate_values(nonzeroes);
    sizetype j = 0;
    for(i=0;i<size;i++)
        if(bv[i]) {
            ret[j] = av[i];
            j++;
        }
    return new Vector(ret, nonzeroes, false);
}

extern "C" EXPORT void* add(void* _a, void* _b) {
    Vector* a = (Vector*)_a;
    Vector* b = (Vector*)_b;
    valuetype* av = a->values;
    valuetype* bv = b->values;
    sizetype size = a->size;
    valuetype* ret = allocate_values(size);
    sizetype i;
    #pragma omp parallel for shared(size, ret, av, bv) private(i)
    for(i=0;i<size;i++)
        ret[i] = av[i] + bv[i];
    return new Vector(ret, size, false);
}

extern "C" EXPORT void* equals(void* _a, void* _b) {
    Vector* a = (Vector*)_a;
    Vector* b = (Vector*)_b;
    valuetype* av = a->values;
    valuetype* bv = b->values;
    sizetype size = a->size;
    valuetype* ret = allocate_values(size);
    sizetype i;
    #pragma omp parallel for shared(size, ret, av, bv) private(i)
    for(i=0;i<size;i++)
        ret[i] = av[i] == bv[i];
    return new Vector(ret, size, false);
}

extern "C" EXPORT void* greater(void* _a, void* _b) {
    Vector* a = (Vector*)_a;
    Vector* b = (Vector*)_b;
    valuetype* av = a->values;
    valuetype* bv = b->values;
    sizetype size = a->size;
    valuetype* ret = allocate_values(size);
    sizetype i;
    #pragma omp parallel for shared(size, ret, av, bv) private(i)
    for(i=0;i<size;i++)
        ret[i] = av[i] > bv[i];
    return new Vector(ret, size, false);
}


extern "C" EXPORT void* greater_eq(void* _a, void* _b) {
    Vector* a = (Vector*)_a;
    Vector* b = (Vector*)_b;
    valuetype* av = a->values;
    valuetype* bv = b->values;
    sizetype size = a->size;
    valuetype* ret = allocate_values(size);
    sizetype i;
    #pragma omp parallel for shared(size, ret, av, bv) private(i)
    for(i=0;i<size;i++)
        ret[i] = av[i] >= bv[i];
    return new Vector(ret, size, false);
}

extern "C" EXPORT void* sub(void* _a, void* _b) {
    Vector* a = (Vector*)_a;
    Vector* b = (Vector*)_b;
    valuetype* av = a->values;
    valuetype* bv = b->values;
    sizetype size = a->size;
    valuetype* ret = allocate_values(size);
    sizetype i;
    #pragma omp parallel for shared(size, ret, av, bv) private(i)
    for(i=0;i<size;i++)
        ret[i] = av[i] - bv[i];
    return new Vector(ret, size, false);
}

extern "C" EXPORT void* v_div(void* _a, void* _b) {
    Vector* a = (Vector*)_a;
    Vector* b = (Vector*)_b;
    valuetype* av = a->values;
    valuetype* bv = b->values;
    sizetype size = a->size;
    valuetype* ret = allocate_values(size);
    sizetype i;
    #pragma omp parallel for shared(size, ret, av, bv) private(i)
    for(i=0;i<size;i++)
        ret[i] = av[i] / bv[i];
    return new Vector(ret, size, false);
}

extern "C" EXPORT void* v_pow(void* _a, void* _b) {
    Vector* a = (Vector*)_a;
    Vector* b = (Vector*)_b;
    valuetype* av = a->values;
    valuetype* bv = b->values;
    sizetype size = a->size;
    valuetype* ret = allocate_values(size);
    sizetype i;
    #pragma omp parallel for shared(size, ret, av, bv) private(i)
    for(i=0;i<size;i++)
        ret[i] = pow(av[i], bv[i]);
    return new Vector(ret, size, false);
}

extern "C" EXPORT void* v_log(void* _a) {
    Vector* a = (Vector*)_a;
    valuetype* av = a->values;
    sizetype size = a->size;
    valuetype* ret = allocate_values(size);
    sizetype i;
    #pragma omp parallel for shared(size, ret, av) private(i)
    for(i=0;i<size;i++)
        ret[i] = log(av[i]);
    return new Vector(ret, size, false);
}

extern "C" EXPORT void* transpose(void* _a) {
    Matrix* a = (Matrix*)_a;
    return new Matrix(a->y, a->x, a->values, a->entries, a->size, true);//TODO: keep count of uses before deleting
}

extern "C" EXPORT valuetype v_sum(void* _a) {
    Vector* a = (Vector*)_a;
    valuetype* av = a->values;
    sizetype size = a->size;
    valuetype ret = 0;
    sizetype i;
    #pragma omp parallel for shared(size, av) private(i) reduction(+:ret)
    for(i=0;i<size;i++)
        ret += av[i];
    return ret;
}


extern "C" EXPORT valuetype v_max(void* _a) {
    Vector* a = (Vector*)_a;
    valuetype* av = a->values;
    sizetype size = a->size;
    valuetype ret = av[0];
    sizetype i;
    //#pragma omp parallel for shared(size, av) private(i) reduction(max:ret)
    for(i=0;i<size;i++)
        ret = av[i]>ret?av[i]:ret;
    return ret;
}

extern "C" EXPORT valuetype v_min(void* _a) {
    Vector* a = (Vector*)_a;
    valuetype* av = a->values;
    sizetype size = a->size;
    valuetype ret = av[0];
    sizetype i;
    //#pragma omp parallel for shared(size, av) private(i) reduction(min:ret)
    for(i=0;i<size;i++)
        ret = av[i]<ret?av[i]:ret;
    return ret;
}

extern "C" EXPORT valuetype m_sum_all(void* _a) {
    Matrix* a = (Matrix*)_a;
    valuetype* av = a->values;
    sizetype size = a->size;
    valuetype ret = 0;
    sizetype i;
    #pragma omp parallel for shared(size, av) private(i) reduction(+:ret)
    for(i=0;i<size;i++)
        ret += av[i];
    return ret;
}

extern "C" EXPORT void* m_sum_rows(void* _a) {
    Matrix* a = (Matrix*)_a;
    sizetype* ax = a->x;
    valuetype* av = a->values;
    sizetype size = a->size;
    valuetype* ret = allocate_values(size);//new valuetype[size]();
    sizetype i;
    #pragma omp parallel for shared(size, ret) private(i)
    for(i=0;i<size;i++)
        ret[i] = 0;
    #pragma omp parallel for shared(size, ret, ax, av) private(i)
    for(i=0;i<size;i++)
        ret[ax[i]] += av[i];
    return new Vector(ret, size, false);
}


extern "C" EXPORT void* m_sum_cols(void* _a) {
    Matrix* a = (Matrix*)_a;
    sizetype* ay = a->y;
    valuetype* av = a->values;
    sizetype size = a->size;
    valuetype* ret = allocate_values(size);//new valuetype[size]();
    sizetype i;
    #pragma omp parallel for shared(size, ret) private(i)
    for(i=0;i<size;i++)
        ret[i] = 0;
    #pragma omp parallel for shared(size, ret, ay, av) private(i)
    for(i=0;i<size;i++)
        ret[ay[i]] += av[i];
    return new Vector(ret, size, false);
}

extern "C" EXPORT valuetype v_mean(void* _a) {
    Vector* a = (Vector*)_a;
    valuetype* av = a->values;
    sizetype size = a->size;
    valuetype ret = 0;
    sizetype i;
    #pragma omp parallel for shared(size, av) private(i) reduction(+:ret)
    for(i=0;i<size;i++)
        ret += av[i];
    if(size!=0)
        ret /= size;
    return ret;
}

extern "C" EXPORT valuetype dot(void* _a, void* _b) {
    Vector* a = (Vector*)_a;
    Vector* b = (Vector*)_b;
    valuetype* av = a->values;
    valuetype* bv = b->values;
    sizetype size = a->size;
    valuetype ret = 0;
    sizetype i;
    #pragma omp parallel for shared(size, av, bv) private(i) reduction(+:ret)
    for(i=0;i<size;i++)
        ret += av[i]*bv[i];
    return ret;
}

extern "C" EXPORT void* v_abs(void* _a) {
    Vector* a = (Vector*)_a;
    valuetype* av = a->values;
    sizetype size = a->size;
    valuetype* ret = allocate_values(size);
    sizetype i;
    #pragma omp parallel for shared(size, ret, av) private(i)
    for(i=0;i<size;i++) {
        valuetype value = av[i];
        ret[i] = value<0?-value:value;
      }
    return new Vector(ret, size, false);
}

extern "C" EXPORT void* v_exp(void* _a) {
    Vector* a = (Vector*)_a;
    valuetype* av = a->values;
    sizetype size = a->size;
    valuetype* ret = allocate_values(size);
    sizetype i;
    #pragma omp parallel for shared(size, ret, av) private(i)
    for(i=0;i<size;i++)
        ret[i] = exp(av[i]);
    return new Vector(ret, size, false);
}

extern "C" EXPORT void assign(void* _a, void* _b) {
    Vector* a = (Vector*)_a;
    Vector* b = (Vector*)_b;
    valuetype* av = a->values;
    valuetype* bv = b->values;
    sizetype size = a->size;
    sizetype i;
    #pragma omp parallel for shared(size, av, bv) private(i)
    for(i=0;i<size;i++)
        av[i] = bv[i];
}

extern "C" EXPORT void* v_mult(void* _a, void* _b) {
    Vector* a = (Vector*)_a;
    Vector* b = (Vector*)_b;
    valuetype* av = a->values;
    valuetype* bv = b->values;
    sizetype size = a->size;
    valuetype* ret = allocate_values(size);
    sizetype i;
    #pragma omp parallel for shared(size, ret, av, bv) private(i)
    for(i=0;i<size;i++)
        ret[i] = av[i] * bv[i];
    return new Vector(ret, size, false);
}

extern "C" EXPORT void* vc_add(void* _a, valuetype b) {
    Vector* a = (Vector*)_a;
    valuetype* av = a->values;
    sizetype size = a->size;
    valuetype* ret = allocate_values(size);
    sizetype i;
    #pragma omp parallel for shared(size, ret, av) private(i)
    for(i=0;i<size;i++)
        ret[i] = av[i] + b;
    return new Vector(ret, size, false);
}

extern "C" EXPORT void* vc_equals(void* _a, valuetype b) {
    Vector* a = (Vector*)_a;
    valuetype* av = a->values;
    sizetype size = a->size;
    valuetype* ret = allocate_values(size);
    sizetype i;
    #pragma omp parallel for shared(size, ret, av) private(i)
    for(i=0;i<size;i++)
        ret[i] = av[i] == b;
    return new Vector(ret, size, false);
}


extern "C" EXPORT void* vc_greater(void* _a, valuetype b) {
    Vector* a = (Vector*)_a;
    valuetype* av = a->values;
    sizetype size = a->size;
    valuetype* ret = allocate_values(size);
    sizetype i;
    #pragma omp parallel for shared(size, ret, av) private(i)
    for(i=0;i<size;i++)
        ret[i] = av[i] > b;
    return new Vector(ret, size, false);
}

extern "C" EXPORT void* vc_greater_eq(void* _a, valuetype b) {
    Vector* a = (Vector*)_a;
    valuetype* av = a->values;
    sizetype size = a->size;
    valuetype* ret = allocate_values(size);
    sizetype i;
    #pragma omp parallel for shared(size, ret, av) private(i)
    for(i=0;i<size;i++)
        ret[i] = av[i] >= b;
    return new Vector(ret, size, false);
}


extern "C" EXPORT void* vc_less(void* _a, valuetype b) {
    Vector* a = (Vector*)_a;
    valuetype* av = a->values;
    sizetype size = a->size;
    valuetype* ret = allocate_values(size);
    sizetype i;
    #pragma omp parallel for shared(size, ret, av) private(i)
    for(i=0;i<size;i++)
        ret[i] = av[i] < b;
    return new Vector(ret, size, false);
}


extern "C" EXPORT void* vc_less_eq(void* _a, valuetype b) {
    Vector* a = (Vector*)_a;
    valuetype* av = a->values;
    sizetype size = a->size;
    valuetype* ret = allocate_values(size);
    sizetype i;
    #pragma omp parallel for shared(size, ret, av) private(i)
    for(i=0;i<size;i++)
        ret[i] = av[i] <= b;
    return new Vector(ret, size, false);
}



extern "C" EXPORT void* vc_sub(void* _a, valuetype b) {
    Vector* a = (Vector*)_a;
    valuetype* av = a->values;
    sizetype size = a->size;
    valuetype* ret = allocate_values(size);
    sizetype i;
    #pragma omp parallel for shared(size, ret, av) private(i)
    for(i=0;i<size;i++)
        ret[i] = av[i] - b;
    return new Vector(ret, size, false);
}


extern "C" EXPORT void* cv_sub(void* _a, valuetype b) {
    Vector* a = (Vector*)_a;
    valuetype* av = a->values;
    sizetype size = a->size;
    valuetype* ret = allocate_values(size);
    sizetype i;
    #pragma omp parallel for shared(size, ret, av) private(i)
    for(i=0;i<size;i++)
        ret[i] = b-av[i];
    return new Vector(ret, size, false);
}


extern "C" EXPORT void* vc_div(void* _a, valuetype b) {
    Vector* a = (Vector*)_a;
    valuetype* av = a->values;
    sizetype size = a->size;
    valuetype* ret = allocate_values(size);
    sizetype i;
    #pragma omp parallel for shared(size, ret, av) private(i)
    for(i=0;i<size;i++)
        ret[i] = av[i] / b;
    return new Vector(ret, size, false);
}


extern "C" EXPORT void* cv_div(void* _a, valuetype b) {
    Vector* a = (Vector*)_a;
    valuetype* av = a->values;
    sizetype size = a->size;
    valuetype* ret = allocate_values(size);
    sizetype i;
    #pragma omp parallel for shared(size, ret, av) private(i)
    for(i=0;i<size;i++)
        ret[i] = b / av[i];
    return new Vector(ret, size, false);
}


extern "C" EXPORT void* vc_pow(void* _a, valuetype b) {
    Vector* a = (Vector*)_a;
    valuetype* av = a->values;
    sizetype size = a->size;
    valuetype* ret = allocate_values(size);
    sizetype i;
    #pragma omp parallel for shared(size, ret, av) private(i)
    for(i=0;i<size;i++)
        ret[i] = pow(av[i], b);
    return new Vector(ret, size, false);
}


extern "C" EXPORT void* cv_pow(void* _a, valuetype b) {
    Vector* a = (Vector*)_a;
    valuetype* av = a->values;
    sizetype size = a->size;
    valuetype* ret = allocate_values(size);
    sizetype i;
    #pragma omp parallel for shared(size, ret, av) private(i)
    for(i=0;i<size;i++)
        ret[i] = pow(b, av[i]);
    return new Vector(ret, size, false);
}

extern "C" EXPORT void* vc_mult(void* _a, valuetype b) {
    Vector* a = (Vector*)_a;
    valuetype* av = a->values;
    sizetype size = a->size;
    valuetype* ret = allocate_values(size);
    sizetype i;
    #pragma omp parallel for shared(size, ret, av) private(i)
    for(i=0;i<size;i++)
        ret[i] = av[i] * b;
    return new Vector(ret, size, false);
}



extern "C" EXPORT void* matrix(PyObject* x, PyObject* y, PyObject* values, sizetype entries, sizetype size){
    Py_buffer view_x, view_y, view_v;
    if (PyObject_GetBuffer(x, &view_x, PyBUF_ANY_CONTIGUOUS | PyBUF_FORMAT) == -1)
        return NULL;
    if (PyObject_GetBuffer(y, &view_y, PyBUF_ANY_CONTIGUOUS | PyBUF_FORMAT) == -1) {
        PyBuffer_Release(&view_x);
        return NULL;
    }
    if (PyObject_GetBuffer(values, &view_v, PyBUF_ANY_CONTIGUOUS | PyBUF_FORMAT) == -1) {
        PyBuffer_Release(&view_x);
        PyBuffer_Release(&view_y);
        return NULL;
    }
    if (view_x.ndim != 1 || view_y.ndim !=1 || view_v.ndim != 1) {
        PyErr_SetString(PyExc_TypeError, "Expected a 1-dimensional arrays");
        PyBuffer_Release(&view_x);
        PyBuffer_Release(&view_y);
        PyBuffer_Release(&view_v);
        return NULL;
    }
    sizetype* x_ret = new sizetype[entries];
    sizetype* x_buf = (sizetype*)view_x.buf;
    for(sizetype i=0;i<entries;i++)
        x_ret[i] = x_buf[i];

    sizetype* y_ret = new sizetype[entries];
    sizetype* y_buf = (sizetype*)view_y.buf;
    for(sizetype i=0;i<entries;i++)
        y_ret[i] = y_buf[i];

    valuetype* v_ret = new valuetype[entries];
    valuetype* v_buf = (valuetype*)view_v.buf;
    for(sizetype i=0;i<entries;i++)
        v_ret[i] = v_buf[i];

    Matrix* mat = new Matrix(x_ret, y_ret, v_ret, entries, size, false);
    PyBuffer_Release(&view_x);
    PyBuffer_Release(&view_y);
    PyBuffer_Release(&view_v);
    return mat;
}

//extern "C" EXPORT void* vector(valuetype* values, sizetype size) {return new Vector(values, size);}
extern "C" EXPORT valuetype get(void* vector, sizetype i) {return ((Vector*)vector)->values[i];}
extern "C" EXPORT void set(void* vector, sizetype i, valuetype value){((Vector*)vector)->values[i] = value;}
extern "C" EXPORT sizetype len(void* vector){return ((Vector*)vector)->size;}
extern "C" EXPORT sizetype m_len(void* matrix){return ((Matrix*)matrix)->size;}

extern "C" EXPORT void free_vector(void* obj) {
    delete (Vector*)obj;
}

extern "C" EXPORT void free_matrix(void* obj) {
    delete (Matrix*)obj;
}

extern "C" EXPORT void clear() {
    for (std::unordered_map<sizetype, std::vector<valuetype*>*>::iterator it = reusable.begin(); it != reusable.end(); it++) {
        for(int i=0;i <it->second->size();i++)
            delete it->second->at(i);
        delete it->second;
    }
    reusable.clear();
}

extern "C" EXPORT void* vector(PyObject *values, sizetype size) {
    Py_buffer* view = new Py_buffer();
    if (PyObject_GetBuffer(values, view, PyBUF_ANY_CONTIGUOUS | PyBUF_FORMAT) == -1)
        return NULL;
    if (view->ndim != 1) {
        PyErr_SetString(PyExc_TypeError, "Expected a 1-dimensional array");
        PyBuffer_Release(view);
        return NULL;
    }
    valuetype* ret = allocate_values(size);//view->buf
    valuetype* buf = (valuetype*)view->buf;
    for(sizetype i=0;i<size;i++)
        ret[i] = buf[i];

    Vector* vect = new Vector(ret, size, false);//, view);
    PyBuffer_Release(view);
    delete view;
    return vect;
}


extern "C" EXPORT void* repeat(valuetype value, sizetype size) {
    valuetype* ret = allocate_values(size);
    sizetype i;
    #pragma omp parallel for shared(size, ret) private(i)
    for(i=0;i<size;i++)
        ret[i] = value;
    return new Vector(ret, size, false);
}

extern "C" EXPORT void* v_copy(void* _vector) {
    Vector* vector = (Vector*)_vector;
    return new Vector(vector->values, vector->size);
}

extern "C" EXPORT void* get_values(void* _matrix) {
    Matrix* matrix = (Matrix*)_matrix;
    return new Vector(matrix->values, matrix->entries, false);
}

extern "C" EXPORT void* get_rows(void* _matrix) {
    Matrix* matrix = (Matrix*)_matrix;
    sizetype entries = matrix->entries;
    sizetype i;
    sizetype* x = matrix->x;
    valuetype* ret = new valuetype[entries];
    #pragma omp parallel for shared(entries, x, ret) private(i)
    for(i=0;i<entries;i++)
        ret[i] = (valuetype)x[i];
    return new Vector(ret, entries, false);
}

extern "C" EXPORT void* get_cols(void* _matrix) {
    Matrix* matrix = (Matrix*)_matrix;
    sizetype entries = matrix->entries;
    sizetype i;
    sizetype* y = matrix->y;
    valuetype* ret = new valuetype[entries];
    #pragma omp parallel for shared(entries, y, ret) private(i)
    for(i=0;i<entries;i++)
        ret[i] = (valuetype)y[i];
    return new Vector(ret, entries, false);
}

int myrank;
int groupsize;
extern "C" EXPORT void set_number_of_threads(int threads) {
    omp_set_dynamic(0);
    omp_set_num_threads(threads);
}

extern "C" EXPORT PyObject * v_to_array(void* _vector) {
    Vector* vector = (Vector*)_vector;
    for(sizetype i=0;i<vector->size;i++){
        //PyLong_FromLongLong(vector->values[i]);
        //PyFloat_FromDouble(vector->values[i]);
        //PyObject_SetItem(fill, Py_BuildValue("L", i), Py_BuildValue("d", vector->values[i]));
    }
    return NULL;
}