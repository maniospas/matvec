#define sizetype int //long unsigned int
#define iteratortype int
#define valuetype double
#include <iostream>
#include <math.h>
#include <omp.h>
#include <vector>
#include "Python.h"
#include <map>
#include<unordered_map>
#include <cstdlib>

std::unordered_map<sizetype, std::vector<valuetype*>*> reusable;

valuetype* allocate_values(sizetype size) {
    valuetype* pending_values;
    if(reusable.find(size)!=reusable.end() && reusable[size]->size()) {
        pending_values = reusable[size]->back();
        reusable[size]->pop_back();
    }
    else
        pending_values = (valuetype*)malloc(size * sizeof(valuetype));
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
                iteratortype i;
                #pragma omp parallel for shared(_size, pending_values, _values) private(i)
                for(i=0;i<_size;++i)
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
                sizetype* px = (sizetype*)malloc(_entries * sizeof(sizetype));//new sizetype[_entries];
                sizetype* py = (sizetype*)malloc(_entries * sizeof(sizetype));//new sizetype[_entries];
                valuetype* pv = (valuetype*)malloc(_entries * sizeof(valuetype));//new valuetype[_entries];
                iteratortype i;
                #pragma omp parallel for shared(_entries, px, py, pv, _x, _y, _values) private(i)
                for(i=0;i<_entries;++i) {
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
            /*iteratortype i;
            //#pragma omp parallel for shared(_entries, px, py, pv, _x, _y, _values) private(i)
            for(i=0;i<_entries;++i) {
                if(row_indexes.find(_x[i]) != row_indexes.end())
                    row_indexes[_x[i]] = new std::vector<sizetype>();
                row_indexes[_x[i]]->push_back(i);
            }*/
        }
        ~Matrix() {
            free(x);
            free(y);
            free(values);
        }
};

void* multiply(void* _matrix, void* _vector) {
    Matrix* matrix = (Matrix*)_matrix;
    Vector* vector = (Vector*)_vector;
    sizetype size = vector->size;
    sizetype* x = matrix->x;
    sizetype* y = matrix->y;
    valuetype* mv = matrix->values;
    valuetype* vv = vector->values;
    valuetype* ret = allocate_values(size);//new valuetype[size]();
    iteratortype i;
    #pragma omp parallel for shared(size, ret) private(i)
    for(i=0;i<size;++i)
        ret[i] = 0;
    sizetype entries = matrix->entries;
    #pragma omp parallel for shared(entries, ret, x, y, mv, vv) private(i)
    for(i=0;i<entries;++i) {
        valuetype val = mv[i]*vv[y[i]];
        #pragma omp atomic
        ret[x[i]] += val;
    }

    return new Vector(ret, size, false);
}


void* rmultiply(void* _matrix, void* _vector) {
    Matrix* matrix = (Matrix*)_matrix;
    Vector* vector = (Vector*)_vector;
    sizetype size = vector->size;
    sizetype* x = matrix->x;
    sizetype* y = matrix->y;
    valuetype* mv = matrix->values;
    valuetype* vv = vector->values;
    valuetype* ret = allocate_values(size);//new valuetype[size]();
    iteratortype i;
    #pragma omp parallel for shared(size, ret) private(i)
    for(i=0;i<size;++i)
        ret[i] = 0;
    sizetype entries = matrix->entries;
    #pragma omp parallel for shared(entries, ret, x, y, mv, vv) private(i)
    for(i=0;i<entries;++i) {
        valuetype val = mv[i]*vv[x[i]];
        #pragma omp atomic
        ret[y[i]] += val;
    }
    /*
    sizetype row;
    #pragma omp parallel for shared(entries, ret, x, y, mv, vv, matrix) private(row)
    for(row=0;row<size;row++) {
        std::vector<sizetype>* adjacent = matrix->row_indexes[row];
        std::cout<<row<" row\n";
        for(int j=0;j<adjacent->size();++i) {
            i = adjacent->at(j);
            ret[row] += mv[i]*vv[y[i]];
        }
    }*/

    return new Vector(ret, size, false);
}


void* mask(void* _a, void* _b) {
    Vector* a = (Vector*)_a;
    Vector* b = (Vector*)_b;
    valuetype* av = a->values;
    valuetype* bv = b->values;
    sizetype size = a->size;
    iteratortype i;
    sizetype nonzeroes = 0;
    #pragma omp parallel for shared(size, av, bv) private(i) reduction(+:nonzeroes)
    for(i=0;i<size;++i)
        if(bv[i])
            nonzeroes += 1;
    valuetype* ret = allocate_values(nonzeroes);
    sizetype j = 0;
    for(i=0;i<size;++i)
        if(bv[i]) {
            ret[j] = av[i];
            j++;
        }
    return new Vector(ret, nonzeroes, false);
}

void* add(void* _a, void* _b) {
    Vector* a = (Vector*)_a;
    Vector* b = (Vector*)_b;
    valuetype* av = a->values;
    valuetype* bv = b->values;
    sizetype size = a->size;
    valuetype* ret = allocate_values(size);
    iteratortype i;
    #pragma omp parallel for shared(size, ret, av, bv) private(i)
    for(i=0;i<size;++i)
        ret[i] = av[i] + bv[i];
    return new Vector(ret, size, false);
}

void* equals(void* _a, void* _b) {
    Vector* a = (Vector*)_a;
    Vector* b = (Vector*)_b;
    valuetype* av = a->values;
    valuetype* bv = b->values;
    sizetype size = a->size;
    valuetype* ret = allocate_values(size);
    iteratortype i;
    #pragma omp parallel for shared(size, ret, av, bv) private(i)
    for(i=0;i<size;++i)
        ret[i] = av[i] == bv[i];
    return new Vector(ret, size, false);
}

void* greater(void* _a, void* _b) {
    Vector* a = (Vector*)_a;
    Vector* b = (Vector*)_b;
    valuetype* av = a->values;
    valuetype* bv = b->values;
    sizetype size = a->size;
    valuetype* ret = allocate_values(size);
    iteratortype i;
    #pragma omp parallel for shared(size, ret, av, bv) private(i)
    for(i=0;i<size;++i)
        ret[i] = av[i] > bv[i];
    return new Vector(ret, size, false);
}


void* greater_eq(void* _a, void* _b) {
    Vector* a = (Vector*)_a;
    Vector* b = (Vector*)_b;
    valuetype* av = a->values;
    valuetype* bv = b->values;
    sizetype size = a->size;
    valuetype* ret = allocate_values(size);
    iteratortype i;
    #pragma omp parallel for shared(size, ret, av, bv) private(i)
    for(i=0;i<size;++i)
        ret[i] = av[i] >= bv[i];
    return new Vector(ret, size, false);
}

void* sub(void* _a, void* _b) {
    Vector* a = (Vector*)_a;
    Vector* b = (Vector*)_b;
    valuetype* av = a->values;
    valuetype* bv = b->values;
    sizetype size = a->size;
    valuetype* ret = allocate_values(size);
    iteratortype i;
    #pragma omp parallel for shared(size, ret, av, bv) private(i)
    for(i=0;i<size;++i)
        ret[i] = av[i] - bv[i];
    return new Vector(ret, size, false);
}

void* v_div(void* _a, void* _b) {
    Vector* a = (Vector*)_a;
    Vector* b = (Vector*)_b;
    valuetype* av = a->values;
    valuetype* bv = b->values;
    sizetype size = a->size;
    valuetype* ret = allocate_values(size);
    iteratortype i;
    #pragma omp parallel for shared(size, ret, av, bv) private(i)
    for(i=0;i<size;++i)
        ret[i] = av[i] / bv[i];
    return new Vector(ret, size, false);
}

void* v_pow(void* _a, void* _b) {
    Vector* a = (Vector*)_a;
    Vector* b = (Vector*)_b;
    valuetype* av = a->values;
    valuetype* bv = b->values;
    sizetype size = a->size;
    valuetype* ret = allocate_values(size);
    iteratortype i;
    #pragma omp parallel for shared(size, ret, av, bv) private(i)
    for(i=0;i<size;++i)
        ret[i] = pow(av[i], bv[i]);
    return new Vector(ret, size, false);
}

void* v_log(void* _a) {
    Vector* a = (Vector*)_a;
    valuetype* av = a->values;
    sizetype size = a->size;
    valuetype* ret = allocate_values(size);
    iteratortype i;
    #pragma omp parallel for shared(size, ret, av) private(i)
    for(i=0;i<size;++i)
        ret[i] = log(av[i]);
    return new Vector(ret, size, false);
}

void* transpose(void* _a) {
    Matrix* a = (Matrix*)_a;
    return new Matrix(a->y, a->x, a->values, a->entries, a->size, true);//TODO: keep count of uses before deleting
}

valuetype v_sum(void* _a) {
    Vector* a = (Vector*)_a;
    valuetype* av = a->values;
    sizetype size = a->size;
    valuetype ret = 0;
    iteratortype i;
    #pragma omp parallel for shared(size, av) private(i) reduction(+:ret)
    for(i=0;i<size;++i)
        ret += av[i];
    return ret;
}

valuetype v_norm(void* _a) {
    Vector* a = (Vector*)_a;
    valuetype* av = a->values;
    sizetype size = a->size;
    valuetype ret = 0;
    iteratortype i;
    #pragma omp parallel for shared(size, av) private(i) reduction(+:ret)
    for(i=0;i<size;++i)
        ret += av[i]*av[i];
    return sqrt(ret);
}

valuetype v_max(void* _a) {
    Vector* a = (Vector*)_a;
    valuetype* av = a->values;
    sizetype size = a->size;
    valuetype ret = av[0];
    iteratortype i;
    //#pragma omp parallel for shared(size, av) private(i) reduction(max:ret)
    for(i=0;i<size;++i)
        ret = av[i]>ret?av[i]:ret;
    return ret;
}

valuetype v_min(void* _a) {
    Vector* a = (Vector*)_a;
    valuetype* av = a->values;
    sizetype size = a->size;
    valuetype ret = av[0];
    iteratortype i;
    //#pragma omp parallel for shared(size, av) private(i) reduction(min:ret)
    for(i=0;i<size;++i)
        ret = av[i]<ret?av[i]:ret;
    return ret;
}

valuetype m_sum_all(void* _a) {
    Matrix* a = (Matrix*)_a;
    valuetype* av = a->values;
    sizetype size = a->size;
    valuetype ret = 0;
    iteratortype i;
    #pragma omp parallel for shared(size, av) private(i) reduction(+:ret)
    for(i=0;i<size;++i)
        ret += av[i];
    return ret;
}

void* m_sum_rows(void* _a) {
    Matrix* a = (Matrix*)_a;
    sizetype* ax = a->x;
    valuetype* av = a->values;
    sizetype size = a->size;
    valuetype* ret = allocate_values(size);//new valuetype[size]();
    sizetype entries = a->entries;
    iteratortype i;
    #pragma omp parallel for shared(size, ret) private(i)
    for(i=0;i<size;++i)
        ret[i] = 0;
    #pragma omp parallel for shared(size, ret, ax, av) private(i)
    for(i=0;i<entries;++i) {
        valuetype val = av[i];
        #pragma omp atomic
        ret[ax[i]] += val;
     }
    return new Vector(ret, size, false);
}


void* m_sum_cols(void* _a) {
    Matrix* a = (Matrix*)_a;
    sizetype* ay = a->y;
    valuetype* av = a->values;
    sizetype size = a->size;
    valuetype* ret = allocate_values(size);//new valuetype[size]();
    sizetype entries = a->entries;
    iteratortype i;
    #pragma omp parallel for shared(size, ret) private(i)
    for(i=0;i<size;++i)
        ret[i] = 0;
    #pragma omp parallel for shared(size, ret, ay, av) private(i)
    for(i=0;i<entries;++i) {
        valuetype val = av[i];
        #pragma omp atomic
        ret[ay[i]] += val;
     }
    return new Vector(ret, size, false);
}

valuetype v_mean(void* _a) {
    Vector* a = (Vector*)_a;
    valuetype* av = a->values;
    sizetype size = a->size;
    valuetype ret = 0;
    iteratortype i;
    #pragma omp parallel for shared(size, av) private(i) reduction(+:ret)
    for(i=0;i<size;++i)
        ret += av[i];
    if(size!=0)
        ret /= size;
    return ret;
}

valuetype dot(void* _a, void* _b) {
    Vector* a = (Vector*)_a;
    Vector* b = (Vector*)_b;
    valuetype* av = a->values;
    valuetype* bv = b->values;
    sizetype size = a->size;
    valuetype ret = 0;
    iteratortype i;
    #pragma omp parallel for shared(size, av, bv) private(i) reduction(+:ret)
    for(i=0;i<size;++i)
        ret += av[i]*bv[i];
    return ret;
}

void* v_abs(void* _a) {
    Vector* a = (Vector*)_a;
    valuetype* av = a->values;
    sizetype size = a->size;
    valuetype* ret = allocate_values(size);
    iteratortype i;
    #pragma omp parallel for shared(size, ret, av) private(i)
    for(i=0;i<size;++i) {
        valuetype value = av[i];
        ret[i] = value<0?-value:value;
      }
    return new Vector(ret, size, false);
}

void* v_exp(void* _a) {
    Vector* a = (Vector*)_a;
    valuetype* av = a->values;
    sizetype size = a->size;
    valuetype* ret = allocate_values(size);
    iteratortype i;
    #pragma omp parallel for shared(size, ret, av) private(i)
    for(i=0;i<size;++i)
        ret[i] = exp(av[i]);
    return new Vector(ret, size, false);
}

void assign(void* _a, void* _b) {
    Vector* a = (Vector*)_a;
    Vector* b = (Vector*)_b;
    valuetype* av = a->values;
    valuetype* bv = b->values;
    sizetype size = a->size;
    iteratortype i;
    #pragma omp parallel for shared(size, av, bv) private(i)
    for(i=0;i<size;++i)
        av[i] = bv[i];
}

void* v_mult(void* _a, void* _b) {
    Vector* a = (Vector*)_a;
    Vector* b = (Vector*)_b;
    valuetype* av = a->values;
    valuetype* bv = b->values;
    sizetype size = a->size;
    valuetype* ret = allocate_values(size);
    iteratortype i;
    #pragma omp parallel for shared(size, ret, av, bv) private(i)
    for(i=0;i<size;++i)
        ret[i] = av[i] * bv[i];
    return new Vector(ret, size, false);
}

void* vc_add(void* _a, valuetype b) {
    Vector* a = (Vector*)_a;
    valuetype* av = a->values;
    sizetype size = a->size;
    valuetype* ret = allocate_values(size);
    iteratortype i;
    #pragma omp parallel for shared(size, ret, av) private(i)
    for(i=0;i<size;++i)
        ret[i] = av[i] + b;
    return new Vector(ret, size, false);
}

void* vc_equals(void* _a, valuetype b) {
    Vector* a = (Vector*)_a;
    valuetype* av = a->values;
    sizetype size = a->size;
    valuetype* ret = allocate_values(size);
    iteratortype i;
    #pragma omp parallel for shared(size, ret, av) private(i)
    for(i=0;i<size;++i)
        ret[i] = av[i] == b;
    return new Vector(ret, size, false);
}


void* vc_greater(void* _a, valuetype b) {
    Vector* a = (Vector*)_a;
    valuetype* av = a->values;
    sizetype size = a->size;
    valuetype* ret = allocate_values(size);
    iteratortype i;
    #pragma omp parallel for shared(size, ret, av) private(i)
    for(i=0;i<size;++i)
        ret[i] = av[i] > b;
    return new Vector(ret, size, false);
}

void* vc_greater_eq(void* _a, valuetype b) {
    Vector* a = (Vector*)_a;
    valuetype* av = a->values;
    sizetype size = a->size;
    valuetype* ret = allocate_values(size);
    iteratortype i;
    #pragma omp parallel for shared(size, ret, av) private(i)
    for(i=0;i<size;++i)
        ret[i] = av[i] >= b;
    return new Vector(ret, size, false);
}


void* vc_less(void* _a, valuetype b) {
    Vector* a = (Vector*)_a;
    valuetype* av = a->values;
    sizetype size = a->size;
    valuetype* ret = allocate_values(size);
    iteratortype i;
    #pragma omp parallel for shared(size, ret, av) private(i)
    for(i=0;i<size;++i)
        ret[i] = av[i] < b;
    return new Vector(ret, size, false);
}


void* vc_less_eq(void* _a, valuetype b) {
    Vector* a = (Vector*)_a;
    valuetype* av = a->values;
    sizetype size = a->size;
    valuetype* ret = allocate_values(size);
    iteratortype i;
    #pragma omp parallel for shared(size, ret, av) private(i)
    for(i=0;i<size;++i)
        ret[i] = av[i] <= b;
    return new Vector(ret, size, false);
}



void* vc_sub(void* _a, valuetype b) {
    Vector* a = (Vector*)_a;
    valuetype* av = a->values;
    sizetype size = a->size;
    valuetype* ret = allocate_values(size);
    iteratortype i;
    #pragma omp parallel for shared(size, ret, av) private(i)
    for(i=0;i<size;++i)
        ret[i] = av[i] - b;
    return new Vector(ret, size, false);
}


void* cv_sub(void* _a, valuetype b) {
    Vector* a = (Vector*)_a;
    valuetype* av = a->values;
    sizetype size = a->size;
    valuetype* ret = allocate_values(size);
    iteratortype i;
    #pragma omp parallel for shared(size, ret, av) private(i)
    for(i=0;i<size;++i)
        ret[i] = b-av[i];
    return new Vector(ret, size, false);
}


void* vc_div(void* _a, valuetype b) {
    Vector* a = (Vector*)_a;
    valuetype* av = a->values;
    sizetype size = a->size;
    valuetype* ret = allocate_values(size);
    iteratortype i;
    #pragma omp parallel for shared(size, ret, av) private(i)
    for(i=0;i<size;++i)
        ret[i] = av[i] / b;
    return new Vector(ret, size, false);
}


void* cv_div(void* _a, valuetype b) {
    Vector* a = (Vector*)_a;
    valuetype* av = a->values;
    sizetype size = a->size;
    valuetype* ret = allocate_values(size);
    iteratortype i;
    #pragma omp parallel for shared(size, ret, av) private(i)
    for(i=0;i<size;++i)
        ret[i] = b / av[i];
    return new Vector(ret, size, false);
}


void* vc_pow(void* _a, valuetype b) {
    Vector* a = (Vector*)_a;
    valuetype* av = a->values;
    sizetype size = a->size;
    valuetype* ret = allocate_values(size);
    iteratortype i;
    #pragma omp parallel for shared(size, ret, av) private(i)
    for(i=0;i<size;++i)
        ret[i] = pow(av[i], b);
    return new Vector(ret, size, false);
}


void* cv_pow(void* _a, valuetype b) {
    Vector* a = (Vector*)_a;
    valuetype* av = a->values;
    sizetype size = a->size;
    valuetype* ret = allocate_values(size);
    iteratortype i;
    #pragma omp parallel for shared(size, ret, av) private(i)
    for(i=0;i<size;++i)
        ret[i] = pow(b, av[i]);
    return new Vector(ret, size, false);
}

void* vc_mult(void* _a, valuetype b) {
    Vector* a = (Vector*)_a;
    valuetype* av = a->values;
    sizetype size = a->size;
    valuetype* ret = allocate_values(size);
    iteratortype i;
    #pragma omp parallel for shared(size, ret, av) private(i)
    for(i=0;i<size;++i)
        ret[i] = av[i] * b;
    return new Vector(ret, size, false);
}



void* matrix(PyObject* x, PyObject* y, PyObject* values, sizetype entries, sizetype size){
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
    sizetype* x_ret = (sizetype*)malloc(entries * sizeof(sizetype));//new sizetype[entries];
    sizetype* x_buf = (sizetype*)view_x.buf;
    #pragma omp parallel
    for(iteratortype i=0;i<entries;++i)
        x_ret[i] = x_buf[i];

    sizetype* y_ret = (sizetype*)malloc(entries * sizeof(sizetype));//new sizetype[entries];
    sizetype* y_buf = (sizetype*)view_y.buf;
    #pragma omp parallel
    for(iteratortype i=0;i<entries;++i)
        y_ret[i] = y_buf[i];

    valuetype* v_ret = (valuetype*)malloc(entries * sizeof(valuetype));//new valuetype[entries];
    valuetype* v_buf = (valuetype*)view_v.buf;
    #pragma omp parallel
    for(iteratortype i=0;i<entries;++i)
        v_ret[i] = v_buf[i];

    Matrix* mat = new Matrix(x_ret, y_ret, v_ret, entries, size, false);
    PyBuffer_Release(&view_x);
    PyBuffer_Release(&view_y);
    PyBuffer_Release(&view_v);
    return mat;
}

//void* vector(valuetype* values, sizetype size) {return new Vector(values, size);}
valuetype get(void* vector, iteratortype i) {
    if(i>=((Vector*)vector)->size || i<0) {
        PyErr_SetString(PyExc_TypeError, "Cannot get vector element - out of bounds");
        return 0;
    }
    return ((Vector*)vector)->values[i];
}
void set(void* vector, iteratortype i, valuetype value){
    if(i>=((Vector*)vector)->size || i<0) {
        PyErr_SetString(PyExc_TypeError, "Cannot get vector element - out of bounds");
        return;
    }
    ((Vector*)vector)->values[i] = value;
}
sizetype len(void* vector){return ((Vector*)vector)->size;}
sizetype m_len(void* matrix){return ((Matrix*)matrix)->size;}

void free_vector(void* obj) {
    delete (Vector*)obj;
}

void free_matrix(void* obj) {
    delete (Matrix*)obj;
}

void clear() {
    for (std::unordered_map<sizetype, std::vector<valuetype*>*>::iterator it = reusable.begin(); it != reusable.end(); it++) {
        for(sizetype i=0;i <it->second->size();++i)
            delete it->second->at(i);
        delete it->second;
    }
    reusable.clear();
}

void* vector(PyObject *values, sizetype size) {
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
    for(iteratortype i=0;i<size;++i)
        ret[i] = buf[i];

    Vector* vect = new Vector(ret, size, false);//, view);
    PyBuffer_Release(view);
    delete view;
    return vect;
}


void* repeat(valuetype value, sizetype size) {
    valuetype* ret = allocate_values(size);
    iteratortype i;
    #pragma omp parallel for shared(size, ret) private(i)
    for(i=0;i<size;++i)
        ret[i] = value;
    return new Vector(ret, size, false);
}

void* v_copy(void* _vector) {
    Vector* vector = (Vector*)_vector;
    return new Vector(vector->values, vector->size);
}

void* get_values(void* _matrix) {
    Matrix* matrix = (Matrix*)_matrix;
    return new Vector(matrix->values, matrix->entries, false);
}

void* get_rows(void* _matrix) {
    Matrix* matrix = (Matrix*)_matrix;
    sizetype entries = matrix->entries;
    iteratortype i;
    sizetype* x = matrix->x;
    valuetype* ret = new valuetype[entries];
    #pragma omp parallel for shared(entries, x, ret) private(i)
    for(i=0;i<entries;++i)
        ret[i] = (valuetype)x[i];
    return new Vector(ret, entries, false);
}

void* get_cols(void* _matrix) {
    Matrix* matrix = (Matrix*)_matrix;
    sizetype entries = matrix->entries;
    iteratortype i;
    sizetype* y = matrix->y;
    valuetype* ret = new valuetype[entries];
    #pragma omp parallel for shared(entries, y, ret) private(i)
    for(i=0;i<entries;++i)
        ret[i] = (valuetype)y[i];
    return new Vector(ret, entries, false);
}

int myrank;
int groupsize;
void set_number_of_threads(int threads) {
    omp_set_dynamic(0); // Disable dynamic thread adjustment
    omp_set_num_threads(threads);
}

void* v_to_array(void* _vector) {
    Vector* vector = (Vector*)_vector;
    return vector->values;
}
