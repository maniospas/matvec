//#define DLL_EXPORT __declspec(dllimport) extern "C"
#define sizetype long long int
#define valuetype double
#include <iostream>
#include <math.h>
#include <omp.h>

class Vector {
    public:
        valuetype* values;
        sizetype size;

        Vector(valuetype* _values, sizetype _size, bool copy=true) {
            if(copy) {
                valuetype* pending_values = new valuetype[_size];
                sizetype i;
                #pragma omp parallel for shared(_size, pending_values, _values) private(i)
                for(i=0;i<_size;i++)
                    pending_values[i] = _values[i];
                values = pending_values;
            }
            else
                values = _values;
            size = _size;
        }
        ~Vector() {
            delete values;
        }
};


class Matrix {
    public:
        sizetype* x;
        sizetype* y;
        valuetype* values;
        sizetype entries;
        sizetype size;

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
        }
        ~Matrix() {
            delete x;
            delete y;
            delete values;
        }
};

extern "C" __declspec(dllexport) void* multiply(void* _matrix, void* _vector) {
    Matrix* matrix = (Matrix*)_matrix;
    Vector* vector = (Vector*)_vector;
    sizetype size = vector->size;
    sizetype* x = matrix->x;
    sizetype* y = matrix->y;
    valuetype* mv = matrix->values;
    valuetype* vv = vector->values;
    valuetype* ret = new valuetype[size]();
    sizetype entries = matrix->entries;
    sizetype i;
    #pragma omp parallel for shared(entries, ret, x, y, mv, vv) private(i)
    for(i=0;i<entries;i++)
        ret[x[i]] += mv[i]*vv[y[i]];
    return new Vector(ret, size, false);
}


extern "C" __declspec(dllexport) void* rmultiply(void* _matrix, void* _vector) {
    Matrix* matrix = (Matrix*)_matrix;
    Vector* vector = (Vector*)_vector;
    sizetype size = vector->size;
    sizetype* x = matrix->x;
    sizetype* y = matrix->y;
    valuetype* mv = matrix->values;
    valuetype* vv = vector->values;
    valuetype* ret = new valuetype[size]();
    sizetype entries = matrix->entries;
    sizetype i;
    #pragma omp parallel for shared(entries, ret, x, y, mv, vv) private(i)
    for(i=0;i<entries;i++)
        ret[y[i]] += mv[i]*vv[x[i]];
    return new Vector(ret, size, false);
}

extern "C" __declspec(dllexport) void* add(void* _a, void* _b) {
    Vector* a = (Vector*)_a;
    Vector* b = (Vector*)_b;
    valuetype* av = a->values;
    valuetype* bv = b->values;
    sizetype size = a->size;
    valuetype* ret = new valuetype[size];
    sizetype i;
    #pragma omp parallel for shared(size, ret, av, bv) private(i)
    for(i=0;i<size;i++)
        ret[i] = av[i] + bv[i];
    return new Vector(ret, size, false);
}

extern "C" __declspec(dllexport) void* sub(void* _a, void* _b) {
    Vector* a = (Vector*)_a;
    Vector* b = (Vector*)_b;
    valuetype* av = a->values;
    valuetype* bv = b->values;
    sizetype size = a->size;
    valuetype* ret = new valuetype[size];
    sizetype i;
    #pragma omp parallel for shared(size, ret, av, bv) private(i)
    for(i=0;i<size;i++)
        ret[i] = av[i] - bv[i];
    return new Vector(ret, size, false);
}

extern "C" __declspec(dllexport) void* v_div(void* _a, void* _b) {
    Vector* a = (Vector*)_a;
    Vector* b = (Vector*)_b;
    valuetype* av = a->values;
    valuetype* bv = b->values;
    sizetype size = a->size;
    valuetype* ret = new valuetype[size];
    sizetype i;
    #pragma omp parallel for shared(size, ret, av, bv) private(i)
    for(i=0;i<size;i++)
        ret[i] = av[i] / bv[i];
    return new Vector(ret, size, false);
}

extern "C" __declspec(dllexport) void* v_pow(void* _a, void* _b) {
    Vector* a = (Vector*)_a;
    Vector* b = (Vector*)_b;
    valuetype* av = a->values;
    valuetype* bv = b->values;
    sizetype size = a->size;
    valuetype* ret = new valuetype[size];
    sizetype i;
    #pragma omp parallel for shared(size, ret, av, bv) private(i)
    for(i=0;i<size;i++)
        ret[i] = pow(av[i], bv[i]);
    return new Vector(ret, size, false);
}

extern "C" __declspec(dllexport) void* v_log(void* _a) {
    Vector* a = (Vector*)_a;
    valuetype* av = a->values;
    sizetype size = a->size;
    valuetype* ret = new valuetype[size];
    sizetype i;
    #pragma omp parallel for shared(size, ret, av) private(i)
    for(i=0;i<size;i++)
        ret[i] = log(av[i]);
    return new Vector(ret, size, false);
}

extern "C" __declspec(dllexport) void* transpose(void* _a) {
    Matrix* a = (Matrix*)_a;
    return new Matrix(a->y, a->x, a->values, a->entries, a->size, false);
}

extern "C" __declspec(dllexport) valuetype v_sum(void* _a) {
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


extern "C" __declspec(dllexport) valuetype v_max(void* _a) {
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

extern "C" __declspec(dllexport) valuetype v_min(void* _a) {
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

extern "C" __declspec(dllexport) valuetype m_sum_all(void* _a) {
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

extern "C" __declspec(dllexport) void* m_sum_rows(void* _a) {
    Matrix* a = (Matrix*)_a;
    sizetype* ax = a->x;
    valuetype* av = a->values;
    sizetype size = a->size;
    valuetype* ret = new valuetype[size]();
    sizetype i;
    #pragma omp parallel for shared(size, ret, ax, av) private(i)
    for(i=0;i<size;i++)
        ret[ax[i]] += av[i];
    return new Vector(ret, size, false);
}


extern "C" __declspec(dllexport) void* m_sum_cols(void* _a) {
    Matrix* a = (Matrix*)_a;
    sizetype* ay = a->y;
    valuetype* av = a->values;
    sizetype size = a->size;
    valuetype* ret = new valuetype[size]();
    sizetype i;
    #pragma omp parallel for shared(size, ret, ay, av) private(i)
    for(i=0;i<size;i++)
        ret[ay[i]] += av[i];
    return new Vector(ret, size, false);
}

extern "C" __declspec(dllexport) valuetype v_mean(void* _a) {
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

extern "C" __declspec(dllexport) valuetype dot(void* _a, void* _b) {
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

extern "C" __declspec(dllexport) void* v_abs(void* _a) {
    Vector* a = (Vector*)_a;
    valuetype* av = a->values;
    sizetype size = a->size;
    valuetype* ret = new valuetype[size];
    sizetype i;
    #pragma omp parallel for shared(size, ret, av) private(i)
    for(i=0;i<size;i++) {
        valuetype value = av[i];
        ret[i] = value<0?-value:value;
      }
    return new Vector(ret, size, false);
}

extern "C" __declspec(dllexport) void* v_exp(void* _a) {
    Vector* a = (Vector*)_a;
    valuetype* av = a->values;
    sizetype size = a->size;
    valuetype* ret = new valuetype[size];
    sizetype i;
    #pragma omp parallel for shared(size, ret, av) private(i)
    for(i=0;i<size;i++)
        ret[i] = exp(av[i]);
    return new Vector(ret, size, false);
}

extern "C" __declspec(dllexport) void assign(void* _a, void* _b) {
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

extern "C" __declspec(dllexport) void* v_mult(void* _a, void* _b) {
    Vector* a = (Vector*)_a;
    Vector* b = (Vector*)_b;
    valuetype* av = a->values;
    valuetype* bv = b->values;
    sizetype size = a->size;
    valuetype* ret = new valuetype[size];
    sizetype i;
    #pragma omp parallel for shared(size, ret, av, bv) private(i)
    for(i=0;i<size;i++)
        ret[i] = av[i] * bv[i];
    return new Vector(ret, size, false);
}

extern "C" __declspec(dllexport) void* vc_add(void* _a, valuetype b) {
    Vector* a = (Vector*)_a;
    valuetype* av = a->values;
    sizetype size = a->size;
    valuetype* ret = new valuetype[size];
    sizetype i;
    #pragma omp parallel for shared(size, ret, av) private(i)
    for(i=0;i<size;i++)
        ret[i] = av[i] + b;
    return new Vector(ret, size, false);
}

extern "C" __declspec(dllexport) void* vc_sub(void* _a, valuetype b) {
    Vector* a = (Vector*)_a;
    valuetype* av = a->values;
    sizetype size = a->size;
    valuetype* ret = new valuetype[size];
    sizetype i;
    #pragma omp parallel for shared(size, ret, av) private(i)
    for(i=0;i<size;i++)
        ret[i] = av[i] - b;
    return new Vector(ret, size, false);
}


extern "C" __declspec(dllexport) void* cv_sub(void* _a, valuetype b) {
    Vector* a = (Vector*)_a;
    valuetype* av = a->values;
    sizetype size = a->size;
    valuetype* ret = new valuetype[size];
    sizetype i;
    #pragma omp parallel for shared(size, ret, av) private(i)
    for(i=0;i<size;i++)
        ret[i] = b-av[i];
    return new Vector(ret, size, false);
}


extern "C" __declspec(dllexport) void* vc_div(void* _a, valuetype b) {
    Vector* a = (Vector*)_a;
    valuetype* av = a->values;
    sizetype size = a->size;
    valuetype* ret = new valuetype[size];
    sizetype i;
    #pragma omp parallel for shared(size, ret, av) private(i)
    for(i=0;i<size;i++)
        ret[i] = av[i] / b;
    return new Vector(ret, size, false);
}


extern "C" __declspec(dllexport) void* cv_div(void* _a, valuetype b) {
    Vector* a = (Vector*)_a;
    valuetype* av = a->values;
    sizetype size = a->size;
    valuetype* ret = new valuetype[size];
    sizetype i;
    #pragma omp parallel for shared(size, ret, av) private(i)
    for(i=0;i<size;i++)
        ret[i] = b / av[i];
    return new Vector(ret, size, false);
}


extern "C" __declspec(dllexport) void* vc_pow(void* _a, valuetype b) {
    Vector* a = (Vector*)_a;
    valuetype* av = a->values;
    sizetype size = a->size;
    valuetype* ret = new valuetype[size];
    sizetype i;
    #pragma omp parallel for shared(size, ret, av) private(i)
    for(i=0;i<size;i++)
        ret[i] = pow(av[i], b);
    return new Vector(ret, size, false);
}


extern "C" __declspec(dllexport) void* cv_pow(void* _a, valuetype b) {
    Vector* a = (Vector*)_a;
    valuetype* av = a->values;
    sizetype size = a->size;
    valuetype* ret = new valuetype[size];
    sizetype i;
    #pragma omp parallel for shared(size, ret, av) private(i)
    for(i=0;i<size;i++)
        ret[i] = pow(b, av[i]);
    return new Vector(ret, size, false);
}

extern "C" __declspec(dllexport) void* vc_mult(void* _a, valuetype b) {
    Vector* a = (Vector*)_a;
    valuetype* av = a->values;
    sizetype size = a->size;
    valuetype* ret = new valuetype[size];
    sizetype i;
    #pragma omp parallel for shared(size, ret, av) private(i)
    for(i=0;i<size;i++)
        ret[i] = av[i] * b;
    return new Vector(ret, size, false);
}



extern "C" __declspec(dllexport) void* matrix(sizetype* x, sizetype* y, valuetype* values, sizetype entries, sizetype size){return new Matrix(x, y, values, entries, size); }
extern "C" __declspec(dllexport) void* vector(valuetype* values, sizetype size) {return new Vector(values, size);}
extern "C" __declspec(dllexport) valuetype get(void* vector, sizetype i) {return ((Vector*)vector)->values[i];}
extern "C" __declspec(dllexport) void set(void* vector, sizetype i, valuetype value){((Vector*)vector)->values[i] = value;}
extern "C" __declspec(dllexport) sizetype len(void* vector){return ((Vector*)vector)->size;}
extern "C" __declspec(dllexport) sizetype m_len(void* matrix){return ((Matrix*)matrix)->size;}

extern "C" __declspec(dllexport) void* repeat(valuetype value, sizetype size) {
    valuetype* ret = new valuetype[size];
    sizetype i;
    #pragma omp parallel for shared(size, ret) private(i)
    for(i=0;i<size;i++)
        ret[i] = value;
    return new Vector(ret, size, false);
}

extern "C" __declspec(dllexport) void* v_copy(void* _vector) {
    Vector* vector = (Vector*)_vector;
    return new Vector(vector->values, vector->size);
}

extern "C" __declspec(dllexport) void* get_values(void* _matrix) {
    Matrix* matrix = (Matrix*)_matrix;
    return new Vector(matrix->values, matrix->entries, false);
}

extern "C" __declspec(dllexport) void* get_rows(void* _matrix) {
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

extern "C" __declspec(dllexport) void* get_cols(void* _matrix) {
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
extern "C" __declspec(dllexport) void set_number_of_threads(int threads) {
    omp_set_dynamic(0);
    omp_set_num_threads(threads);
}