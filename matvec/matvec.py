from ctypes import *
from array import array
import os
import inspect

sizetype = c_longlong
valuetype = POINTER(c_double)
c_int_p = POINTER(sizetype)

def set_lib_file(libfile):
    global lib

    #libfile = r"build\lib.win-amd64-3.10\matvec\matvec\py.pyd"
    lib = CDLL(libfile)

    for constructor in [lib.vector, lib.matrix]:
        constructor.restype = c_void_p
    lib.vector.argtypes = [valuetype, sizetype]
    lib.matrix.argtypes = [c_int_p, c_int_p, valuetype, sizetype, sizetype]

    lib.get.argtypes = [c_void_p, sizetype]
    lib.get.restype = c_double
    lib.set.argtypes = [c_void_p, sizetype, c_double]
    lib.len.argtypes = [c_void_p]
    lib.len.restype = sizetype
    lib.v_sum.argtypes = [c_void_p]
    lib.v_sum.restype = c_double
    lib.dot.argtypes = [c_void_p, c_void_p]
    lib.dot.restype = c_double
    lib.assign.argtypes = [c_void_p, c_void_p]
    lib.get_values.argtypes = [c_void_p]
    lib.get_values.restype = c_void_p
    lib.set_number_of_threads.argtypes = [c_int]

    for operation in [lib.add, lib.sub, lib.v_mult, lib.v_pow, lib.v_div, lib.multiply]:
        operation.argtypes = [c_void_p, c_void_p]
        operation.restype = c_void_p

    for operation in [lib.vc_add, lib.vc_sub, lib.cv_sub, lib.vc_mult, lib.vc_pow, lib.cv_pow, lib.vc_div, lib.cv_div]:
        operation.argtypes = [c_void_p, c_double]
        operation.restype = c_void_p

    for operation in [lib.v_log, lib.v_exp, lib.v_abs]:
        operation.argtypes = [c_void_p]
        operation.restype = c_void_p

def set_number_of_threads(threads):
    lib.set_number_of_threads(threads)


def log(vec):
    ret = Vector()
    ret.data = lib.log(vec.data)
    return ret


def exp(vec):
    ret = Vector()
    ret.data = lib.exp(vec.data)
    return ret


def abs(vec):
    ret = Vector()
    ret.data = lib.v_abs(vec.data)
    return ret


def sum(vec):
    return lib.v_sum(vec.data)


def dot(a, b):
    return lib.dot(a.data, b.data)


def multiply(matrix, vec):
    ret = Vector()
    ret.data = lib.multiply(matrix.data, vec.data)
    return ret


class Matrix(object):
    def __init__(self, x, y, values, size):
        entries = len(values)
        x = (sizetype * entries).from_buffer(array('q', x))
        y = (sizetype * entries).from_buffer(array('q', y))
        values = (c_double * len(values)).from_buffer(array('d', values))
        self.data = lib.matrix(x, y, values, entries, size)

    def get_values(self):
        ret = Vector()
        ret.data = lib.get_values(self.data)
        return ret


class Vector(object):
    def __init__(self, x=None):
        if x is not None:
            self.data = lib.vector((c_double * len(x)).from_buffer(array('d', x)), len(x))

    def assign(self, other):
        lib.assign(self.data, other.data)

    def __getitem__(self, i):
        return lib.get(self.data, i)

    def __setitem__(self, i, value):
        lib.set(self.data, i, value)

    def __add__(self, other):
        ret = Vector()
        if isinstance(other, float) or isinstance(other, int):
            ret.data = lib.vc_add(self.data, other)
        else:
            ret.data = lib.add(self.data, other.data)
        return ret

    def __neg__(self):
        ret = Vector()
        ret.data = lib.cv_sub(self.data, 0.)
        return ret

    def __radd__(self, other):
        ret = Vector()
        if isinstance(other, float) or isinstance(other, int):
            ret.data = lib.vc_add(self.data, other)
        else:
            ret.data = lib.add(self.data, other.data)
        return ret

    def __sub__(self, other):
        ret = Vector()
        if isinstance(other, float) or isinstance(other, int):
            ret.data = lib.vc_sub(self.data, other)
        else:
            ret.data = lib.sub(self.data, other.data)
        return ret

    def __rsub__(self, other):
        ret = Vector()
        if isinstance(other, float) or isinstance(other, int):
            ret.data = lib.cv_sub(self.data, other)
        else:
            ret.data = lib.sub(other.data, self.data)
        return ret


    def __truediv__(self, other):
        ret = Vector()
        if isinstance(other, float) or isinstance(other, int):
            ret.data = lib.vc_div(self.data, other)
        else:
            ret.data = lib.v_div(self.data, other.data)
        return ret

    def __rtruediv__(self, other):
        ret = Vector()
        if isinstance(other, float) or isinstance(other, int):
            ret.data = lib.cv_div(self.data, other)
        else:
            ret.data = lib.v_div(other.data, self.data)
        return ret

    def __pow__(self, other):
        ret = Vector()
        if isinstance(other, float) or isinstance(other, int):
            ret.data = lib.vc_pow(self.data, other)
        else:
            ret.data = lib.v_pow(self.data, other.data)
        return ret

    def __rpow__(self, other):
        ret = Vector()
        if isinstance(other, float) or isinstance(other, int):
            ret.data = lib.cv_pow(self.data, other)
        else:
            ret.data = lib.v_pow(other.data, self.data)
        return ret

    def __len__(self):
        #print(lib.len(self.data))
        return lib.len(self.data)

    def __mul__(self, other):
        ret = Vector()
        if isinstance(other, float) or isinstance(other, int):
            ret.data = lib.vc_mult(self.data, other)
        else:
            ret.data = lib.v_mult(self.data, other.data)
        return ret

    def __rmul__(self, other):
        ret = Vector()
        if isinstance(other, float) or isinstance(other, int):
            ret.data = lib.vc_mult(self.data, other)
        else:
            ret.data = lib.v_mult(self.data, other.data)
        return ret

    def __abs__(self):
        return abs(self)

    def __str__(self):
        return "["+(", ".join(str(self[i]) for i in range(len(self))))+"]"
