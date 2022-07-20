import ctypes
from ctypes import *
from array import array
import os
import numpy as np


sizetype = c_longlong


def load_matvec(matvec_libfile, threads=None):
    global matvec_lib
    matvec_lib = CDLL(matvec_libfile)

    for constructor in [matvec_lib.vector, matvec_lib.matrix]:
        constructor.restype = c_void_p
    matvec_lib.vector.argtypes = [ctypes.py_object, sizetype]
    matvec_lib.matrix.argtypes = [ctypes.py_object, ctypes.py_object, ctypes.py_object, sizetype, sizetype]
    matvec_lib.free_matrix.argtypes = [c_void_p]
    matvec_lib.free_vector.argtypes = [c_void_p]

    matvec_lib.get.argtypes = [c_void_p, sizetype]
    matvec_lib.get.restype = c_double
    matvec_lib.set.argtypes = [c_void_p, sizetype, c_double]

    for operation in matvec_lib.len, matvec_lib.m_len:
        operation.argtypes = [c_void_p]
        operation.restype = sizetype

    matvec_lib.dot.argtypes = [c_void_p, c_void_p]
    matvec_lib.dot.restype = c_double
    matvec_lib.assign.argtypes = [c_void_p, c_void_p]
    matvec_lib.repeat.argtypes = [c_double, sizetype]
    matvec_lib.repeat.restype = c_void_p

    for operation in [matvec_lib.get_values, matvec_lib.get_rows, matvec_lib.get_cols]:
        operation.argtypes = [c_void_p]
        operation.restype = c_void_p
    matvec_lib.set_number_of_threads.argtypes = [c_int]
    matvec_lib.v_copy.argtypes = [c_void_p]
    matvec_lib.v_copy.restype = c_void_p

    for operation in [matvec_lib.add, matvec_lib.sub, matvec_lib.v_mult, matvec_lib.v_pow, matvec_lib.v_div, matvec_lib.multiply, matvec_lib.rmultiply]:
        operation.argtypes = [c_void_p, c_void_p]
        operation.restype = c_void_p

    for operation in [matvec_lib.vc_add, matvec_lib.vc_sub, matvec_lib.cv_sub, matvec_lib.vc_mult, matvec_lib.vc_pow, matvec_lib.cv_pow, matvec_lib.vc_div, matvec_lib.cv_div]:
        operation.argtypes = [c_void_p, c_double]
        operation.restype = c_void_p

    for operation in [matvec_lib.v_log, matvec_lib.v_exp, matvec_lib.v_abs, matvec_lib.m_sum_rows, matvec_lib.m_sum_cols, matvec_lib.transpose]:
        operation.argtypes = [c_void_p]
        operation.restype = c_void_p

    for operation in [matvec_lib.v_sum, matvec_lib.v_max, matvec_lib.v_min, matvec_lib.v_mean, matvec_lib.m_sum_all]:
        operation.argtypes = [c_void_p]
        operation.restype = c_double

    set_number_of_threads(os.cpu_count() if threads is None else threads)


def clear():
    matvec_lib.clear()


def set_number_of_threads(threads):
    matvec_lib.set_number_of_threads(threads)


def repeat(value, times):
    ret = Vector()
    ret.data = matvec_lib.repeat(value, times)
    return ret


def log(vec):
    ret = Vector()
    ret.data = matvec_lib.log(vec.data)
    return ret


def exp(vec):
    ret = Vector()
    ret.data = matvec_lib.exp(vec.data)
    return ret


def abs(vec):
    ret = Vector()
    ret.data = matvec_lib.v_abs(vec.data)
    return ret


def sum(vec, axis=None):
    return vec.sum() if axis is None else vec.sum(axis)


def min(vec):
    return vec.min()


def max(vec):
    return vec.max()


def mean(vec, axis=None):
    return vec.sum(axis) / len(vec)


def dot(a, b):
    return matvec_lib.dot(a.data, b.data)


def multiply(matrix, vec):
    ret = Vector()
    ret.data = matvec_lib.multiply(matrix.data, vec.data)
    return ret


class Matrix(object):
    def __init__(self, x, y, values, size):
        if size > 0:
            x = np.array(x, dtype=np.longlong)
            y = np.array(y, dtype=np.longlong)
            values = np.array(values, dtype="double")
            entries = values.shape[0]
            self.data = matvec_lib.matrix(x, y, values, entries, size)

    def transpose(self):
        ret = Matrix([], [], [], -1)
        ret.data = matvec_lib.transpose(self.data)
        return ret

    def get_rows(self):
        ret = Vector()
        ret.data = matvec_lib.get_rows(self.data)
        return ret

    def get_cols(self):
        ret = Vector()
        ret.data = matvec_lib.get_cols(self.data)
        return ret

    def get_values(self):
        ret = Vector()
        ret.data = matvec_lib.get_values(self.data)
        return ret

    def sum(self, axis=None):
        if axis is None:
            return matvec_lib.m_sum_all(self.data)
        ret = Vector()
        ret.data = matvec_lib.m_sum_rows(self.data) if axis == 0 else matvec_lib.m_sum_cols(self.data)
        return ret

    def __mul__(self, other):
        ret = Vector()
        ret.data = matvec_lib.multiply(self.data, other.data)
        return ret

    def __rmul__(self, other):
        ret = Vector()
        ret.data = matvec_lib.rmultiply(other.data, self.data)
        return ret

    def __len__(self):
        return matvec_lib.m_len(self.data)

    def __del__(self):
        matvec_lib.free_matrix(self.data)


class Vector(object):
    def __init__(self, x=None):
        if x is not None:
            x = np.array(x, dtype="double")
            self.data = matvec_lib.vector(x, x.shape[0])

    def assign(self, other):
        matvec_lib.assign(self.data, other.data)

    def __getitem__(self, i):
        return matvec_lib.get(self.data, i)

    def __setitem__(self, i, value):
        matvec_lib.set(self.data, i, value)

    def __add__(self, other):
        ret = Vector()
        if isinstance(other, float) or isinstance(other, int):
            ret.data = matvec_lib.vc_add(self.data, other)
        else:
            ret.data = matvec_lib.add(self.data, other.data)
        return ret

    def __neg__(self):
        ret = Vector()
        ret.data = matvec_lib.cv_sub(self.data, 0.)
        return ret

    def __radd__(self, other):
        ret = Vector()
        if isinstance(other, float) or isinstance(other, int):
            ret.data = matvec_lib.vc_add(self.data, other)
        else:
            ret.data = matvec_lib.add(self.data, other.data)
        return ret

    def __sub__(self, other):
        ret = Vector()
        if isinstance(other, float) or isinstance(other, int):
            ret.data = matvec_lib.vc_sub(self.data, other)
        else:
            ret.data = matvec_lib.sub(self.data, other.data)
        return ret

    def __rsub__(self, other):
        ret = Vector()
        if isinstance(other, float) or isinstance(other, int):
            ret.data = matvec_lib.cv_sub(self.data, other)
        else:
            ret.data = matvec_lib.sub(other.data, self.data)
        return ret

    def __truediv__(self, other):
        ret = Vector()
        if isinstance(other, float) or isinstance(other, int):
            ret.data = matvec_lib.vc_div(self.data, other)
        else:
            ret.data = matvec_lib.v_div(self.data, other.data)
        return ret

    def __rtruediv__(self, other):
        ret = Vector()
        if isinstance(other, float) or isinstance(other, int):
            ret.data = matvec_lib.cv_div(self.data, other)
        else:
            ret.data = matvec_lib.v_div(other.data, self.data)
        return ret

    def __pow__(self, other):
        ret = Vector()
        if isinstance(other, float) or isinstance(other, int):
            ret.data = matvec_lib.vc_pow(self.data, other)
        else:
            ret.data = matvec_lib.v_pow(self.data, other.data)
        return ret

    def __rpow__(self, other):
        ret = Vector()
        if isinstance(other, float) or isinstance(other, int):
            ret.data = matvec_lib.cv_pow(self.data, other)
        else:
            ret.data = matvec_lib.v_pow(other.data, self.data)
        return ret

    def __len__(self):
        return matvec_lib.len(self.data)

    def __mul__(self, other):
        ret = Vector()
        if isinstance(other, Matrix):
            ret.data = matvec_lib.rmultiply(other.data, self.data)
        elif isinstance(other, float) or isinstance(other, int):
            ret.data = matvec_lib.vc_mult(self.data, other)
        else:
            ret.data = matvec_lib.v_mult(self.data, other.data)
        return ret

    def __rmul__(self, other):
        ret = Vector()
        if isinstance(other, Matrix):
            ret.data = matvec_lib.multiply(other.data, self.data)
        elif isinstance(other, float) or isinstance(other, int):
            ret.data = matvec_lib.vc_mult(self.data, other)
        else:
            ret.data = matvec_lib.v_mult(self.data, other.data)
        return ret

    def __abs__(self):
        return abs(self)

    def __del__(self):
        matvec_lib.free_vector(self.data)

    def sum(self):
        return matvec_lib.v_sum(self.data)

    def mean(self):
        return matvec_lib.v_mean(self.data)

    def max(self):
        return matvec_lib.v_max(self.data)

    def min(self):
        return matvec_lib.v_min(self.data)

    def copy(self):
        ret = Vector()
        ret.data = matvec_lib.v_copy(self.data)
        return ret

    def __str__(self):
        return "["+(", ".join(str(self[i]) for i in range(len(self))))+"]"
