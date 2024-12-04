#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "impl.h"

namespace py = pybind11;

PYBIND11_MODULE(matvec, m) {
    m.doc() = "Matrix and Vector types with methods";

    // Vector class
    py::class_<Vector>(m, "Vector")
        .def(py::init([](py::object values) {
            py::array array = py::array_t<valuetype>::ensure(values);
            if (!array)
                throw std::runtime_error("Expected a 1-dimensional array or list.");
            py::buffer_info info = array.request();
            if (info.ndim != 1)
                throw std::runtime_error("Expected a 1-dimensional array or list.");
            auto data_ptr = static_cast<valuetype*>(info.ptr);
            return new Vector(data_ptr, info.shape[0], true);
        }))
        .def("__getitem__", [](Vector &self, iteratortype i) { return get(&self, i); })
        .def("__getitem__", [](Vector &self, iteratortype i) { return get(&self, i); })
        .def("__setitem__", [](Vector &self, iteratortype i, valuetype value) { set(&self, i, value); })
        .def("__len__", [](Vector &self) { return self.size; })
        .def("__add__", [](Vector &self, Vector &other) { return (Vector *)add(&self, &other); })
        .def("__sub__", [](Vector &self, Vector &other) { return (Vector *)sub(&self, &other); })
        .def("__mul__", [](Vector &self, Vector &other) { return (Vector *)v_mult(&self, &other); })
        .def("__div__", [](Vector &self, Vector &other) { return (Vector *)v_div(&self, &other); })
        .def("__eq__", [](Vector &self, Vector &other) { return (Vector *)equals(&self, &other); })
        .def("__gt__", [](Vector &self, Vector &other) { return (Vector *)greater(&self, &other); })
        .def("__ge__", [](Vector &self, Vector &other) { return (Vector *)greater_eq(&self, &other); })
        .def("dot", [](Vector &self, Vector &other) { return dot(&self, &other); })
        .def("norm", [](Vector &self) { return v_norm(&self); })
        .def("sum", [](Vector &self) { return v_sum(&self); })
        .def("max", [](Vector &self) { return v_max(&self); })
        .def("min", [](Vector &self) { return v_min(&self); })
        .def("__str__", [](Vector &self) {
            std::ostringstream oss;
            oss << "[";
            for (size_t i = 0; i < self.size; ++i) {
                if (i)
                    oss << ", ";
                oss << self.values[i];
            }
            oss << "]";
            return oss.str(); // Convert the stream to a string
        })
        .def("__array__", [](Vector &self) {
            return py::array_t<valuetype>(self.size, self.values);
        })
        .def("__array__", [](Vector &self, bool copy) {
            if (copy) {
                py::array_t<valuetype> array(self.size);
                auto buf = array.request();
                std::memcpy(buf.ptr, self.values, self.size * sizeof(valuetype));
                return array;
            } else {
                return py::array_t<valuetype>(
                    self.size,                  // Shape (1D array with size elements)
                    self.values,                // Pointer to the data
                    py::cast(&self)             // Ensure the array keeps the Vector alive
                );
            }
        }, py::arg("copy") = true);

    // Matrix class
    py::class_<Matrix>(m, "Matrix")
        .def(py::init([](py::object x, py::object y, py::object values, sizetype size) {
            // Ensure inputs are numpy-compatible arrays
            py::array array_x = py::array_t<sizetype>::ensure(x);
            py::array array_y = py::array_t<sizetype>::ensure(y);
            py::array array_values = py::array_t<valuetype>::ensure(values);

            if (!array_x || !array_y || !array_values) {
                throw std::runtime_error("Inputs must be 1-dimensional arrays or lists.");
            }

            // Request buffer information
            py::buffer_info info_x = array_x.request();
            py::buffer_info info_y = array_y.request();
            py::buffer_info info_values = array_values.request();

            if (info_x.ndim != 1 || info_y.ndim != 1 || info_values.ndim != 1) {
                throw std::runtime_error("Expected 1-dimensional arrays or lists for x, y, and values.");
            }

            sizetype entries = info_values.shape[0];
            if (info_x.shape[0]!=entries)
                throw std::runtime_error("Mismatching size between x and values.");
            if (info_y.shape[0]!=entries)
                throw std::runtime_error("Mismatching size between y and values.");

            return new Matrix(
                static_cast<sizetype*>(info_x.ptr),
                static_cast<sizetype*>(info_y.ptr),
                static_cast<valuetype*>(info_values.ptr),
                entries,
                size,
                true);
        }))
        .def("__matmul__", [](Matrix &self, Vector &vec) { return (Vector *)multiply(&self, &vec); })
        .def("__rmatmul__", [](Matrix &self, Vector &vec) { return (Vector *)rmultiply(&self, &vec); })
        .def("T", [](Matrix &self) { return (Matrix *)transpose(&self); })
        .def("sum", [](Matrix &self) { return m_sum_all(&self); })
        .def("sum", [](Matrix &self, int axis) {
                if(axis==0)
                    return (Vector *)m_sum_rows(&self);
                return (Vector *)m_sum_cols(&self);
        }, py::arg("axis") = 0)
        .def("values", [](Matrix &self) { return (Vector *)get_values(&self); })
        .def("rows", [](Matrix &self) { return (Vector *)get_rows(&self); })
        .def("cols", [](Matrix &self) { return (Vector *)get_cols(&self); });

    // Module-level functions
    m.def("clear", &clear, "Clear reusable cache");
    m.def("repeat", [](valuetype value, sizetype size) {
        return (Vector *)repeat(value, size);
    }, "Create a vector by repeating a value");
    m.def("set_number_of_threads", &set_number_of_threads, "Set the number of OpenMP threads");
}
