#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "napcas/tensor.h"

namespace py = pybind11;
using namespace napcas;

PYBIND11_MODULE(napcas, m) {
    py::class_<Tensor>(m, "Tensor")
        .def_static("ones", &Tensor::ones)
        .def_static("zeros", &Tensor::zeros)
        .def("shape", &Tensor::shape)
        .def("__add__", &Tensor::operator+)
        .def("__sub__", &Tensor::operator-)
        .def("__mul__", &Tensor::operator*)
        .def("__truediv__", &Tensor::operator/)
        .def("matmul", &Tensor::matmul)
        .def("clone", &Tensor::clone)
        .def("reshape", &Tensor::reshape)
        .def("to", &Tensor::to)
        .def("astype", &Tensor::astype)
        .def("requires_grad_", &Tensor::requires_grad_)
        .def("requires_grad", &Tensor::requires_grad)
        .def("detach", &Tensor::detach)
        .def("print_shape", &Tensor::print_shape)
        .def("print_summary", &Tensor::print_summary);
}
