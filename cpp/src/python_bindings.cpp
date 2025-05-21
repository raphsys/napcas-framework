#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "napcas/tensor.h"
#include "napcas/common.h"
#include "napcas/module.h"
#include "napcas/autograd.h"
#include "napcas/grad_fn.h"
#include "napcas/device.h"

namespace py = pybind11;
using namespace napcas;

PYBIND11_MODULE(_napcas, m) {
    m.doc() = "napcas C++ backend";

    // --- DeviceType enum ---
    py::enum_<DeviceType>(m, "DeviceType")
        .value("CPU",  DeviceType::CPU)
        .value("CUDA", DeviceType::CUDA)
        .export_values();

    // --- DType enum ---
    py::enum_<DType>(m, "DType")
        .value("Float32", DType::Float32)
        .value("Int32",   DType::Int32)
        .export_values();

    // --- Device struct ---
    py::class_<Device>(m, "Device")
        .def(py::init<DeviceType, int>(),
             py::arg("type")  = DeviceType::CPU,
             py::arg("index") = 0)
        .def_readwrite("type",  &Device::type)
        .def_readwrite("index", &Device::index)
        .def("to_string",       &Device::to_string)
        .def("__str__",         &Device::to_string)
        ;

    // --- Tensor ---
    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
        // constructors
        .def(py::init<>())
        // static factories
        .def_static("ones",  &Tensor::ones,
             py::arg("shape"), py::arg("dtype") = DType::Float32, py::arg("device") = Device{DeviceType::CPU,0})
        .def_static("zeros", &Tensor::zeros,
             py::arg("shape"), py::arg("dtype") = DType::Float32, py::arg("device") = Device{DeviceType::CPU,0})
        // metadata
        .def("shape",        &Tensor::shape)
        .def("numel",        &Tensor::numel)
        .def("is_contiguous", &Tensor::is_contiguous)
        .def("dtype",        &Tensor::dtype)
        .def("device",       &Tensor::device)
        // basic ops
        .def("__add__",      &Tensor::operator+)
        .def("__sub__",      &Tensor::operator-)
        .def("__mul__",      &Tensor::operator*)
        .def("__truediv__",  &Tensor::operator/)
        .def("matmul",       &Tensor::matmul)
        // transforms
        .def("clone",        &Tensor::clone)
        .def("detach",       &Tensor::detach)
        .def("reshape",      &Tensor::reshape)
        .def("view",         &Tensor::view)
        .def("to",           &Tensor::to)
        .def("astype",       &Tensor::astype)
        // autograd interface
        .def("requires_grad_", &Tensor::requires_grad_)
        .def("requires_grad",  &Tensor::requires_grad)
        // disambiguate overloaded grad()
        .def("grad", 
             static_cast<Tensor& (Tensor::*)()>(&Tensor::grad),
             "Mutable grad")
        .def("grad_", 
             static_cast<const Tensor& (Tensor::*)() const>(&Tensor::grad),
             "Const‐version of grad")
        .def("backward", &Tensor::backward)
        // debugging
        .def("print_shape",   &Tensor::print_shape)
        .def("print_summary", &Tensor::print_summary)
        ;


    // --- Module ---
    py::class_<Module, std::shared_ptr<Module>>(m, "Module")
        .def(py::init<>())
        .def("register_parameter", &Module::register_parameter,
             py::arg("name"), py::arg("tensor"))
        .def("register_module",    &Module::register_module,
             py::arg("name"), py::arg("module"))
        .def("parameters",         &Module::parameters)
        .def("modules",            &Module::modules)
        .def("state_dict",         &Module::state_dict)
        .def("load_state_dict",    &Module::load_state_dict)
        ;

    // --- Autograd ---
    py::class_<Autograd, std::shared_ptr<Autograd>>(m, "Autograd")
        .def(py::init<>())
        .def("backward", &Autograd::backward,
             py::arg("tensor"), py::arg("retain_graph") = false)
        ;
}

