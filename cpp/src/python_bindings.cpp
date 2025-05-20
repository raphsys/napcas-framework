#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "napcas/tensor.h"
#include "napcas/common.h"
#include "napcas/module.h"
#include "napcas/autograd.h"

namespace py = pybind11;
using namespace napcas;

PYBIND11_MODULE(_napcas, m) {

    // --- Tensor ---
    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
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
             py::arg("type")=DeviceType::CPU,
             py::arg("index")=0)
        .def_readwrite("type",  &Device::type)
        .def_readwrite("index", &Device::index)
        .def("to_string", &Device::to_string)
        .def("__str__",     &Device::to_string);
        
    // Expose Module
    py::class_<Module, std::shared_ptr<Module>>(m, "Module")
        .def(py::init<>())
        .def("register_parameter", &Module::register_parameter,
             py::arg("name"), py::arg("tensor"))
        .def("register_module", &Module::register_module,
             py::arg("name"), py::arg("module"))
        .def("parameters", &Module::parameters)
        .def("modules", &Module::modules)
        .def("state_dict", &Module::state_dict)
        .def("load_state_dict", &Module::load_state_dict);


    // --- Autograd ---
    py::class_<Autograd, std::shared_ptr<Autograd>>(m, "Autograd")
        .def(py::init<>())
        .def("backward", &Autograd::backward,
             py::arg("tensor"), py::arg("retain_graph") = false);
                    
}

