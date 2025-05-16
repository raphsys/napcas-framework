#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "napcas/tensor.h"
#include "napcas/module.h"

namespace py = pybind11;

// Bindings pour le module Python _napcas
PYBIND11_MODULE(_napcas, m) {
    m.doc() = "NAPCAS core Python bindings";

    // --- DType enum ---
    py::enum_<napcas::DType>(m, "DType")
        .value("Float32", napcas::DType::Float32)
        .value("Float64", napcas::DType::Float64)
        .export_values();

    // --- DeviceType enum ---
    py::enum_<napcas::DeviceType>(m, "DeviceType")
        .value("CPU",  napcas::DeviceType::CPU)
        .value("CUDA", napcas::DeviceType::CUDA)
        .export_values();

    // --- Device class ---
    py::class_<napcas::Device>(m, "Device")
        .def(py::init<napcas::DeviceType,int>(),
             py::arg("type"), py::arg("index") = 0)
        .def_readonly("type",  &napcas::Device::type)
        .def_readonly("index", &napcas::Device::index)
        ;

    // --- Tensor class ---
    py::class_<napcas::Tensor>(m, "Tensor")
        .def(py::init<const std::vector<std::size_t>&,
                      napcas::DType,
                      napcas::Device>(),
             py::arg("shape"),
             py::arg("dtype")  = napcas::DType::Float32,
             py::arg("device") = napcas::Device{})
        .def_property_readonly("shape",  &napcas::Tensor::shape)
        .def_property_readonly("dtype",  &napcas::Tensor::dtype)
        .def_property_readonly("device", &napcas::Tensor::device)
        .def("__repr__", [](const napcas::Tensor& t){
            return "<napcas.Tensor numel=" + std::to_string(t.numel()) + ">";
        })
        // Autograd
        .def("requires_grad_", &napcas::Tensor::requires_grad_, py::arg("flag") = true)
        .def_property_readonly("grad",
            [](napcas::Tensor &t) -> napcas::Tensor { return t.grad(); })
        .def("backward", &napcas::Tensor::backward)
        // Opérations et utilitaires
        .def("numel",      &napcas::Tensor::numel)
        .def("to",         &napcas::Tensor::to,        py::arg("device"))
        .def("astype",     &napcas::Tensor::astype,    py::arg("dtype"))
        .def("reshape",    &napcas::Tensor::reshape,   py::arg("shape"))
        .def("contiguous", &napcas::Tensor::contiguous)
        .def("__add__",     &napcas::Tensor::operator+)
        .def("__sub__",     &napcas::Tensor::operator-)
        .def("__mul__",     &napcas::Tensor::operator*)
        .def("__truediv__", &napcas::Tensor::operator/)
        .def("matmul",      &napcas::Tensor::matmul,  py::arg("other"))
        ;

    // --- Module class (abstraite) ---
    py::class_<napcas::Module, napcas::ModulePtr>(m, "Module")
        // PAS de .def(py::init<>()) : Module est abstraite (forward() = 0) :contentReference[oaicite:4]{index=4}:contentReference[oaicite:5]{index=5}
        .def("forward", &napcas::Module::forward, py::arg("input"))
        .def("train",   &napcas::Module::train)
        .def("eval",    &napcas::Module::eval)
        .def("parameters",
             [](napcas::Module &self) { return self.parameters(); })
        ;
}

