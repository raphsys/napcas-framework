#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <memory>

#include "tensor.h"
#include "module.h"
#include "linear.h"
#include "conv2d.h"
#include "activation.h"
#include "loss.h"
#include "optimizer.h"
#include "autograd.h"
#include "napcas.h"
#include "napca_sim.h"

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(napcas, m) {
    // Tensor
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<const std::vector<int>&, const std::vector<float>&>(),
             "shape"_a, "data"_a = std::vector<float>())
        .def("shape", &Tensor::shape)
        .def("data", py::overload_cast<>(&Tensor::data), py::return_value_policy::reference_internal)
        .def("size", &Tensor::size)
        .def("fill", &Tensor::fill)
        .def("zero_grad", &Tensor::zero_grad);

    // Module de base
    py::class_<Module, std::shared_ptr<Module>>(m, "Module")
        .def("forward", &Module::forward)
        .def("backward", &Module::backward)
        .def("update", &Module::update)
        .def("get_weights", &Module::get_weights, py::return_value_policy::reference_internal)
        .def("get_grad_weights", &Module::get_grad_weights, py::return_value_policy::reference_internal)
        .def("set_weights", &Module::set_weights);

    // Modules dérivés
    py::class_<Linear, Module, std::shared_ptr<Linear>>(m, "Linear")
        .def(py::init<int, int>(), "in_features"_a, "out_features"_a);

    py::class_<Conv2d, Module, std::shared_ptr<Conv2d>>(m, "Conv2d")
        .def(py::init<int, int, int>(), "in_channels"_a, "out_channels"_a, "kernel_size"_a);

    py::class_<ReLU, Module, std::shared_ptr<ReLU>>(m, "ReLU")
        .def(py::init<>());

    py::class_<Sigmoid, Module, std::shared_ptr<Sigmoid>>(m, "Sigmoid")
        .def(py::init<>());

    py::class_<Tanh, Module, std::shared_ptr<Tanh>>(m, "Tanh")
        .def(py::init<>());

    // NAPCAS
    py::class_<NAPCAS, Module, std::shared_ptr<NAPCAS>>(m, "NAPCAS")
        .def(py::init<int, int>(), "in_features"_a, "out_features"_a)
        .def("forward", &NAPCAS::forward)
        .def("backward", &NAPCAS::backward)
        .def("update", &NAPCAS::update)
        .def("get_weights", &NAPCAS::get_weights, py::return_value_policy::reference_internal)
        .def("get_grad_weights", &NAPCAS::get_grad_weights, py::return_value_policy::reference_internal)
        .def("set_weights", &NAPCAS::set_weights);

    // NAPCA_Sim
    py::class_<NAPCA_Sim, Module, std::shared_ptr<NAPCA_Sim>>(m, "NAPCA_Sim")
        .def(py::init<int, int, float, float>(),
             "in_features"_a, "out_features"_a, "alpha"_a = 0.6f, "threshold"_a = 0.5f)
        .def("forward", &NAPCA_Sim::forward)
        .def("backward", &NAPCA_Sim::backward)
        .def("update", &NAPCA_Sim::update)
        .def("get_weights", &NAPCA_Sim::get_weights, py::return_value_policy::reference_internal)
        .def("get_grad_weights", &NAPCA_Sim::get_grad_weights, py::return_value_policy::reference_internal)
        .def("set_weights", &NAPCA_Sim::set_weights)
        .def("compute_path_similarity", &NAPCA_Sim::compute_path_similarity)
        .def("update_weights_conditionally", &NAPCA_Sim::update_weights_conditionally)
        .def("prune_connections", &NAPCA_Sim::prune_connections);

    // Fonctions de perte
    py::class_<MSELoss>(m, "MSELoss")
        .def(py::init<>())
        .def("forward", &MSELoss::forward)
        .def("backward", &MSELoss::backward);

    py::class_<CrossEntropyLoss>(m, "CrossEntropyLoss")
        .def(py::init<>())
        .def("forward", &CrossEntropyLoss::forward)
        .def("backward", &CrossEntropyLoss::backward);

    // Optimiseurs
    py::class_<SGD>(m, "SGD")
        .def(py::init<std::vector<std::shared_ptr<Module>>, float>(), "modules"_a, "lr"_a = 0.01f)
        .def("step", &SGD::step);

    py::class_<Adam>(m, "Adam")
        .def(py::init<std::vector<std::shared_ptr<Module>>, float, float, float, float>(),
             "modules"_a, "lr"_a = 0.001f, "beta1"_a = 0.9f, "beta2"_a = 0.999f, "epsilon"_a = 1e-8f)
        .def("step", &Adam::step);

    // Autograd
    m.def("zero_grad", &Autograd::zero_grad, "Efface les gradients", "tensors"_a);
}

