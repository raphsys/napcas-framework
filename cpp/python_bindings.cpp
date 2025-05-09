#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "cpp/include/tensor.h"
#include "cpp/include/module.h"
#include "cpp/include/nncell.h"
#include "cpp/include/napcas.h"
#include "cpp/include/linear.h"
#include "cpp/include/conv2d.h"
#include "cpp/include/activation.h"
#include "cpp/include/loss.h"
#include "cpp/include/optimizer.h"
#include "cpp/include/data_loader.h"
#include "cpp/include/autograd.h"

namespace py = pybind11;

PYBIND11_MODULE(napcas, m) {
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<std::vector<int>>())
        .def(py::init<std::vector<int>, std::vector<float>>())
        .def("ndim", &Tensor::ndim)
        .def("shape", &Tensor::shape)
        .def("__getitem__", [](Tensor &t, int i) { return t[i]; })
        .def("reshape", &Tensor::reshape)
        .def("fill", &Tensor::fill)
        .def("zero_grad", &Tensor::zero_grad);

    py::class_<Module, std::shared_ptr<Module>>(m, "Module")
        .def("forward", &Module::forward)
        .def("backward", &Module::backward)
        .def("update", &Module::update);

    py::class_<NNCell, Module>(m, "NNCell")
        .def(py::init<int, int>())
        .def("forward", &NNCell::forward)
        .def("backward", &NNCell::backward)
        .def("update", &NNCell::update);

    py::class_<NAPCAS, Module>(m, "NAPCAS")
        .def(py::init<int, int>())
        .def("forward", &NAPCAS::forward)
        .def("backward", &NAPCAS::backward)
        .def("update", &NAPCAS::update);

    py::class_<Linear, Module>(m, "Linear")
        .def(py::init<int, int>())
        .def("forward", &Linear::forward)
        .def("backward", &Linear::backward)
        .def("update", &Linear::update);

    py::class_<Conv2d, Module>(m, "Conv2d")
        .def(py::init<int, int, int>())
        .def("forward", &Conv2d::forward)
        .def("backward", &Conv2d::backward)
        .def("update", &Conv2d::update);

    py::class_<ReLU, Module>(m, "ReLU")
        .def(py::init<>())
        .def("forward", &ReLU::forward)
        .def("backward", &ReLU::backward)
        .def("update", &ReLU::update);

    py::class_<Sigmoid, Module>(m, "Sigmoid")
        .def(py::init<>())
        .def("forward", &Sigmoid::forward)
        .def("backward", &Sigmoid::backward)
        .def("update", &Sigmoid::update);

    py::class_<Tanh, Module>(m, "Tanh")
        .def(py::init<>())
        .def("forward", &Tanh::forward)
        .def("backward", &Tanh::backward)
        .def("update", &Tanh::update);

    py::class_<MSELoss>(m, "MSELoss")
        .def(py::init<>())
        .def("forward", &MSELoss::forward)
        .def("backward", &MSELoss::backward);

    py::class_<CrossEntropyLoss>(m, "CrossEntropyLoss")
        .def(py::init<>())
        .def("forward", &CrossEntropyLoss::forward)
        .def("backward", &CrossEntropyLoss::backward);

    py::class_<SGD>(m, "SGD")
        .def(py::init<std::vector<std::shared_ptr<Module>>>())
        .def("step", &SGD::step);

    py::class_<Adam>(m, "Adam")
        .def(py::init<std::vector<std::shared_ptr<Module>>>())
        .def("step", &Adam::step);

    py::class_<DataLoader>(m, "DataLoader")
        .def(py::init<std::string, int>())
        .def("next", &DataLoader::next);

    py::class_<Autograd>(m, "Autograd")
        .def_static("grad", &Autograd::grad)
        .def_static("zero_grad", &Autograd::zero_grad);
}
