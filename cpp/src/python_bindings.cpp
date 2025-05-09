#include <pybind11/pybind11.h>
#include "linear.h"
#include "conv2d.h"
#include "activation.h"
#include "loss.h"
#include "optimizer.h"
#include "data_loader.h"
#include "autograd.h"
#include "module.h"
#include "nncell.h"
#include "tensor.h"
#include "napcas.h"

namespace py = pybind11;

PYBIND11_MODULE(napcas, m) {
    py::class_<Linear>(m, "Linear")
        .def(py::init<int, int>())
        .def("forward", &Linear::forward)
        .def("backward", &Linear::backward)
        .def("update", &Linear::update);

    py::class_<Conv2d>(m, "Conv2d")
        .def(py::init<int, int, int>())
        .def("forward", &Conv2d::forward)
        .def("backward", &Conv2d::backward)
        .def("update", &Conv2d::update);

    py::class_<ReLU>(m, "ReLU")
        .def(py::init<>())
        .def("forward", &ReLU::forward)
        .def("backward", &ReLU::backward)
        .def("update", &ReLU::update);

    py::class_<Sigmoid>(m, "Sigmoid")
        .def(py::init<>())
        .def("forward", &Sigmoid::forward)
        .def("backward", &Sigmoid::backward)
        .def("update", &Sigmoid::update);

    py::class_<Tanh>(m, "Tanh")
        .def(py::init<>())
        .def("forward", &Tanh::forward)
        .def("backward", &Tanh::backward)
        .def("update", &Tanh::update);

    py::class_<MSELoss>(m, "MSELoss")
        .def("forward", &MSELoss::forward)
        .def("backward", &MSELoss::backward);

    py::class_<CrossEntropyLoss>(m, "CrossEntropyLoss")
        .def("forward", &CrossEntropyLoss::forward)
        .def("backward", &CrossEntropyLoss::backward);

    py::class_<SGD>(m, "SGD")
        .def(py::init<std::vector<Module*>>())
        .def("step", &SGD::step);

    py::class_<Adam>(m, "Adam")
        .def(py::init<std::vector<Module*>>())
        .def("step", &Adam::step);

    py::class_<DataLoader>(m, "DataLoader")
        .def(py::init<std::string, int>())
        .def("next", &DataLoader::next);

    py::class_<Autograd>(m, "Autograd")
        .def_static("grad", &Autograd::grad)
        .def_static("zero_grad", &Autograd::zero_grad);

    py::class_<NAPCAS>(m, "NAPCAS")
        .def(py::init<int, int>())
        .def("forward", &NAPCAS::forward)
        .def("backward", &NAPCAS::backward)
        .def("update", &NAPCAS::update);
}
