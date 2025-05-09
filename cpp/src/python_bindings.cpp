#include "linear.h"
#include "conv2d.h"
#include "activation.h"
#include "loss.h"
#include "optimizer.h"
#include "tensor.h"
#include "data_loader.h"
#include "nncell.h"
#include "napcas.h"
#include "autograd.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

class PyModule : public Module {
public:
    using Module::Module;

    void forward(Tensor& input, Tensor& output) override {
        PYBIND11_OVERRIDE_PURE(
            void,       // type de retour
            Module,     // classe de base
            forward,    // nom de la méthode
            input, output
        );
    }

    void backward(Tensor& grad_output, Tensor& grad_input) override {
        PYBIND11_OVERRIDE_PURE(void, Module, backward, grad_output, grad_input);
    }

    void update(float lr) override {
        PYBIND11_OVERRIDE_PURE(void, Module, update, lr);
    }

    Tensor& get_weights() override {
        PYBIND11_OVERRIDE_PURE(Tensor&, Module, get_weights);
    }

    Tensor& get_grad_weights() override {
        PYBIND11_OVERRIDE_PURE(Tensor&, Module, get_grad_weights);
    }

    void set_weights(const Tensor& weights) override {
        PYBIND11_OVERRIDE_PURE(void, Module, set_weights, weights);
    }
};


PYBIND11_MODULE(_napcas, m) {
    m.doc() = "NAPCAS: A lightweight deep learning framework";

    // Bind Tensor
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<std::vector<int>, std::vector<float>>())
        .def("shape", &Tensor::shape)
        .def("size", &Tensor::size)
        .def("data", (const std::vector<float>& (Tensor::*)() const) &Tensor::data, py::return_value_policy::reference)
        .def("zero_grad", &Tensor::zero_grad)
        .def("__getitem__", [](Tensor& t, int i) { return t[i]; })
        .def("__setitem__", [](Tensor& t, int i, float v) { t[i] = v; });

    // Bind Module
    py::class_<Module, PyModule, std::shared_ptr<Module>>(m, "Module")
        .def("forward", &Module::forward)
        .def("backward", &Module::backward)
        .def("update", &Module::update)
        .def("get_weights", &Module::get_weights, py::return_value_policy::reference)
        .def("get_grad_weights", &Module::get_grad_weights, py::return_value_policy::reference)
        .def("set_weights", &Module::set_weights);
    
    // Bind Linear
    py::class_<Linear, Module, std::shared_ptr<Linear>>(m, "Linear")
        .def(py::init<int, int>())
        .def("forward", &Linear::forward)
        .def("backward", &Linear::backward)
        .def("update", &Linear::update)
        .def("get_weights", &Linear::get_weights, py::return_value_policy::reference)
        .def("get_grad_weights", &Linear::get_grad_weights, py::return_value_policy::reference)
        .def("set_weights", &Linear::set_weights);

    // Bind Conv2d
    py::class_<Conv2d, Module, std::shared_ptr<Conv2d>>(m, "Conv2d")
        .def(py::init<int, int, int>())
        .def("forward", &Conv2d::forward)
        .def("backward", &Conv2d::backward)
        .def("update", &Conv2d::update)
        .def("get_weights", &Conv2d::get_weights, py::return_value_policy::reference)
        .def("get_grad_weights", &Conv2d::get_grad_weights, py::return_value_policy::reference)
        .def("set_weights", &Conv2d::set_weights);

    // Bind Activation Functions
    py::class_<ReLU, Module, std::shared_ptr<ReLU>>(m, "ReLU")
        .def(py::init<>())
        .def("forward", &ReLU::forward)
        .def("backward", &ReLU::backward);

    py::class_<Sigmoid, Module, std::shared_ptr<Sigmoid>>(m, "Sigmoid")
        .def(py::init<>())
        .def("forward", &Sigmoid::forward)
        .def("backward", &Sigmoid::backward);

    py::class_<Tanh, Module, std::shared_ptr<Tanh>>(m, "Tanh")
        .def(py::init<>())
        .def("forward", &Tanh::forward)
        .def("backward", &Tanh::backward);

    // Bind Loss Functions
    py::class_<MSELoss>(m, "MSELoss")
        .def(py::init<>())
        .def("forward", &MSELoss::forward)
        .def("backward", &MSELoss::backward);

    py::class_<CrossEntropyLoss>(m, "CrossEntropyLoss")
        .def(py::init<>())
        .def("forward", &CrossEntropyLoss::forward)
        .def("backward", &CrossEntropyLoss::backward);

    // Bind Optimizers
    py::class_<SGD>(m, "SGD")
        .def(py::init<float>())
        .def(py::init<std::vector<std::shared_ptr<Module>>, float>())
        .def("step", &SGD::step);

    py::class_<Adam>(m, "Adam")
        .def(py::init<float, float, float, float>())
        .def(py::init<std::vector<std::shared_ptr<Module>>, float, float, float, float>())
        .def("step", &Adam::step);

    // Bind DataLoader
    py::class_<DataLoader>(m, "DataLoader")
        .def(py::init<std::string, int>())
        .def("next", &DataLoader::next);

    // Bind NNCell
    py::class_<NNCell, Module, std::shared_ptr<NNCell>>(m, "NNCell")
        .def(py::init<int, int>())
        .def("forward", &NNCell::forward)
        .def("backward", &NNCell::backward)
        .def("update", &NNCell::update)
        .def("get_weights", &NNCell::get_weights, py::return_value_policy::reference)
        .def("get_grad_weights", &NNCell::get_grad_weights, py::return_value_policy::reference)
        .def("set_weights", &NNCell::set_weights);

    // Bind NAPCAS
    py::class_<NAPCAS, Module, std::shared_ptr<NAPCAS>>(m, "NAPCAS")
        .def(py::init<int, int>())
        .def("forward", &NAPCAS::forward)
        .def("backward", &NAPCAS::backward)
        .def("update", &NAPCAS::update)
        .def("get_weights", &NAPCAS::get_weights, py::return_value_policy::reference)
        .def("get_grad_weights", &NAPCAS::get_grad_weights, py::return_value_policy::reference)
        .def("set_weights", &NAPCAS::set_weights);
        
    // Bind Autograd
    py::class_<Autograd, std::shared_ptr<Autograd>>(m, "Autograd")
        .def(py::init<>());
        
}
