// python_bindings.cpp — pybind11 bindings for NAPCAS
// -----------------------------------------------------------------------------
//  * Build the `_napcas` extension exposing core C++ classes to Python.
//  * Provides user‑friendly overloads where appropriate (e.g. forward(x) → Tensor).
// -----------------------------------------------------------------------------

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <memory>

#include "tensor.h"
#include "module.h"
#include "napcas.h"
#include "napca_sim.h"
#include "nncell.h"
#include "linear.h"
#include "conv2d.h"
#include "pooling.h"
#include "attention.h"
#include "activation.h"
#include "loss.h"
#include "optimizer.h"
#include "autograd.h"
#include "data_loader.h"
#include "mlp.h"
#include "rnn.h"
#include "transformer.h"
#include "gan.h"
#include "lstm.h"
#include "gru.h"
#include "visualization.h"

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(_napcas, m) {
    m.doc() = "Low-level C++ bindings for napcas";

    // Tensor
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<const std::vector<int>&, const std::vector<float>&>(),
             "shape"_a, "data"_a = std::vector<float>{})
        .def("size", &Tensor::size)
        .def("shape", &Tensor::shape)
        .def("data", py::overload_cast<>(&Tensor::data),
             py::return_value_policy::reference_internal)
        .def("fill", &Tensor::fill)
        .def("zero_grad", &Tensor::zero_grad)
        .def("reshape", &Tensor::reshape)
        .def("save", &Tensor::save)
        .def("load", &Tensor::load);

    // Module base
    py::class_<Module, std::shared_ptr<Module>>(m, "Module")
        .def("forward", &Module::forward)
        .def("backward", &Module::backward)
        .def("update", &Module::update)
        .def("get_weights", &Module::get_weights,
             py::return_value_policy::reference_internal)
        .def("get_grad_weights", &Module::get_grad_weights,
             py::return_value_policy::reference_internal)
        .def("set_weights", &Module::set_weights)
        .def("save", &Module::save)
        .def("load", &Module::load);

    // NAPCAS standalone
    py::class_<NAPCAS>(m, "NAPCAS")
        .def(py::init<int, int>(), "input_dim"_a, "hidden_dim"_a)
        .def("forward", &NAPCAS::forward)
        .def("backward", &NAPCAS::backward)
        .def("update", &NAPCAS::update)
        .def("set_weights", &NAPCAS::set_weights)
        .def("save", &NAPCAS::save)
        .def("load", &NAPCAS::load);

    // NAPCA_Sim
    py::class_<NAPCA_Sim, Module, std::shared_ptr<NAPCA_Sim>>(m, "NAPCA_Sim")
        .def(py::init<int, int, float, float>(),
             "in_dim"_a, "out_dim"_a, "alpha"_a, "beta"_a)
        .def("forward", &NAPCA_Sim::forward)
        .def("backward", &NAPCA_Sim::backward)
        .def("update", &NAPCA_Sim::update)
        .def("compute_path_similarity", &NAPCA_Sim::compute_path_similarity)
        .def("update_weights_conditionally", &NAPCA_Sim::update_weights_conditionally)
        .def("prune_connections", &NAPCA_Sim::prune_connections)
        .def("get_weights", &NAPCA_Sim::get_weights,
             py::return_value_policy::reference_internal)
        .def("get_grad_weights", &NAPCA_Sim::get_grad_weights,
             py::return_value_policy::reference_internal)
        .def("set_weights", &NAPCA_Sim::set_weights)
        .def("save", &NAPCA_Sim::save)
        .def("load", &NAPCA_Sim::load);

    // NNCel with Pythonic forward
    py::class_<NNCel, Module, std::shared_ptr<NNCel>>(m, "NNCel")
        .def(py::init<int, int>(), "in_features"_a, "out_features"_a)
        // low-level
        .def("forward_out", &NNCel::forward, "x"_a, "y"_a)
        // high-level
        .def("forward", [](NNCel &self, const Tensor &x) {
            auto in_shape = x.shape();
            int batch = !in_shape.empty() ? in_shape[0] : 1;
            int out_feat = self.get_weights().shape()[0];
            Tensor y(std::vector<int>{batch, out_feat});
            self.forward(const_cast<Tensor&>(x), y);
            return y;
        }, "x"_a)
        .def("backward", &NNCel::backward)
        .def("update", &NNCel::update)
        .def("get_weights", &NNCel::get_weights,
             py::return_value_policy::reference_internal)
        .def("get_grad_weights", &NNCel::get_grad_weights,
             py::return_value_policy::reference_internal)
        .def("set_weights", &NNCel::set_weights)
        .def("save", &NNCel::save)
        .def("load", &NNCel::load);

    // Linear with Pythonic forward
    py::class_<Linear, Module, std::shared_ptr<Linear>>(m, "Linear")
        .def(py::init<int, int>(), "in_features"_a, "out_features"_a)
        // low-level
        .def("forward_out", &Linear::forward, "x"_a, "y"_a)
        // high-level
        .def("forward", [](Linear &self, const Tensor &x) {
            auto in_shape = x.shape();
            int batch = !in_shape.empty() ? in_shape[0] : 1;
            auto w = self.get_weights();
            int out_feat = !w.shape().empty() ? w.shape()[0] : 0;
            Tensor y(std::vector<int>{batch, out_feat});
            self.forward(const_cast<Tensor&>(x), y);
            return y;
        }, "x"_a)
        .def("backward", &Linear::backward)
        .def("update", &Linear::update)
        .def("get_weights", &Linear::get_weights,
             py::return_value_policy::reference_internal)
        .def("get_grad_weights", &Linear::get_grad_weights,
             py::return_value_policy::reference_internal)
        .def("set_weights", &Linear::set_weights)
        .def("save", &Linear::save)
        .def("load", &Linear::load);

    // Conv2d
    py::class_<Conv2d, Module, std::shared_ptr<Conv2d>>(m, "Conv2d")
        .def(py::init<int, int, int, int, int>(),
             "in_channels"_a, "out_channels"_a,
             "kernel_size"_a, "stride"_a = 1, "padding"_a = 0)
        .def("forward", &Conv2d::forward)
        .def("backward", &Conv2d::backward)
        .def("update", &Conv2d::update)
        .def("get_weights", &Conv2d::get_weights,
             py::return_value_policy::reference_internal)
        .def("get_grad_weights", &Conv2d::get_grad_weights,
             py::return_value_policy::reference_internal)
        .def("set_weights", &Conv2d::set_weights)
        .def("save", &Conv2d::save)
        .def("load", &Conv2d::load);

    // MaxPool2d
    py::class_<MaxPool2d, Module, std::shared_ptr<MaxPool2d>>(m, "MaxPool2d")
        .def(py::init<int, int>(), "kernel_size"_a, "stride"_a);

    // Activations
    py::class_<ReLU, Module, std::shared_ptr<ReLU>>(m, "ReLU").def(py::init<>());
    py::class_<Sigmoid, Module, std::shared_ptr<Sigmoid>>(m, "Sigmoid").def(py::init<>());
    py::class_<Tanh, Module, std::shared_ptr<Tanh>>(m, "Tanh").def(py::init<>());

    // Losses
    py::class_<MSELoss>(m, "MSELoss")
        .def(py::init<>())
        .def("forward", &MSELoss::forward)
        .def("backward", &MSELoss::backward);
    py::class_<CrossEntropyLoss>(m, "CrossEntropyLoss")
        .def(py::init<>())
        .def("forward", &CrossEntropyLoss::forward)
        .def("backward", &CrossEntropyLoss::backward);

    // Optimizers
    py::class_<SGD>(m, "SGD")
        .def(py::init<const std::vector<std::shared_ptr<Module>>&,
                      float>(),
             "modules"_a, "lr"_a = 0.01f)
        .def("step", &SGD::step);
    py::class_<Adam>(m, "Adam")
        .def(py::init<const std::vector<std::shared_ptr<Module>>&,
                      float, float, float, float>(),
             "modules"_a, "lr"_a = 0.001f,
             "beta1"_a = 0.9f, "beta2"_a = 0.999f,
             "epsilon"_a = 1e-8f)
        .def("step", &Adam::step);

    // Autograd
    py::class_<Autograd>(m, "Autograd")
        .def(py::init<>())
        .def("zero_grad", &Autograd::zero_grad)
        .def("backward", &Autograd::backward);

    // DataLoader
    py::class_<DataLoader>(m, "DataLoader")
        .def(py::init<const std::string&, int, bool>(),
             "dataset_path"_a, "batch_size"_a, "augment"_a = false)
        .def("next", &DataLoader::next);

    // MLP
    py::class_<MLP, Module, std::shared_ptr<MLP>>(m, "MLP")
        .def(py::init<const std::vector<int>&, const std::string&>(),
             "layers"_a, "activation"_a = "relu")
        .def("forward", &MLP::forward)
        .def("backward", &MLP::backward)
        .def("update", &MLP::update)
        .def("get_weights", &MLP::get_weights)
        .def("get_grad_weights", &MLP::get_grad_weights)
        .def("set_weights", &MLP::set_weights)
        .def("save", &MLP::save)
        .def("load", &MLP::load);

    // RNN
    py::class_<RNN, Module, std::shared_ptr<RNN>>(m, "RNN")
        .def(py::init<int, int, int>(),
             "input_size"_a, "hidden_size"_a, "num_layers"_a)
        .def("forward", &RNN::forward)
        .def("backward", &RNN::backward)
        .def("update", &RNN::update)
        .def("get_weights", &RNN::get_weights)
        .def("get_grad_weights", &RNN::get_grad_weights)
        .def("set_weights", &RNN::set_weights)
        .def("save", &RNN::save)
        .def("load", &RNN::load);

    // Transformer
    py::class_<Transformer, Module, std::shared_ptr<Transformer>>(m, "Transformer")
        .def(py::init<int, int, int, int>(),
             "d_model"_a, "num_heads"_a, "num_layers"_a, "d_ff"_a)
        .def("forward", &Transformer::forward)
        .def("backward", &Transformer::backward)
        .def("update", &Transformer::update)
        .def("get_weights", &Transformer::get_weights)
        .def("get_grad_weights", &Transformer::get_grad_weights)
        .def("set_weights", &Transformer::set_weights)
        .def("save", &Transformer::save)
        .def("load", &Transformer::load);

    // GAN
    py::class_<GAN, Module, std::shared_ptr<GAN>>(m, "GAN")
        .def(py::init<const std::vector<int>&, const std::vector<int>&>(),
             "generator_layers"_a, "discriminator_layers"_a)
        .def("forward", &GAN::forward)
        .def("backward", &GAN::backward)
        .def("train_step", &GAN::train_step)
        .def("update", &GAN::update)
        .def("get_weights", &GAN::get_weights)
        .def("get_grad_weights", &GAN::get_grad_weights)
        .def("set_weights", &GAN::set_weights)
        .def("save", &GAN::save)
        .def("load", &GAN::load);

    // LSTM
    py::class_<LSTM, Module, std::shared_ptr<LSTM>>(m, "LSTM")
        .def(py::init<int, int, int>(),
             "input_size"_a, "hidden_size"_a, "num_layers"_a)
        .def("forward", &LSTM::forward)
        .def("backward", &LSTM::backward)
        .def("update", &LSTM::update)
        .def("get_weights", &LSTM::get_weights)
        .def("get_grad_weights", &LSTM::get_grad_weights)
        .def("set_weights", &LSTM::set_weights)
        .def("save", &LSTM::save)
        .def("load", &LSTM::load);

    // GRU
    py::class_<GRU, Module, std::shared_ptr<GRU>>(m, "GRU")
        .def(py::init<int, int, int>(),
             "input_size"_a, "hidden_size"_a, "num_layers"_a)
        .def("forward", &GRU::forward)
        .def("backward", &GRU::backward)
        .def("update", &GRU::update)
        .def("get_weights", &GRU::get_weights)
        .def("get_grad_weights", &GRU::get_grad_weights)
        .def("set_weights", &GRU::set_weights)
        .def("save", &GRU::save)
        .def("load", &GRU::load);

    // Visualization
    py::class_<Visualization>(m, "Visualization")
        .def(py::init<>())
        .def("plot_tensor", &Visualization::plot_tensor)
        .def("log_to_tensorboard", &Visualization::log_to_tensorboard)
        .def("plot_training_curves", &Visualization::plot_training_curves);
}

