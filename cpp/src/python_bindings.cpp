#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "napcas.h"
#include "napca_sim.h"
#include "nncell.h"
#include "linear.h"
#include "conv2d.h"
#include "pooling.h"
#include "activation.h"
#include "loss.h"
#include "optimizer.h"
#include "data_loader.h"
#include "mlp.h"
#include "rnn.h"
#include "transformer.h"
#include "gan.h"
#include "lstm.h"
#include "gru.h"
#include "visualization.h"
#include "autograd.h"

namespace py = pybind11;

PYBIND11_MODULE(napcas, m) {
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<const std::vector<int>&, const std::vector<float>&>())
        .def("size", &Tensor::size)
        .def("shape", &Tensor::shape)
        .def("data", py::overload_cast<>(&Tensor::data))
        .def("fill", &Tensor::fill)
        .def("zero_grad", &Tensor::zero_grad)
        .def("reshape", &Tensor::reshape)
        .def("save", &Tensor::save)
        .def("load", &Tensor::load);

    py::class_<Module, std::shared_ptr<Module>>(m, "Module")
        .def("forward", &Module::forward)
        .def("backward", &Module::backward)
        .def("update", &Module::update)
        .def("get_weights", &Module::get_weights)
        .def("get_grad_weights", &Module::get_grad_weights)
        .def("set_weights", &Module::set_weights)
        .def("save", &Module::save)
        .def("load", &Module::load);

    py::class_<NAPCAS, Module, std::shared_ptr<NAPCAS>>(m, "NAPCAS")
        .def(py::init<int, int>())
        .def("forward", &NAPCAS::forward)
        .def("backward", &NAPCAS::backward)
        .def("update", &NAPCAS::update)
        .def("get_weights", &NAPCAS::get_weights)
        .def("get_grad_weights", &NAPCAS::get_grad_weights)
        .def("set_weights", &NAPCAS::set_weights)
        .def("save", &NAPCAS::save)
        .def("load", &NAPCAS::load);

    py::class_<NAPCA_Sim, Module, std::shared_ptr<NAPCA_Sim>>(m, "NAPCA_Sim")
        .def(py::init<int, int, float, float>())
        .def("forward", &NAPCA_Sim::forward)
        .def("backward", &NAPCA_Sim::backward)
        .def("update", &NAPCA_Sim::update)
        .def("compute_path_similarity", &NAPCA_Sim::compute_path_similarity)
        .def("update_weights_conditionally", &NAPCA_Sim::update_weights_conditionally)
        .def("prune_connections", &NAPCA_Sim::prune_connections)
        .def("get_weights", &NAPCA_Sim::get_weights)
        .def("get_grad_weights", &NAPCA_Sim::get_grad_weights)
        .def("set_weights", &NAPCA_Sim::set_weights)
        .def("save", &NAPCA_Sim::save)
        .def("load", &NAPCA_Sim::load);

    py::class_<NNCel, Module, std::shared_ptr<NNCel>>(m, "NNCel")
        .def(py::init<int, int>())
        .def("forward", &NNCel::forward)
        .def("backward", &NNCel::backward)
        .def("update", &NNCel::update)
        .def("get_weights", &NNCel::get_weights)
        .def("get_grad_weights", &NNCel::get_grad_weights)
        .def("set_weights", &NNCel::set_weights)
        .def("save", &NNCel::save)
        .def("load", &NNCel::load);

    py::class_<Linear, Module, std::shared_ptr<Linear>>(m, "Linear")
        .def(py::init<int, int>())
        .def("forward", &Linear::forward)
        .def("backward", &Linear::backward)
        .def("update", &Linear::update)
        .def("get_weights", &Linear::get_weights)
        .def("get_grad_weights", &Linear::get_grad_weights)
        .def("set_weights", &Linear::set_weights)
        .def("save", &Linear::save)
        .def("load", &Linear::load);

    py::class_<Conv2d, Module, std::shared_ptr<Conv2d>>(m, "Conv2d")
        .def(py::init<int, int, int, int, int>())
        .def("forward", &Conv2d::forward)
        .def("backward", &Conv2d::backward)
        .def("update", &Conv2d::update)
        .def("get_weights", &Conv2d::get_weights)
        .def("get_grad_weights", &Conv2d::get_grad_weights)
        .def("set_weights", &Conv2d::set_weights)
        .def("save", &Conv2d::save)
        .def("load", &Conv2d::load);

    py::class_<MaxPool2d, Module, std::shared_ptr<MaxPool2d>>(m, "MaxPool2d")
        .def(py::init<int, int>())
        .def("forward", &MaxPool2d::forward)
        .def("backward", &MaxPool2d::backward)
        .def("update", &MaxPool2d::update)
        .def("get_weights", &MaxPool2d::get_weights)
        .def("get_grad_weights", &MaxPool2d::get_grad_weights)
        .def("set_weights", &MaxPool2d::set_weights)
        .def("save", &MaxPool2d::save)
        .def("load", &MaxPool2d::load);

    py::class_<ReLU, Module, std::shared_ptr<ReLU>>(m, "ReLU")
        .def(py::init<>())
        .def("forward", &ReLU::forward)
        .def("backward", &ReLU::backward)
        .def("update", &ReLU::update)
        .def("get_weights", &ReLU::get_weights)
        .def("get_grad_weights", &ReLU::get_grad_weights)
        .def("set_weights", &ReLU::set_weights)
        .def("save", &ReLU::save)
        .def("load", &ReLU::load);

    py::class_<Sigmoid, Module, std::shared_ptr<Sigmoid>>(m, "Sigmoid")
        .def(py::init<>())
        .def("forward", &Sigmoid::forward)
        .def("backward", &Sigmoid::backward)
        .def("update", &Sigmoid::update)
        .def("get_weights", &Sigmoid::get_weights)
        .def("get_grad_weights", &Sigmoid::get_grad_weights)
        .def("set_weights", &Sigmoid::set_weights)
        .def("save", &Sigmoid::save)
        .def("load", &Sigmoid::load);

    py::class_<Tanh, Module, std::shared_ptr<Tanh>>(m, "Tanh")
        .def(py::init<>())
        .def("forward", &Tanh::forward)
        .def("backward", &Tanh::backward)
        .def("update", &Tanh::update)
        .def("get_weights", &Tanh::get_weights)
        .def("get_grad_weights", &Tanh::get_grad_weights)
        .def("set_weights", &Tanh::set_weights)
        .def("save", &Tanh::save)
        .def("load", &Tanh::load);

    py::class_<MSELoss>(m, "MSELoss")
        .def(py::init<>())
        .def("forward", &MSELoss::forward)
        .def("backward", &MSELoss::backward);

    py::class_<CrossEntropyLoss>(m, "CrossEntropyLoss")
        .def(py::init<>())
        .def("forward", &CrossEntropyLoss::forward)
        .def("backward", &CrossEntropyLoss::backward);

    py::class_<SGD>(m, "SGD")
        .def(py::init<float>())
        .def(py::init<const std::vector<std::shared_ptr<Module>>&, float>())
        .def("step", &SGD::step)
        .def("save", &SGD::save)
        .def("load", &SGD::load);

    py::class_<Adam>(m, "Adam")
        .def(py::init<float, float, float, float>())
        .def(py::init<const std::vector<std::shared_ptr<Module>>&, float, float, float, float>())
        .def("step", &Adam::step)
        .def("save", &Adam::save)
        .def("load", &Adam::load);

    py::class_<DataLoader>(m, "DataLoader")
        .def(py::init<const std::string&, int, bool>())
        .def("next", &DataLoader::next);

    py::class_<MLP, Module, std::shared_ptr<MLP>>(m, "MLP")
        .def(py::init<const std::vector<int>&, const std::string&>())
        .def("forward", &MLP::forward)
        .def("backward", &MLP::backward)
        .def("update", &MLP::update)
        .def("get_weights", &MLP::get_weights)
        .def("get_grad_weights", &MLP::get_grad_weights)
        .def("set_weights", &MLP::set_weights)
        .def("save", &MLP::save)
        .def("load", &MLP::load);

    py::class_<RNN, Module, std::shared_ptr<RNN>>(m, "RNN")
        .def(py::init<int, int, int>())
        .def("forward", &RNN::forward)
        .def("backward", &RNN::backward)
        .def("update", &RNN::update)
        .def("get_weights", &RNN::get_weights)
        .def("get_grad_weights", &RNN::get_grad_weights)
        .def("set_weights", &RNN::set_weights)
        .def("save", &RNN::save)
        .def("load", &RNN::load);

    py::class_<Transformer, Module, std::shared_ptr<Transformer>>(m, "Transformer")
        .def(py::init<int, int, int, int>())
        .def("forward", &Transformer::forward)
        .def("backward", &Transformer::backward)
        .def("update", &Transformer::update)
        .def("get_weights", &Transformer::get_weights)
        .def("get_grad_weights", &Transformer::get_grad_weights)
        .def("set_weights", &Transformer::set_weights)
        .def("save", &Transformer::save)
        .def("load", &Transformer::load);

    py::class_<GAN, Module, std::shared_ptr<GAN>>(m, "GAN")
        .def(py::init<const std::vector<int>&, const std::vector<int>&>())
        .def("forward", &GAN::forward)
        .def("backward", &GAN::backward)
        .def("train_step", &GAN::train_step)
        .def("update", &GAN::update)
        .def("get_weights", &GAN::get_weights)
        .def("get_grad_weights", &GAN::get_grad_weights)
        .def("set_weights", &GAN::set_weights)
        .def("save", &GAN::save)
        .def("load", &GAN::load);

    py::class_<LSTM, Module, std::shared_ptr<LSTM>>(m, "LSTM")
        .def(py::init<int, int, int>())
        .def("forward", &LSTM::forward)
        .def("backward", &LSTM::backward)
        .def("update", &LSTM::update)
        .def("get_weights", &LSTM::get_weights)
        .def("get_grad_weights", &LSTM::get_grad_weights)
        .def("set_weights", &LSTM::set_weights)
        .def("save", &LSTM::save)
        .def("load", &LSTM::load);

    py::class_<GRU, Module, std::shared_ptr<GRU>>(m, "GRU")
        .def(py::init<int, int, int>())
        .def("forward", &GRU::forward)
        .def("backward", &GRU::backward)
        .def("update", &GRU::update)
        .def("get_weights", &GRU::get_weights)
        .def("get_grad_weights", &GRU::get_grad_weights)
        .def("set_weights", &GRU::set_weights)
        .def("save", &GRU::save)
        .def("load", &GRU::load);

    py::class_<Visualization>(m, "Visualization")
        .def(py::init<>())
        .def("plot_tensor", &Visualization::plot_tensor)
        .def("log_to_tensorboard", &Visualization::log_to_tensorboard)
        .def("plot_training_curves", &Visualization::plot_training_curves);

    py::class_<Autograd>(m, "Autograd")
        .def(py::init<>())
        .def("zero_grad", &Autograd::zero_grad)
        .def("backward", &Autograd::backward);
}
