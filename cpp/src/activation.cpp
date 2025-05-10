#include "activation.h"
#include <cmath>
#include <stdexcept>

ReLU::ReLU() {}

void ReLU::forward(Tensor& input, Tensor& output) {
    if (input.shape() != output.shape()) {
        throw std::invalid_argument("Input and output shapes must match");
    }
    for (int i = 0; i < input.size(); ++i) {
        output[i] = std::max(0.0f, input[i]);
    }
}

void ReLU::backward(Tensor& grad_output, Tensor& grad_input) {
    if (grad_output.shape() != grad_input.shape()) {
        throw std::invalid_argument("Gradient shapes must match");
    }
    for (int i = 0; i < grad_output.size(); ++i) {
        grad_input[i] = (grad_output[i] > 0) ? grad_output[i] : 0.0f;
    }
}

void ReLU::update(float) {}

Tensor& ReLU::get_weights() {
    throw std::runtime_error("ReLU has no weights");
}

Tensor& ReLU::get_grad_weights() {
    throw std::runtime_error("ReLU has no gradient weights");
}

void ReLU::set_weights(const Tensor& weights) {
    throw std::runtime_error("ReLU has no weights to set");
}

Sigmoid::Sigmoid() {}

void Sigmoid::forward(Tensor& input, Tensor& output) {
    if (input.shape() != output.shape()) {
        throw std::invalid_argument("Input and output shapes must match");
    }
    for (int i = 0; i < input.size(); ++i) {
        output[i] = 1.0f / (1.0f + std::exp(-input[i]));
    }
}

void Sigmoid::backward(Tensor& grad_output, Tensor& grad_input) {
    if (grad_output.shape() != grad_input.shape()) {
        throw std::invalid_argument("Gradient shapes must match");
    }
    for (int i = 0; i < grad_output.size(); ++i) {
        float sigmoid = 1.0f / (1.0f + std::exp(-grad_output[i]));
        grad_input[i] = grad_output[i] * sigmoid * (1.0f - sigmoid);
    }
}

void Sigmoid::update(float) {}

Tensor& Sigmoid::get_weights() {
    throw std::runtime_error("Sigmoid has no weights");
}

Tensor& Sigmoid::get_grad_weights() {
    throw std::runtime_error("Sigmoid has no gradient weights");
}

void Sigmoid::set_weights(const Tensor& weights) {
    throw std::runtime_error("Sigmoid has no weights to set");
}

Tanh::Tanh() {}

void Tanh::forward(Tensor& input, Tensor& output) {
    if (input.shape() != output.shape()) {
        throw std::invalid_argument("Input and output shapes must match");
    }
    for (int i = 0; i < input.size(); ++i) {
        output[i] = std::tanh(input[i]);
    }
}

void Tanh::backward(Tensor& grad_output, Tensor& grad_input) {
    if (grad_output.shape() != grad_input.shape()) {
        throw std::invalid_argument("Gradient shapes must match");
    }
    for (int i = 0; i < grad_output.size(); ++i) {
        float tanh_val = std::tanh(grad_output[i]);
        grad_input[i] = grad_output[i] * (1.0f - tanh_val * tanh_val);
    }
}

void Tanh::update(float) {}

Tensor& Tanh::get_weights() {
    throw std::runtime_error("Tanh has no weights");
}

Tensor& Tanh::get_grad_weights() {
    throw std::runtime_error("Tanh has no gradient weights");
}

void Tanh::set_weights(const Tensor& weights) {
    throw std::runtime_error("Tanh has no weights to set");
}
