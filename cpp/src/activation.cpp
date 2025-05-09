#include "activation.h"
#include <cmath>

// Constructeurs
ReLU::ReLU() {}

Sigmoid::Sigmoid() : output_({1}) {}  // Initialisation avec une forme par défaut

Tanh::Tanh() : output_({1}) {}       // Initialisation avec une forme par défaut

// Implémentation des méthodes
void ReLU::forward(Tensor& input, Tensor& output) {
    output.reshape(input.shape());
    for (int i = 0; i < input.size(); ++i) {
        output[i] = std::max(0.0f, input[i]);
    }
}

void ReLU::backward(Tensor& grad_output, Tensor& grad_input) {
    for (int i = 0; i < grad_output.size(); ++i) {
        grad_input[i] = grad_output[i] * (grad_input[i] > 0 ? 1.0f : 0.0f);
    }
}

void ReLU::update(float lr) {}

void Sigmoid::forward(Tensor& input, Tensor& output) {
    output.reshape(input.shape());
    for (int i = 0; i < input.size(); ++i) {
        output[i] = 1.0f / (1.0f + std::exp(-input[i]));
    }
    output_ = output;
}

void Sigmoid::backward(Tensor& grad_output, Tensor& grad_input) {
    for (int i = 0; i < grad_output.size(); ++i) {
        grad_input[i] = grad_output[i] * (output_[i] * (1.0f - output_[i]));  // Ajout de la parenthèse manquante
    }
}

void Sigmoid::update(float lr) {}

void Tanh::forward(Tensor& input, Tensor& output) {
    output.reshape(input.shape());
    for (int i = 0; i < input.size(); ++i) {
        output[i] = std::tanh(input[i]);
    }
    output_ = output;
}

void Tanh::backward(Tensor& grad_output, Tensor& grad_input) {
    for (int i = 0; i < grad_output.size(); ++i) {
        grad_input[i] = grad_output[i] * (1.0f - output_[i] * output_[i]);
    }
}

void Tanh::update(float lr) {}
