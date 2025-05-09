#include "nncell.h"
#include <random>
#include <cmath>
#include <stdexcept>

NNCell::NNCell(int in_features, int out_features) 
    : weights_({out_features, in_features}), 
      bias_({out_features}), 
      grad_weights_({out_features, in_features}), 
      grad_bias_({out_features}),
      input_(),
      learning_rate_(0.01f) {
    
    // Initialisation des poids avec une distribution normale
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.01f);

    for (int i = 0; i < weights_.size(); ++i) {
        weights_[i] = dist(gen);
    }

    // Initialisation des biais à 0
    for (int i = 0; i < bias_.size(); ++i) {
        bias_[i] = 0.0f;
    }
}

void NNCell::forward(Tensor& input, Tensor& output) {
    input_ = input; // Stockage de l'input pour la backpropagation
    
    if (input.shape().size() != 1 || input.shape()[0] != weights_.shape()[1]) {
        throw std::invalid_argument("Input dimensions do not match weights dimensions");
    }

    output.reshape({weights_.shape()[0]});
    
    for (int i = 0; i < weights_.shape()[0]; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < weights_.shape()[1]; ++j) {
            sum += input[j] * weights_[i * weights_.shape()[1] + j];
        }
        output[i] = sum + bias_[i];
    }
}

void NNCell::backward(Tensor& grad_output, Tensor& grad_input) {
    // Calcul du gradient par rapport à l'input
    grad_input.reshape({weights_.shape()[1]});
    grad_input.fill(0.0f);
    
    for (int i = 0; i < grad_output.size(); ++i) {
        for (int j = 0; j < grad_input.size(); ++j) {
            grad_input[j] += grad_output[i] * weights_[i * weights_.shape()[1] + j];
        }
    }

    // Calcul du gradient pour les poids et biais
    grad_weights_.fill(0.0f);
    grad_bias_.fill(0.0f);
    
    for (int i = 0; i < grad_output.size(); ++i) {
        for (int j = 0; j < input_.size(); ++j) {
            grad_weights_[i * weights_.shape()[1] + j] += grad_output[i] * input_[j];
        }
        grad_bias_[i] += grad_output[i];
    }
}

void NNCell::update(float lr) {
    // Mise à jour des poids
    for (int i = 0; i < weights_.size(); ++i) {
        weights_[i] -= lr * grad_weights_[i];
    }

    // Mise à jour des biais
    for (int i = 0; i < bias_.size(); ++i) {
        bias_[i] -= lr * grad_bias_[i];
    }
}
