#include "mlp.h"
#include <stdexcept>
#include <fstream>

MLP::MLP(const std::vector<int>& layers, const std::string& activation) {
    if (layers.size() < 2) {
        throw std::invalid_argument("MLP must have at least input and output layers");
    }
    for (size_t i = 1; i < layers.size(); ++i) {
        linear_layers_.push_back(std::make_shared<Linear>(layers[i-1], layers[i]));
        if (i < layers.size() - 1) {
            if (activation == "relu") {
                activation_layers_.push_back(std::make_shared<ReLU>());
            } else if (activation == "sigmoid") {
                activation_layers_.push_back(std::make_shared<Sigmoid>());
            } else if (activation == "tanh") {
                activation_layers_.push_back(std::make_shared<Tanh>());
            } else {
                throw std::invalid_argument("Unsupported activation: " + activation);
            }
        }
    }
    weights_ = Tensor({static_cast<int>(layers.size()), 1}); // Placeholder for serialization
    grad_weights_ = Tensor({static_cast<int>(layers.size()), 1});
}

void MLP::forward(Tensor& input, Tensor& output) {
    Tensor temp = input;
    for (size_t i = 0; i < linear_layers_.size(); ++i) {
        Tensor next;
        linear_layers_[i]->forward(temp, next);
        if (i < activation_layers_.size()) {
            activation_layers_[i]->forward(next, temp);
        } else {
            temp = next;
        }
    }
    output = temp;
}

void MLP::backward(Tensor& grad_output, Tensor& grad_input) {
    Tensor grad_temp = grad_output;
    for (size_t i = linear_layers_.size(); i > 0; --i) {
        Tensor grad_next;
        if (i - 1 < activation_layers_.size()) {
            activation_layers_[i-1]->backward(grad_temp, grad_next);
        } else {
            grad_next = grad_temp;
        }
        linear_layers_[i-1]->backward(grad_next, grad_temp);
    }
    grad_input = grad_temp;
}

void MLP::update(float lr) {
    for (auto& layer : linear_layers_) {
        layer->update(lr);
    }
}

Tensor& MLP::get_weights() { return weights_; }
Tensor& MLP::get_grad_weights() { return grad_weights_; }

void MLP::set_weights(const Tensor& weights) {
    // Placeholder for setting weights across layers
}

void MLP::save(const std::string& path) {
    for (size_t i = 0; i < linear_layers_.size(); ++i) {
        linear_layers_[i]->save(path + "_layer_" + std::to_string(i));
    }
}

void MLP::load(const std::string& path) {
    for (size_t i = 0; i < linear_layers_.size(); ++i) {
        linear_layers_[i]->load(path + "_layer_" + std::to_string(i));
    }
}
