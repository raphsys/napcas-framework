#include "rnn.h"
#include <stdexcept>
#include <fstream>

RNN::RNN(int input_size, int hidden_size, int num_layers)
    : input_size_(input_size), hidden_size_(hidden_size), num_layers_(num_layers) {
    for (int i = 0; i < num_layers; ++i) {
        input_to_hidden_.push_back(std::make_shared<Linear>(i == 0 ? input_size : hidden_size, hidden_size));
        hidden_to_hidden_.push_back(std::make_shared<Linear>(hidden_size, hidden_size));
        activations_.push_back(std::make_shared<Tanh>());
    }
    weights_ = Tensor({num_layers, 1}); // Placeholder for serialization
    grad_weights_ = Tensor({num_layers, 1});
}

void RNN::forward(Tensor& input, Tensor& output) {
    if (input.ndim() != 3) {
        throw std::invalid_argument("Input must be 3D (seq_len, batch_size, input_size)");
    }
    int seq_len = input.shape()[0];
    int batch_size = input.shape()[1];
    Tensor hidden = Tensor({batch_size, hidden_size_});
    hidden.fill(0.0f);

    for (int t = 0; t < seq_len; ++t) {
        Tensor input_t({batch_size, input_size_});
        for (int b = 0; b < batch_size; ++b) {
            for (int i = 0; i < input_size_; ++i) {
                input_t[b * input_size_ + i] = input[t * batch_size * input_size_ + b * input_size_ + i];
            }
        }

        Tensor next_hidden = hidden;
        for (int l = 0; l < num_layers_; ++l) {
            Tensor temp;
            input_to_hidden_[l]->forward(l == 0 ? input_t : next_hidden, temp);
            hidden_to_hidden_[l]->forward(next_hidden, temp);
            activations_[l]->forward(temp, next_hidden);
        }

        for (int b = 0; b < batch_size; ++b) {
            for (int i = 0; i < hidden_size_; ++i) {
                output[t * batch_size * hidden_size_ + b * hidden_size_ + i] = next_hidden[b * hidden_size_ + i];
            }
        }
        hidden = next_hidden;
    }
}

void RNN::backward(Tensor& grad_output, Tensor& grad_input) {
    // Simplified backward pass (placeholder)
    grad_input = grad_output; // Implement full BPTT for production
}

void RNN::update(float lr) {
    for (size_t i = 0; i < input_to_hidden_.size(); ++i) {
        input_to_hidden_[i]->update(lr);
        hidden_to_hidden_[i]->update(lr);
    }
}

Tensor& RNN::get_weights() { return weights_; }
Tensor& RNN::get_grad_weights() { return grad_weights_; }

void RNN::set_weights(const Tensor& weights) {
    // Placeholder for setting weights
}

void RNN::save(const std::string& path) {
    for (size_t i = 0; i < input_to_hidden_.size(); ++i) {
        input_to_hidden_[i]->save(path + "_input_to_hidden_" + std::to_string(i));
        hidden_to_hidden_[i]->save(path + "_hidden_to_hidden_" + std::to_string(i));
    }
}

void RNN::load(const std::string& path) {
    for (size_t i = 0; i < input_to_hidden_.size(); ++i) {
        input_to_hidden_[i]->load(path + "_input_to_hidden_" + std::to_string(i));
        hidden_to_hidden_[i]->load(path + "_hidden_to_hidden_" + std::to_string(i));
    }
}
