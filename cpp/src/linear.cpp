#include "linear.h"
#include <random>
#include <cmath>

Linear::Linear(int in_features, int out_features)
    : weights_(std::vector<int>{out_features, in_features}, std::vector<float>(out_features * in_features, 0.0f)),
      bias_(std::vector<int>{out_features}, std::vector<float>(out_features, 0.0f)),
      grad_weights_(std::vector<int>{out_features, in_features}, std::vector<float>(out_features * in_features, 0.0f)),
      grad_bias_(std::vector<int>{out_features}, std::vector<float>(out_features, 0.0f)),
      learning_rate_(0.01f) {

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(0.0f, 0.01f);

    for (int i = 0; i < weights_.size(); ++i) {
        weights_[i] = dist(gen);
    }
}

void Linear::forward(Tensor& input, Tensor& output) {
    if (input.shape()[1] != weights_.shape()[1]) {
        throw std::invalid_argument("Input and weights dimensions do not match.");
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

void Linear::backward(Tensor& grad_output, Tensor& grad_input) {
    for (int i = 0; i < grad_output.size(); ++i) {
        for (int j = 0; j < grad_input.size(); ++j) {
            grad_input[j] += grad_output[i] * weights_[i * weights_.shape()[1] + j];
        }
    }

    for (int i = 0; i < grad_output.size(); ++i) {
        for (int j = 0; j < grad_input.size(); ++j) {
            grad_weights_[i * weights_.shape()[1] + j] += grad_output[i] * grad_input[j];
            grad_bias_[i] += grad_output[i];
        }
    }
}

void Linear::update(float lr) {
    for (int i = 0; i < weights_.size(); ++i) {
        weights_[i] -= lr * grad_weights_[i];
    }

    for (int i = 0; i < bias_.size(); ++i) {
        bias_[i] -= lr * grad_bias_[i];
    }

    grad_weights_.zero_grad();
    grad_bias_.zero_grad();
}
