// napcas.cpp
#include "napcas.h"
#include <random>
#include <cmath>

NAPCAS::NAPCAS(int in_features, int out_features)
    : weights_(std::vector<int>{out_features, in_features}),
      connections_(std::vector<int>{out_features, in_features}, std::vector<float>(out_features * in_features, 1.0f)),
      threshold_(std::vector<int>{out_features}, std::vector<float>(out_features, 0.5f)),
      alpha_(std::vector<int>{out_features}, std::vector<float>(out_features, 0.6f)),
      memory_paths_(std::vector<int>{out_features}, std::vector<float>(out_features, 0.0f)),
      grad_weights_(std::vector<int>{out_features, in_features}, std::vector<float>(out_features * in_features, 0.0f)),
      grad_connections_(std::vector<int>{out_features, in_features}, std::vector<float>(out_features * in_features, 0.0f)),
      grad_threshold_(std::vector<int>{out_features}, std::vector<float>(out_features, 0.0f)),
      grad_alpha_(std::vector<int>{out_features}, std::vector<float>(out_features, 0.0f)),
      grad_memory_paths_(std::vector<int>{out_features}, std::vector<float>(out_features, 0.0f)),
      learning_rate_(0.01f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(0.0f, 0.01f);

    for (int i = 0; i < weights_.size(); ++i) {
        weights_[i] = dist(gen);
    }
}

void NAPCAS::forward(Tensor& input, Tensor& output) {
    if (input.shape()[1] != weights_.shape()[1]) {
        throw std::invalid_argument("Input and weights dimensions do not match.");
    }

    output.reshape({weights_.shape()[0]});
    for (int i = 0; i < weights_.shape()[0]; ++i) {
        float weighted_sum = 0.0f;
        for (int j = 0; j < weights_.shape()[1]; ++j) {
            if (connections_[i * weights_.shape()[1] + j] > 0.5f) {
                weighted_sum += std::copysign(1.0f, weights_[i * weights_.shape()[1] + j]) * std::pow(std::abs(input[j]), alpha_[i]);
            }
        }
        output[i] = (weighted_sum > threshold_[i]) ? 1.0f : 0.0f;
    }
}

void NAPCAS::backward(Tensor& grad_output, Tensor& grad_input) {
    for (int i = 0; i < grad_output.size(); ++i) {
        for (int j = 0; j < grad_input.size(); ++j) {
            grad_input[j] += grad_output[i] * weights_[i * weights_.shape()[1] + j];
        }
    }

    for (int i = 0; i < grad_output.size(); ++i) {
        for (int j = 0; j < grad_input.size(); ++j) {
            grad_weights_[i * weights_.shape()[1] + j] += grad_output[i] * grad_input[j];
            grad_connections_[i * weights_.shape()[1] + j] += grad_output[i] * grad_input[j];
        }
        grad_threshold_[i] += grad_output[i];
        grad_alpha_[i] += grad_output[i];
    }
}

void NAPCAS::update(float lr) {
    for (int i = 0; i < weights_.size(); ++i) {
        weights_[i] -= lr * grad_weights_[i];
    }

    for (int i = 0; i < connections_.size(); ++i) {
        connections_[i] -= lr * grad_connections_[i];
    }

    for (int i = 0; i < threshold_.size(); ++i) {
        threshold_[i] -= lr * grad_threshold_[i];
    }

    for (int i = 0; i < alpha_.size(); ++i) {
        alpha_[i] -= lr * grad_alpha_[i];
    }

    grad_weights_.zero_grad();
    grad_connections_.zero_grad();
    grad_threshold_.zero_grad();
    grad_alpha_.zero_grad();
}
