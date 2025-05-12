#include "napcas.h"
#include <Eigen/Dense>
#include <stdexcept>
#include <fstream>
#include <cmath>

NAPCAS::NAPCAS(int input_size, int output_size)
    : input_size_(input_size), output_size_(output_size) {
    weights_ = Tensor({output_size, input_size});
    alpha_ = Tensor({output_size});
    grad_weights_ = Tensor({output_size, input_size});
    grad_alpha_ = Tensor({output_size});
    weights_.fill(0.01f); // Simple initialization
    alpha_.fill(0.0f);
}

void NAPCAS::compute_mask() {
    masked_weights_ = Tensor(weights_.shape());
    for (int i = 0; i < output_size_; ++i) {
        for (int j = 0; j < input_size_; ++j) {
            int idx = i * input_size_ + j;
            masked_weights_[idx] = weights_[idx] * std::exp(-std::abs(alpha_[i]));
        }
    }
}

void NAPCAS::forward(Tensor& input, Tensor& output) {
    if (input.shape() != std::vector<int>({1, input_size_})) {
        throw std::invalid_argument("Input shape must be {1, input_size}");
    }
    if (output.shape() != std::vector<int>({1, output_size_})) {
        throw std::invalid_argument("Output shape must be {1, output_size}");
    }

    compute_mask();
    Eigen::Map<Eigen::MatrixXf> input_mat(input.data().data(), 1, input_size_);
    Eigen::Map<Eigen::MatrixXf> weight_mat(masked_weights_.data().data(), output_size_, input_size_);
    Eigen::Map<Eigen::MatrixXf> output_mat(output.data().data(), 1, output_size_);
    output_mat = input_mat * weight_mat.transpose();
    for (int i = 0; i < output_size_; ++i) {
        output[i] += alpha_[i];
    }

    input_ = input; // Cache input for backward pass
}

void NAPCAS::backward(Tensor& grad_output, Tensor& grad_input) {
    if (grad_output.shape() != std::vector<int>({1, output_size_})) {
        throw std::invalid_argument("Grad output shape must be {1, output_size}");
    }
    if (grad_input.shape() != std::vector<int>({1, input_size_})) {
        throw std::invalid_argument("Grad input shape must be {1, input_size}");
    }

    // Map tensors to Eigen matrices
    Eigen::Map<Eigen::MatrixXf> grad_output_mat(grad_output.data().data(), 1, output_size_);
    Eigen::Map<Eigen::MatrixXf> input_mat(input_.data().data(), 1, input_size_);
    Eigen::Map<Eigen::MatrixXf> masked_weights_mat(masked_weights_.data().data(), output_size_, input_size_);
    Eigen::Map<Eigen::MatrixXf> grad_input_mat(grad_input.data().data(), 1, input_size_);
    Eigen::Map<Eigen::MatrixXf> grad_weights_mat(grad_weights_.data().data(), output_size_, input_size_);

    // Compute gradients
    grad_input_mat = grad_output_mat * masked_weights_mat;
    grad_weights_mat = grad_output_mat.transpose() * input_mat;
    grad_alpha_[0] = (grad_output_mat * (input_mat * masked_weights_mat.transpose())).sum();

    // Update masked weights gradients
    for (int i = 0; i < output_size_; ++i) {
        for (int j = 0; j < input_size_; ++j) {
            int idx = i * input_size_ + j;
            grad_weights_[idx] *= std::exp(-std::abs(alpha_[i]));
        }
    }
}

void NAPCAS::update(float lr) {
    for (size_t i = 0; i < weights_.size(); ++i) {
        weights_[i] -= lr * grad_weights_[i];
    }
    for (size_t i = 0; i < alpha_.size(); ++i) {
        alpha_[i] -= lr * grad_alpha_[i];
    }
}

void NAPCAS::set_weights(const Tensor& weights) {
    if (weights.shape() != std::vector<int>({output_size_, input_size_})) {
        throw std::invalid_argument("Weight shape mismatch");
    }
    weights_ = weights;
}

void NAPCAS::save(const std::string& path) {
    weights_.save(path + "_weights.tensor");
    alpha_.save(path + "_alpha.tensor");
}

void NAPCAS::load(const std::string& path) {
    weights_.load(path + "_weights.tensor");
    alpha_.load(path + "_alpha.tensor");
}
