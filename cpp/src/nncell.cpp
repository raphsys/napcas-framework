#include "nncell.h"
#include <Eigen/Dense>
#include <stdexcept>

NNCell::NNCell(int in_features, int out_features) : learning_rate_(0.0f) {
    std::vector<int> weight_shape = {out_features, in_features};
    std::vector<int> bias_shape = {out_features};
    weights_ = Tensor(weight_shape);
    bias_ = Tensor(bias_shape);
    grad_weights_ = Tensor(weight_shape);
    grad_bias_ = Tensor(bias_shape);

    // Initialisation des poids (exemple : initialisation aléatoire simple)
    for (int i = 0; i < weights_.size(); ++i) {
        weights_[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
    }
    for (int i = 0; i < bias_.size(); ++i) {
        bias_[i] = 0.0f;
    }
}

void NNCell::forward(Tensor& input, Tensor& output) {
    if (input.ndim() != 2) {
        throw std::invalid_argument("Input must be 2D (batch_size, in_features)");
    }
    int batch_size = input.shape()[0];
    int in_features = input.shape()[1];
    int out_features = weights_.shape()[0];

    if (output.shape() != std::vector<int>{batch_size, out_features}) {
        throw std::invalid_argument("Output shape mismatch");
    }

    Eigen::Map<Eigen::MatrixXf> input_mat(input.data().data(), batch_size, in_features);
    Eigen::Map<Eigen::MatrixXf> weights_mat(weights_.data().data(), out_features, in_features);
    Eigen::Map<Eigen::MatrixXf> output_mat(output.data().data(), batch_size, out_features);
    Eigen::Map<Eigen::VectorXf> bias_vec(bias_.data().data(), bias_.shape()[0]);

    output_mat = input_mat * weights_mat.transpose();
    output_mat.rowwise() += bias_vec.transpose();
}

void NNCell::backward(Tensor& grad_output, Tensor& grad_input) {
    if (grad_output.ndim() != 2 || grad_input.ndim() != 2) {
        throw std::invalid_argument("Gradients must be 2D");
    }
    int batch_size = grad_input.shape()[0];
    int in_features = grad_input.shape()[1];
    int out_features = weights_.shape()[0];

    if (grad_output.shape() != std::vector<int>{batch_size, out_features}) {
        throw std::invalid_argument("Gradient output shape mismatch");
    }

    Eigen::Map<Eigen::MatrixXf> grad_output_mat(grad_output.data().data(), batch_size, out_features);
    Eigen::Map<Eigen::MatrixXf> grad_input_mat(grad_input.data().data(), batch_size, in_features);
    Eigen::Map<Eigen::MatrixXf> weights_mat(weights_.data().data(), out_features, in_features);
    Eigen::Map<Eigen::MatrixXf> grad_weights_mat(grad_weights_.data().data(), out_features, in_features);
    Eigen::Map<Eigen::VectorXf> grad_bias_vec(grad_bias_.data().data(), grad_bias_.shape()[0]);

    grad_input_mat = grad_output_mat * weights_mat;
    grad_weights_mat = grad_output_mat.transpose() * grad_input_mat;
    grad_bias_vec = grad_output_mat.colwise().sum();
}

void NNCell::update(float lr) {
    learning_rate_ = lr;
    for (int i = 0; i < weights_.size(); ++i) {
        weights_[i] -= lr * grad_weights_[i];
    }
    for (int i = 0; i < bias_.size(); ++i) {
        bias_[i] -= lr * grad_bias_[i];
    }
}

Tensor& NNCell::get_weights() { return weights_; }
Tensor& NNCell::get_grad_weights() { return grad_weights_; }
