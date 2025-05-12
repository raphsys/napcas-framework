#include "nncell.h"
#include "gpu_utils.h"
#include <Eigen/Dense>
#include <stdexcept>
#include <fstream>

NNCel::NNCel(int in_features, int out_features) : learning_rate_(0.0f) {
    std::vector<int> weight_shape = {out_features, in_features};
    std::vector<int> bias_shape = {out_features};
    std::vector<int> connection_shape = {out_features, in_features};
    std::vector<int> scalar_shape = {1};

    weights_ = Tensor(weight_shape);
    bias_ = Tensor(bias_shape);
    grad_weights_ = Tensor(weight_shape);
    grad_bias_ = Tensor(bias_shape);
    connections_ = Tensor(connection_shape);
    threshold_ = Tensor(scalar_shape);
    alpha_ = Tensor(scalar_shape);
    memory_paths_ = Tensor({out_features});
    grad_connections_ = Tensor(connection_shape);
    grad_threshold_ = Tensor(scalar_shape);
    grad_alpha_ = Tensor(scalar_shape);
    grad_memory_paths_ = Tensor({out_features});

    for (int i = 0; i < weights_.size(); ++i) {
        weights_[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.1f;
        connections_[i] = 1.0f;
    }
    bias_.fill(0.0f);
    threshold_.fill(0.5f);
    alpha_.fill(0.6f);
}

void NNCel::forward(Tensor& input, Tensor& output) {
    if (input.ndim() != 2) {
        throw std::invalid_argument("Input must be 2D (batch_size, in_features)");
    }
#ifdef USE_CUDA
    if (input.cuda_data_) {
        forward_cuda(input, output);
        return;
    }
#endif
    int batch_size = input.shape()[0];
    int in_features = input.shape()[1];
    int out_features = weights_.shape()[0];

    if (output.shape() != std::vector<int>{batch_size, out_features}) {
        throw std::invalid_argument("Output shape mismatch");
    }

    Eigen::Map<Eigen::MatrixXf> input_mat(input.data().data(), batch_size, in_features);
    Eigen::Map<Eigen::MatrixXf> weights_mat(weights_.data().data(), out_features, in_features);
    Eigen::Map<Eigen::MatrixXf> connections_mat(connections_.data().data(), out_features, in_features);
    Eigen::Map<Eigen::MatrixXf> output_mat(output.data().data(), batch_size, out_features);
    Eigen::Map<Eigen::VectorXf> bias_vec(bias_.data().data(), bias_.shape()[0]);
    float alpha_val = alpha_[0];

    // Apply connections mask
    Eigen::MatrixXf masked_weights = weights_mat.array() * connections_mat.array();
    output_mat = input_mat * masked_weights.transpose() * alpha_val;
    output_mat.rowwise() += bias_vec.transpose();

    // Update memory paths
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < out_features; ++i) {
            memory_paths_[i] = output_mat(b, i) > threshold_[0] ? 1.0f : 0.0f;
        }
    }
}

void NNCel::backward(Tensor& grad_output, Tensor& grad_input) {
    if (grad_output.ndim() != 2 || grad_input.ndim() != 2) {
        throw std::invalid_argument("Gradients must be 2D");
    }
#ifdef USE_CUDA
    if (grad_output.cuda_data_) {
        backward_cuda(grad_output, grad_input);
        return;
    }
#endif
    int batch_size = grad_input.shape()[0];
    int in_features = grad_input.shape()[1];
    int out_features = weights_.shape()[0];

    if (grad_output.shape() != std::vector<int>{batch_size, out_features}) {
        throw std::invalid_argument("Gradient output shape mismatch");
    }

    Eigen::Map<Eigen::MatrixXf> grad_output_mat(grad_output.data().data(), batch_size, out_features);
    Eigen::Map<Eigen::MatrixXf> grad_input_mat(grad_input.data().data(), batch_size, in_features);
    Eigen::Map<Eigen::MatrixXf> weights_mat(weights_.data().data(), out_features, in_features);
    Eigen::Map<Eigen::MatrixXf> connections_mat(connections_.data().data(), out_features, in_features);
    Eigen::Map<Eigen::MatrixXf> grad_weights_mat(grad_weights_.data().data(), out_features, in_features);
    Eigen::Map<Eigen::MatrixXf> grad_connections_mat(grad_connections_.data().data(), out_features, in_features);
    Eigen::Map<Eigen::VectorXf> grad_bias_vec(grad_bias_.data().data(), grad_bias_.shape()[0]);
    float alpha_val = alpha_[0];

    // Apply connections mask
    Eigen::MatrixXf masked_weights = weights_mat.array() * connections_mat.array();
    grad_input_mat = grad_output_mat * masked_weights * alpha_val;
    grad_weights_mat = (grad_output_mat.transpose() * grad_input_mat * alpha_val).array() * connections_mat.array();
    grad_connections_mat = (grad_output_mat.transpose() * grad_input_mat * alpha_val).array() * weights_mat.array();
    grad_bias_vec = grad_output_mat.colwise().sum();
    grad_alpha_[0] = (grad_output_mat * (input_mat * masked_weights.transpose())).sum();
    grad_threshold_[0] = 0.0f; // Threshold gradient not used
}

void NNCel::update(float lr) {
    learning_rate_ = lr;
    for (int i = 0; i < weights_.size(); ++i) {
        weights_[i] -= lr * grad_weights_[i];
        connections_[i] = std::min(1.0f, std::max(0.0f, connections_[i] - lr * grad_connections_[i]));
    }
    for (int i = 0; i < bias_.size(); ++i) {
        bias_[i] -= lr * grad_bias_[i];
    }
    alpha_[0] -= lr * grad_alpha_[0];
    threshold_[0] = std::max(0.1f, threshold_[0] - lr * grad_threshold_[0]);
}

Tensor& NNCel::get_weights() { return weights_; }
Tensor& NNCel::get_grad_weights() { return grad_weights_; }

void NNCel::set_weights(const Tensor& weights) {
    if (weights.shape() != weights_.shape()) {
        throw std::invalid_argument("Weight shape mismatch");
    }
    weights_ = weights;
}

void NNCel::save(const std::string& path) {
    weights_.save(path + "_weights.tensor");
    bias_.save(path + "_bias.tensor");
    connections_.save(path + "_connections.tensor");
    threshold_.save(path + "_threshold.tensor");
    alpha_.save(path + "_alpha.tensor");
    memory_paths_.save(path + "_memory_paths.tensor");
}

void NNCel::load(const std::string& path) {
    weights_.load(path + "_weights.tensor");
    bias_.load(path + "_bias.tensor");
    connections_.load(path + "_connections.tensor");
    threshold_.load(path + "_threshold.tensor");
    alpha_.load(path + "_alpha.tensor");
    memory_paths_.load(path + "_memory_paths.tensor");
}

#ifdef USE_CUDA
void NNCel::forward_cuda(Tensor& input, Tensor& output) {
    // Implement CUDA forward pass with masked weights
}

void NNCel::backward_cuda(Tensor& grad_output, Tensor& grad_input) {
    // Implement CUDA backward pass
}
#endif
