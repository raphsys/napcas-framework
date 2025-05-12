#include "napca_sim.h"
#include "gpu_utils.h"
#include <Eigen/Dense>
#include <cmath>
#include <algorithm>
#include <unordered_set>
#include <fstream>
#include <stdexcept>

NAPCA_Sim::NAPCA_Sim(int in_features, int out_features, float alpha, float threshold)
    : learning_rate_(0.0f), alpha_(alpha), threshold_(threshold) {
    std::vector<int> weight_shape = {out_features, in_features};
    std::vector<int> bias_shape = {out_features};
    
    weights_ = Tensor(weight_shape);
    bias_ = Tensor(bias_shape);
    grad_weights_ = Tensor(weight_shape);
    grad_bias_ = Tensor(bias_shape);

    for (int i = 0; i < weights_.size(); ++i) {
        weights_[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.1f;
        active_connections_[i] = true;
    }
    bias_.fill(0.0f);
}

void NAPCA_Sim::forward(Tensor& input, Tensor& output) {
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
    Eigen::Map<Eigen::MatrixXf> output_mat(output.data().data(), batch_size, out_features);
    Eigen::Map<Eigen::VectorXf> bias_vec(bias_.data().data(), bias_.shape()[0]);

    // Apply active connections mask
    Eigen::MatrixXf masked_weights = weights_mat;
    for (int i = 0; i < out_features; ++i) {
        for (int j = 0; j < in_features; ++j) {
            if (!active_connections_[i * in_features + j]) {
                masked_weights(i, j) = 0.0f;
            }
        }
    }

    output_mat = input_mat * masked_weights.transpose() * alpha_;
    output_mat.rowwise() += bias_vec.transpose();

    // Store activation paths for similarity computation
    memory_paths_.clear();
    for (int b = 0; b < batch_size; ++b) {
        std::vector<int> path;
        for (int i = 0; i < out_features; ++i) {
            if (output_mat(b, i) > threshold_) {
                path.push_back(i);
            }
        }
        memory_paths_.push_back(path);
    }
}

void NAPCA_Sim::backward(Tensor& grad_output, Tensor& grad_input) {
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
    Eigen::Map<Eigen::MatrixXf> grad_weights_mat(grad_weights_.data().data(), out_features, in_features);
    Eigen::Map<Eigen::VectorXf> grad_bias_vec(grad_bias_.data().data(), grad_bias_.shape()[0]);

    // Apply active connections mask
    Eigen::MatrixXf masked_weights = weights_mat;
    for (int i = 0; i < out_features; ++i) {
        for (int j = 0; j < in_features; ++j) {
            if (!active_connections_[i * in_features + j]) {
                masked_weights(i, j) = 0.0f;
            }
        }
    }

    grad_input_mat = grad_output_mat * masked_weights * alpha_;
    grad_weights_mat = grad_output_mat.transpose() * grad_input_mat * alpha_;
    grad_bias_vec = grad_output_mat.colwise().sum();
}

void NAPCA_Sim::update(float lr) {
    learning_rate_ = lr;
    for (int i = 0; i < weights_.size(); ++i) {
        if (active_connections_[i]) {
            weights_[i] -= lr * grad_weights_[i];
        }
    }
    for (int i = 0; i < bias_.size(); ++i) {
        bias_[i] -= lr * grad_bias_[i];
    }
}

float NAPCA_Sim::compute_path_similarity(const std::vector<int>& path1, const std::vector<int>& path2) {
    std::unordered_set<int> set1(path1.begin(), path1.end());
    std::unordered_set<int> set2(path2.begin(), path2.end());
    std::vector<int> intersection;
    std::vector<int> union_set;

    // Compute intersection
    for (const int& elem : set1) {
        if (set2.find(elem) != set2.end()) {
            intersection.push_back(elem);
        }
    }

    // Compute union
    union_set = path1;
    for (const int& elem : path2) {
        if (std::find(union_set.begin(), union_set.end(), elem) == union_set.end()) {
            union_set.push_back(elem);
        }
    }

    // Jaccard similarity
    return intersection.size() / static_cast<float>(union_set.size());
}

void NAPCA_Sim::update_weights_conditionally(const std::vector<std::pair<int,int>>& similar_pairs, float eta) {
    for (const auto& pair : similar_pairs) {
        int idx1 = pair.first;
        int idx2 = pair.second;
        for (int j = 0; j < weights_.shape()[1]; ++j) {
            int w_idx1 = idx1 * weights_.shape()[1] + j;
            int w_idx2 = idx2 * weights_.shape()[1] + j;
            if (active_connections_[w_idx1] && active_connections_[w_idx2]) {
                weights_[w_idx1] += eta * (weights_[w_idx2] - weights_[w_idx1]);
                weights_[w_idx2] += eta * (weights_[w_idx1] - weights_[w_idx2]);
            }
        }
    }
}

void NAPCA_Sim::prune_connections(float threshold) {
    for (int i = 0; i < weights_.size(); ++i) {
        if (std::abs(weights_[i]) < threshold) {
            active_connections_[i] = false;
        }
    }
}

Tensor& NAPCA_Sim::get_weights() { return weights_; }
Tensor& NAPCA_Sim::get_grad_weights() { return grad_weights_; }

void NAPCA_Sim::set_weights(const Tensor& weights) {
    if (weights.shape() != weights_.shape()) {
        throw std::invalid_argument("Weight shape mismatch");
    }
    weights_ = weights;
}

void NAPCA_Sim::save(const std::string& path) {
    weights_.save(path + "_weights.tensor");
    bias_.save(path + "_bias.tensor");
    std::ofstream file(path + "_active_connections.txt");
    for (const auto& [idx, active] : active_connections_) {
        file << idx << " " << active << "\n";
    }
    file.close();
}

void NAPCA_Sim::load(const std::string& path) {
    weights_.load(path + "_weights.tensor");
    bias_.load(path + "_bias.tensor");
    std::ifstream file(path + "_active_connections.txt");
    active_connections_.clear();
    int idx;
    bool active;
    while (file >> idx >> active) {
        active_connections_[idx] = active;
    }
    file.close();
}

#ifdef USE_CUDA
void NAPCA_Sim::forward_cuda(Tensor& input, Tensor& output) {
    // Implement CUDA forward pass with masked weights
}

void NAPCA_Sim::backward_cuda(Tensor& grad_output, Tensor& grad_input) {
    // Implement CUDA backward pass
}
#endif
