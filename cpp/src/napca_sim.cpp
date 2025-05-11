#include "napca_sim.h"
#include <Eigen/Dense>
#include <cmath>
#include <algorithm>
#include <unordered_set>

NAPCA_Sim::NAPCA_Sim(int in_features, int out_features, float alpha, float threshold)
    : learning_rate_(0.0f), alpha_(alpha), threshold_(threshold) {
    
    std::vector<int> weight_shape = {out_features, in_features};
    std::vector<int> bias_shape = {out_features};
    
    weights_ = Tensor(weight_shape);
    bias_ = Tensor(bias_shape);
    grad_weights_ = Tensor(weight_shape);
    grad_bias_ = Tensor(bias_shape);

    // Initialisation adaptative
    for (int i = 0; i < weights_.size(); ++i) {
        weights_[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.1f;
        active_connections_[i] = true;
    }
    
    for (int i = 0; i < bias_.size(); ++i) {
        bias_[i] = 0.0f;
    }
}

void NAPCA_Sim::forward(Tensor& input, Tensor& output) {
    // Implémentation du forward avec calcul simplifié
    std::vector<int> current_path;
    
    Eigen::Map<Eigen::MatrixXf> input_mat(input.data().data(), input.shape()[0], input.shape()[1]);
    Eigen::Map<Eigen::MatrixXf> weights_mat(weights_.data().data(), weights_.shape()[0], weights_.shape()[1]);
    Eigen::Map<Eigen::MatrixXf> output_mat(output.data().data(), output.shape()[0], output.shape()[1]);
    
    // Calcul simplifié avec alpha
    for (int i = 0; i < input.size(); ++i) {
        if (active_connections_[i % weights_.size()]) {
            float val = std::copysign(std::pow(std::abs(input[i]), alpha_), weights_[i % weights_.size()]);
            output[i % output.size()] += val;
            
            // Enregistrement du chemin
            if (val > threshold_) {
                current_path.push_back(i % weights_.size());
            }
        }
    }
    
    // Application du seuil
    for (int i = 0; i < output.size(); ++i) {
        output[i] = (output[i] > threshold_) ? 1.0f : 0.0f;
    }
    
    // Mémorisation du chemin
    memory_paths_.push_back(current_path);
}

void NAPCA_Sim::backward(Tensor& grad_output, Tensor& grad_input) {
    // Implémentation spécifique du backward
    for (int i = 0; i < grad_output.size(); ++i) {
        if (active_connections_[i % weights_.size()]) {
            grad_weights_[i % grad_weights_.size()] += 
                grad_output[i] * std::copysign(std::pow(std::abs(grad_input[i]), alpha_), weights_[i % weights_.size()]);
        }
    }
}

void NAPCA_Sim::update(float lr) {
    // Mise à jour des poids avec le taux d'apprentissage
    learning_rate_ = lr;
    for (int i = 0; i < weights_.size(); ++i) {
        if (active_connections_[i]) {
            weights_[i] -= learning_rate_ * grad_weights_[i];
        }
    }
    for (int i = 0; i < bias_.size(); ++i) {
        bias_[i] -= learning_rate_ * grad_bias_[i];
    }
}

Tensor& NAPCA_Sim::get_weights() {
    return weights_;
}

Tensor& NAPCA_Sim::get_grad_weights() {
    return grad_weights_;
}

void NAPCA_Sim::set_weights(const Tensor& weights) {
    weights_ = weights;
}

float NAPCA_Sim::compute_path_similarity(const std::vector<int>& path1, const std::vector<int>& path2) {
    // Implémentation de la similarité Jaccard
    int intersection = 0;
    std::unordered_set<int> union_set(path1.begin(), path1.end());
    union_set.insert(path2.begin(), path2.end());
    
    for (int id : path1) {
        if (std::find(path2.begin(), path2.end(), id) != path2.end()) {
            intersection++;
        }
    }
    
    return union_set.empty() ? 0.0f : static_cast<float>(intersection) / union_set.size();
}

void NAPCA_Sim::update_weights_conditionally(const std::vector<std::pair<int,int>>& similar_pairs, float eta) {
    // Mise à jour conditionnelle des poids
    for (const auto& pair : similar_pairs) {
        const auto& path1 = memory_paths_[pair.first];
        const auto& path2 = memory_paths_[pair.second];
        
        for (int id : path1) {
            if (std::find(path2.begin(), path2.end(), id) == path2.end()) {
                weights_[id] -= eta * grad_weights_[id];
            }
        }
    }
}

void NAPCA_Sim::prune_connections(float threshold) {
    // Élagage des connexions peu utilisées
    for (auto& [id, active] : active_connections_) {
        if (std::abs(weights_[id]) < threshold) {
            active = false;
        }
    }
}
