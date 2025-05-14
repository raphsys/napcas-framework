// File: cpp/src/optimizer.cpp

#include "optimizer.h"
#include <fstream>
#include <stdexcept>
#include <cmath>        // pour std::pow et std::sqrt
#include <vector>
#include <memory>

//
// --- SGD ---
//

SGD::SGD(float lr) 
    : learning_rate_(lr) 
{}

SGD::SGD(const std::vector<std::shared_ptr<Module>>& modules, float lr)
    : modules_(modules), learning_rate_(lr) 
{}

void SGD::step() {
    for (auto& module : modules_) {
        module->update(learning_rate_);
    }
}

void SGD::save(const std::string& path) {
    std::ofstream file(path + "_sgd.txt");
    if (!file) throw std::runtime_error("Cannot open file for writing: " + path);
    file << learning_rate_ << "\n";
    file.close();
}

void SGD::load(const std::string& path) {
    std::ifstream file(path + "_sgd.txt");
    if (!file) throw std::runtime_error("Cannot open file for reading: " + path);
    file >> learning_rate_;
    file.close();
}

//
// --- Adam ---
//

Adam::Adam(float lr, float beta1, float beta2, float epsilon)
    : learning_rate_(lr)
    , beta1_(beta1)
    , beta2_(beta2)
    , epsilon_(epsilon)
    , t_(0)
{}

Adam::Adam(const std::vector<std::shared_ptr<Module>>& modules,
           float lr, float beta1, float beta2, float epsilon)
    : modules_(modules)
    , learning_rate_(lr)
    , beta1_(beta1)
    , beta2_(beta2)
    , epsilon_(epsilon)
    , t_(0)
{
    for (const auto& module : modules_) {
        auto& wshape = module->get_weights().shape();
        m_.emplace_back(wshape);
        v_.emplace_back(wshape);
        m_.back().fill(0.0f);
        v_.back().fill(0.0f);
    }
}

void Adam::step() {
    ++t_;
    for (size_t i = 0; i < modules_.size(); ++i) {
        Tensor& weights = modules_[i]->get_weights();
        Tensor& grad    = modules_[i]->get_grad_weights();
        Tensor& m       = m_[i];
        Tensor& v       = v_[i];

        for (int j = 0; j < weights.size(); ++j) {
            // Moments 1 et 2
            m[j] = beta1_ * m[j] + (1.0f - beta1_) * grad[j];
            v[j] = beta2_ * v[j] + (1.0f - beta2_) * grad[j] * grad[j];
            // Correction des biais
            float m_hat = m[j] / (1.0f - std::pow(beta1_, static_cast<float>(t_)));
            float v_hat = v[j] / (1.0f - std::pow(beta2_, static_cast<float>(t_)));
            // Mise à jour des poids
            weights[j] -= learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
        }
    }
}

void Adam::save(const std::string& path) {
    std::ofstream file(path + "_adam.txt");
    if (!file) throw std::runtime_error("Cannot open file for writing: " + path);
    file << learning_rate_ << " "
         << beta1_        << " "
         << beta2_        << " "
         << epsilon_      << " "
         << t_            << "\n";
    file.close();
    for (size_t i = 0; i < m_.size(); ++i) {
        m_[i].save(path + "_m_" + std::to_string(i) + ".tensor");
        v_[i].save(path + "_v_" + std::to_string(i) + ".tensor");
    }
}

void Adam::load(const std::string& path) {
    std::ifstream file(path + "_adam.txt");
    if (!file) throw std::runtime_error("Cannot open file for reading: " + path);
    file >> learning_rate_ >> beta1_ >> beta2_ >> epsilon_ >> t_;
    file.close();
    for (size_t i = 0; i < m_.size(); ++i) {
        m_[i].load(path + "_m_" + std::to_string(i) + ".tensor");
        v_[i].load(path + "_v_" + std::to_string(i) + ".tensor");
    }
}

