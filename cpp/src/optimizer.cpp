#include "optimizer.h"
#include <cmath>

SGD::SGD(float lr) : learning_rate_(lr) {}

SGD::SGD(std::vector<std::shared_ptr<Module>> modules, float lr) : learning_rate_(lr), modules_(modules) {}

void SGD::step() {
    for (auto& module : modules_) {
        Tensor& weights = module->get_weights();
        Tensor& grad_weights = module->get_grad_weights();
        for (int i = 0; i < weights.size(); ++i) {
            weights[i] -= learning_rate_ * grad_weights[i];
        }
    }
}

Adam::Adam(float lr, float beta1, float beta2, float epsilon)
    : learning_rate_(lr), beta1_(beta1), beta2_(beta2), epsilon_(epsilon), t_(0) {}

Adam::Adam(std::vector<std::shared_ptr<Module>> modules, float lr, float beta1, float beta2, float epsilon)
    : learning_rate_(lr), beta1_(beta1), beta2_(beta2), epsilon_(epsilon), t_(0), modules_(modules) {
    for (auto& module : modules_) {
        Tensor& weights = module->get_weights();
        m_.emplace_back(weights.shape(), std::vector<float>(weights.size(), 0.0f));
        v_.emplace_back(weights.shape(), std::vector<float>(weights.size(), 0.0f));
    }
}

void Adam::step() {
    ++t_;
    for (size_t i = 0; i < modules_.size(); ++i) {
        Tensor& weights = modules_[i]->get_weights();
        Tensor& grad_weights = modules_[i]->get_grad_weights();
        for (int j = 0; j < weights.size(); ++j) {
            m_[i][j] = beta1_ * m_[i][j] + (1.0f - beta1_) * grad_weights[j];
            v_[i][j] = beta2_ * v_[i][j] + (1.0f - beta2_) * grad_weights[j] * grad_weights[j];
            float m_hat = m_[i][j] / (1.0f - std::pow(beta1_, t_));
            float v_hat = v_[i][j] / (1.0f - std::pow(beta2_, t_));
            weights[j] -= learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
        }
    }
}
