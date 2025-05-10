#include "optimizer.h"
#include <cmath>
#include <vector>
#include <memory>

// Constructeur par défaut pour SGD avec une learning rate par défaut
SGD::SGD(float lr) : learning_rate_(lr) {}

// Constructeur pour SGD avec une liste de modules et une learning rate par défaut
SGD::SGD(const std::vector<std::shared_ptr<Module>>& modules, float lr)
    : learning_rate_(lr), modules_(modules) {}

// Méthode pour effectuer un pas d'optimisation
void SGD::step() {
    for (auto& module : modules_) {
        auto weights = module->get_weights();
        auto grad_weights = module->get_grad_weights();
        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i] -= learning_rate_ * grad_weights[i];
        }
        module->set_weights(weights);
    }
}

// Constructeur par défaut pour Adam avec des valeurs par défaut pour les paramètres
Adam::Adam(float lr, float beta1, float beta2, float epsilon)
    : learning_rate_(lr), beta1_(beta1), beta2_(beta2), epsilon_(epsilon), t_(0) {}

// Constructeur pour Adam avec une liste de modules et des valeurs par défaut pour les paramètres
Adam::Adam(const std::vector<std::shared_ptr<Module>>& modules, float lr, float beta1, float beta2, float epsilon)
    : learning_rate_(lr), beta1_(beta1), beta2_(beta2), epsilon_(epsilon), t_(0), modules_(modules) {
    for (auto& module : modules) {
        auto weights = module->get_weights();
        m_.push_back(Tensor(weights.shape()));
        v_.push_back(Tensor(weights.shape()));
    }
}

// Méthode pour effectuer un pas d'optimisation
void Adam::step() {
    t_++;
    for (size_t i = 0; i < modules_.size(); ++i) {
        auto module = modules_[i];
        auto weights = module->get_weights();
        auto grad_weights = module->get_grad_weights();
        for (size_t j = 0; j < weights.size(); ++j) {
            // Calcul des moments premiers et seconds
            m_[i][j] = beta1_ * m_[i][j] + (1 - beta1_) * grad_weights[j];
            v_[i][j] = beta2_ * v_[i][j] + (1 - beta2_) * grad_weights[j] * grad_weights[j];

            // Correction des biais
            float m_hat = m_[i][j] / (1 - std::pow(beta1_, t_));
            float v_hat = v_[i][j] / (1 - std::pow(beta2_, t_));

            // Mise à jour des poids
            weights[j] -= learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
        }
        module->set_weights(weights);
    }
}
