// File: cpp/include/transformer.h

#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "module.h"
#include "tensor.h"
#include "attention.h"
#include "linear.h"
#include "activation.h"
#include <memory>
#include <vector>
#include <string>

/// @brief Transformer composé de blocs self-attention + feed-forward.
class Transformer : public Module {
public:
    /// @param d_model     Dimension des embeddings.
    /// @param num_heads   Nombre de têtes d’attention.
    /// @param d_ff        Dimension interne du feed-forward.
    /// @param num_layers  Nombre de blocs.
    Transformer(int d_model,
                int num_heads,
                int d_ff,
                int num_layers);

    /// @brief Passe avant (seq_len, batch, d_model) → même forme.
    void forward(Tensor& input, Tensor& output) override;

    /// @brief Rétro-propagation complète sur tous les blocs.
    void backward(Tensor& grad_output, Tensor& grad_input) override;

    /// @brief Mise à jour (SGD) de tous les sous-modules.
    void update(float lr) override;

    /// @brief Pas de poids « globaux » pour le Transformer.
    Tensor& get_weights() override { return dummy_; }
    Tensor& get_grad_weights() override { return dummy_; }
    void set_weights(const Tensor& w) override { dummy_ = w; }

    void save(const std::string& path) override {}
    void load(const std::string& path) override {}

private:
    int d_model_;
    int num_heads_;
    int d_ff_;
    int num_layers_;

    // Les sous-modules
    std::vector<std::shared_ptr<MultiHeadAttention>> attention_layers_;
    std::vector<std::shared_ptr<Linear>>            ff_w1_, ff_w2_;
    std::vector<std::shared_ptr<ReLU>>              ff_activations_;

    // Caches pour backward
    std::vector<Tensor> residual1_cache_;    // avant attention
    std::vector<Tensor> attn_out_cache_;     // sortie attention
    std::vector<Tensor> residual2_cache_;    // après attention
    std::vector<Tensor> ff_hidden_cache_;    // sortie W1 avant activation
    std::vector<Tensor> ff_out_cache_;       // sortie W2

    // placeholder abstrait
    Tensor dummy_;
};

#endif // TRANSFORMER_H

