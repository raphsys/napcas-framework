// File: cpp/include/attention.h

#ifndef ATTENTION_H
#define ATTENTION_H

#include "module.h"
#include "tensor.h"
#include "linear.h"
#include <memory>
#include <vector>
#include <string>

/// @brief Multi-head self-attention module.
class MultiHeadAttention : public Module {
public:
    /// @param d_model    Dimension du modèle.
    /// @param num_heads  Nombre de têtes.
    MultiHeadAttention(int d_model, int num_heads);

    /// @brief Forward (self-attention, 2-args).  
    void forward(Tensor& input, Tensor& output) override;

    /// @brief Forward (Q, K, V explicites, 4-args).
    void forward(Tensor& query,
                 Tensor& key,
                 Tensor& value,
                 Tensor& output);

    /// @brief Backward complet pour self-attention.
    void backward(Tensor& grad_output, Tensor& grad_input) override;

    /// @brief Met à jour tous les poids internes.
    void update(float lr) override;

    Tensor& get_weights() override { return weights_; }
    Tensor& get_grad_weights() override { return grad_weights_; }
    void set_weights(const Tensor& weights) override { weights_ = weights; }

    void save(const std::string& path) override;
    void load(const std::string& path) override;

private:
    int d_model_;
    int num_heads_;
    int d_k_;
    int chunk_size_;

    // projections Q/K/V et projection finale
    std::vector<std::shared_ptr<Linear>> w_q_, w_k_, w_v_;
    std::shared_ptr<Linear> w_o_;

    // placeholders pour sérialisation
    Tensor weights_;
    Tensor grad_weights_;

    // caches pour backward
    Tensor attention_weights_;           // [seq_len, seq_len, batch, num_heads]
    std::vector<Tensor> q_cache_,         // un Tensor par tête: [seq_len, batch, d_k]
                        k_cache_,
                        v_cache_;
    Tensor concat_cache_;                // concaténation avant w_o_: [seq_len, batch, d_model]
};

#endif // ATTENTION_H

