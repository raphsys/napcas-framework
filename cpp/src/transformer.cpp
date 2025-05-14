// File: cpp/src/transformer.cpp

#include "transformer.h"
#include <stdexcept>
#include <algorithm>

Transformer::Transformer(int d_model,
                         int num_heads,
                         int d_ff,
                         int num_layers)
    : d_model_(d_model),
      num_heads_(num_heads),
      d_ff_(d_ff),
      num_layers_(num_layers),
      dummy_(Tensor())
{
    if (d_model_ % num_heads_ != 0) {
        throw std::invalid_argument("d_model doit être divisible par num_heads");
    }
    attention_layers_.reserve(num_layers_);
    ff_w1_.reserve(num_layers_);
    ff_w2_.reserve(num_layers_);
    ff_activations_.reserve(num_layers_);

    for (int i = 0; i < num_layers_; ++i) {
        attention_layers_.push_back(
            std::make_shared<MultiHeadAttention>(d_model_, num_heads_));
        ff_w1_.push_back(std::make_shared<Linear>(d_model_, d_ff_));
        ff_activations_.push_back(std::make_shared<ReLU>());
        ff_w2_.push_back(std::make_shared<Linear>(d_ff_, d_model_));
    }
}

void Transformer::forward(Tensor& input, Tensor& output) {
    // shape must be [seq_len, batch, d_model]
    if (input.ndim() != 3 || input.shape()[2] != d_model_) {
        throw std::invalid_argument("Input Transformer invalide");
    }
    int seq_len = input.shape()[0];
    int batch   = input.shape()[1];

    // Vider les caches
    residual1_cache_.clear();
    attn_out_cache_.clear();
    residual2_cache_.clear();
    ff_hidden_cache_.clear();
    ff_out_cache_.clear();

    // Copie de travail
    Tensor x = input;

    // Parcours de chaque couche
    for (int l = 0; l < num_layers_; ++l) {
        // 1) Self-Attention + résidu
        residual1_cache_.push_back(x);                // x_before_attn
        Tensor attn_out(x.shape());                   
        attention_layers_[l]->forward(x, attn_out);   // attn_out_cache_[l]
        attn_out_cache_.push_back(attn_out);
        // x = x + attn_out
        for (int i = 0; i < x.size(); ++i) {
            x[i] = residual1_cache_[l][i] + attn_out[i];
        }
        residual2_cache_.push_back(x);                // x_before_ff

        // 2) Feed-Forward + résidu
        // 2a) W1
        Tensor hidden({seq_len, batch, d_ff_});
        ff_w1_[l]->forward(x, hidden);
        ff_hidden_cache_.push_back(hidden);
        // 2b) Activation
        Tensor activated(hidden.shape());
        ff_activations_[l]->forward(hidden, activated);
        // 2c) W2
        Tensor ff_out(x.shape());                     
        ff_w2_[l]->forward(activated, ff_out);
        ff_out_cache_.push_back(ff_out);
        // 2d) Résidu
        for (int i = 0; i < x.size(); ++i) {
            x[i] = residual2_cache_[l][i] + ff_out[i];
        }
    }

    // Résultat final
    output = x;
}

void Transformer::backward(Tensor& grad_output, Tensor& grad_input) {
    // grad_x = grad_output en topo
    Tensor grad_x = grad_output;

    for (int li = num_layers_ - 1; li >= 0; --li) {
        // dimensions
        int sz = grad_x.size();

        // 2) Backward feed-forward
        // 2c) W2.backward
        Tensor grad_ff_out = grad_x;                     // gradient sur W2 output
        Tensor grad_activated(grad_ff_out.shape());
        ff_w2_[li]->backward(grad_ff_out, grad_activated);

        // 2b) Activation.backward
        Tensor grad_hidden(grad_activated.shape());
        ff_activations_[li]->backward(grad_activated, grad_hidden);

        // 2a) W1.backward
        Tensor grad_pre_ff(grad_hidden.shape());
        ff_w1_[li]->backward(grad_hidden, grad_pre_ff);

        // Résidu FF: x_pre_ff = residual2_cache_[li]
        // grad_res2   = grad_x
        Tensor grad_res2 = grad_x;
        // grad_to_attn_out = grad_x (identique pour résidu)
        // grad_to_attn_in  = grad_pre_ff
        // gradient total vers x_before_attn = grad_res2 + grad_pre_ff
        Tensor grad_before_attn(grad_x.shape());
        for (int i = 0; i < sz; ++i) {
            grad_before_attn[i] = grad_res2[i] + grad_pre_ff[i];
        }

        // 1) Backward attention
        // grad passed to attention output = grad_before_attn
        Tensor grad_attn_out = grad_before_attn;
        Tensor grad_pre_attn(grad_attn_out.shape());
        attention_layers_[li]->backward(grad_attn_out, grad_pre_attn);

        // Résidu attention: 
        // residual1 cache = input to attention
        // gradient total vers x_prev = grad_pre_attn + grad_before_attn
        Tensor grad_prev(grad_pre_attn.shape());
        for (int i = 0; i < sz; ++i) {
            grad_prev[i] = grad_pre_attn[i] + grad_before_attn[i];
        }

        // Préparer pour la couche précédente
        grad_x = grad_prev;
    }

    // Renvoyer le gradient sur l'input global
    grad_input = grad_x;
}

void Transformer::update(float lr) {
    // mise à jour de tous les sous-modules
    for (int l = 0; l < num_layers_; ++l) {
        attention_layers_[l]->update(lr);
        ff_w1_[l]->update(lr);
        ff_activations_[l]->update(lr);  // pas d’effet, no-op
        ff_w2_[l]->update(lr);
    }
}

