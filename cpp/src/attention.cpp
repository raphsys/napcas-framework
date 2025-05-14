// File: cpp/src/attention.cpp

#include "attention.h"
#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>
#include <algorithm>

MultiHeadAttention::MultiHeadAttention(int d_model, int num_heads)
    : d_model_(d_model),
      num_heads_(num_heads),
      d_k_(d_model / num_heads),
      chunk_size_(512),
      weights_({d_model, d_model}),
      grad_weights_({d_model, d_model})
{
    if (d_model_ % num_heads_ != 0) {
        throw std::invalid_argument("d_model must be divisible by num_heads");
    }
    w_q_.reserve(num_heads_);
    w_k_.reserve(num_heads_);
    w_v_.reserve(num_heads_);
    for (int i = 0; i < num_heads_; ++i) {
        w_q_.push_back(std::make_shared<Linear>(d_model_, d_k_));
        w_k_.push_back(std::make_shared<Linear>(d_model_, d_k_));
        w_v_.push_back(std::make_shared<Linear>(d_model_, d_k_));
    }
    w_o_ = std::make_shared<Linear>(d_model_, d_model_);
}

void MultiHeadAttention::forward(Tensor& input, Tensor& output) {
    // self-attention
    forward(input, input, input, output);
}

void MultiHeadAttention::forward(Tensor& query,
                                 Tensor& key,
                                 Tensor& value,
                                 Tensor& output)
{
    // Vérification des dimensions
    if (query.ndim() != 3 || key.ndim() != 3 || value.ndim() != 3) {
        throw std::invalid_argument("Inputs must be 3D (seq_len, batch, d_model)");
    }
    int seq_len = query.shape()[0];
    int batch   = query.shape()[1];
    if (output.shape() != std::vector<int>{seq_len, batch, d_model_}) {
        throw std::invalid_argument("Output shape mismatch");
    }

    // Vider les caches
    q_cache_.clear(); k_cache_.clear(); v_cache_.clear();

    // Préparer le tensor de poids d'attention
    attention_weights_.reshape({seq_len, seq_len, batch, num_heads_});

    // Stockage temporaire des sorties de chaque tête
    std::vector<Tensor> heads;
    heads.reserve(num_heads_);

    // 1) Projections Q, K, V + caches
    for (int h = 0; h < num_heads_; ++h) {
        Tensor q({seq_len, batch, d_k_});
        Tensor k({seq_len, batch, d_k_});
        Tensor v({seq_len, batch, d_k_});
        w_q_[h]->forward(query, q);
        w_k_[h]->forward(key,   k);
        w_v_[h]->forward(value, v);
        q_cache_.push_back(q);
        k_cache_.push_back(k);
        v_cache_.push_back(v);

        // 2) Attention échelonnée (chunked) pour économiser la mémoire
        Tensor head_out({seq_len, batch, d_k_});
        for (int start = 0; start < seq_len; start += chunk_size_) {
            int end = std::min(start + chunk_size_, seq_len);
            int len = end - start;

            // Map à Eigen
            Eigen::Map<Eigen::MatrixXf> qmat(
                q.data().data() + start * batch * d_k_, len, batch * d_k_);
            Eigen::Map<Eigen::MatrixXf> kmat(
                k.data().data(), seq_len, batch * d_k_);
            Eigen::Map<Eigen::MatrixXf> smat(
                attention_weights_.data().data() +
                    h * seq_len * seq_len * batch +
                    start * seq_len * batch,
                len, seq_len * batch);

            // Produit scalaire / √d_k
            smat = (qmat * kmat.transpose()) / std::sqrt(static_cast<float>(d_k_));

            // Softmax par position et batch
            for (int b = 0; b < batch; ++b) {
                for (int i = 0; i < len; ++i) {
                    // max pour stabilité
                    float m = smat(i, b);
                    for (int j = 1; j < seq_len; ++j) {
                        m = std::max(m, smat(i, j * batch + b));
                    }
                    // exponentielle et normalisation
                    float sum = 0.0f;
                    for (int j = 0; j < seq_len; ++j) {
                        float e = std::exp(smat(i, j * batch + b) - m);
                        smat(i, j * batch + b) = e;
                        sum += e;
                    }
                    for (int j = 0; j < seq_len; ++j) {
                        smat(i, j * batch + b) /= sum;
                    }
                }
            }

            // Multiplication par V
            Eigen::Map<Eigen::MatrixXf> vmat(
                v.data().data(), seq_len, batch * d_k_);
            Eigen::Map<Eigen::MatrixXf> hmat(
                head_out.data().data() + start * batch * d_k_, len, batch * d_k_);
            hmat = smat * vmat;
        }
        heads.push_back(std::move(head_out));
    }

    // 3) Concaténation des têtes → concat_cache_
    concat_cache_.reshape({seq_len, batch, d_model_});
    for (int i = 0; i < seq_len; ++i) {
        for (int b = 0; b < batch; ++b) {
            for (int h = 0; h < num_heads_; ++h) {
                for (int k = 0; k < d_k_; ++k) {
                    int out_idx  = i * batch * d_model_ + b * d_model_ + h * d_k_ + k;
                    int head_idx = i * batch * d_k_ + b * d_k_ + k;
                    concat_cache_.data()[out_idx] = heads[h].data()[head_idx];
                }
            }
        }
    }

    // 4) Projection finale
    w_o_->forward(concat_cache_, output);
}

void MultiHeadAttention::backward(Tensor& grad_output, Tensor& grad_input) {
    // grad_output : [seq_len, batch, d_model_]
    int seq_len = concat_cache_.shape()[0];
    int batch   = concat_cache_.shape()[1];

    // Initialiser grad_input à zéro
    grad_input = Tensor(concat_cache_.shape());
    grad_input.fill(0.0f);

    // 1) Backward projection finale
    Tensor grad_concat(concat_cache_.shape());
    w_o_->backward(grad_output, grad_concat);

    // 2) Séparer grad_concat en un grad par tête
    std::vector<Tensor> grad_heads;
    grad_heads.reserve(num_heads_);
    for (int h = 0; h < num_heads_; ++h) {
        grad_heads.emplace_back(Tensor({seq_len, batch, d_k_}));
    }
    for (int i = 0; i < seq_len; ++i) {
        for (int b = 0; b < batch; ++b) {
            for (int h = 0; h < num_heads_; ++h) {
                for (int k = 0; k < d_k_; ++k) {
                    int idx_concat = i * batch * d_model_ + b * d_model_ + h * d_k_ + k;
                    int idx_head   = i * batch * d_k_   + b * d_k_   + k;
                    grad_heads[h].data()[idx_head] = grad_concat.data()[idx_concat];
                }
            }
        }
    }

    // 3) Pour chaque tête, backprop à travers attention
    float scale = 1.0f / std::sqrt(static_cast<float>(d_k_));
    for (int h = 0; h < num_heads_; ++h) {
        // Récupérer caches
        auto& Q = q_cache_[h];
        auto& K = k_cache_[h];
        auto& V = v_cache_[h];

        // 3a) Gradients initiaux
        Tensor grad_P({seq_len, seq_len, batch});
        grad_P.fill(0.0f);
        Tensor grad_V({seq_len, batch, d_k_});
        grad_V.fill(0.0f);

        // 3b) Calcul de grad_P et grad_V
        for (int i = 0; i < seq_len; ++i) {
            for (int b = 0; b < batch; ++b) {
                for (int k = 0; k < d_k_; ++k) {
                    int idx_head = i * batch * d_k_ + b * d_k_ + k;
                    float g = grad_heads[h].data()[idx_head];
                    for (int j = 0; j < seq_len; ++j) {
                        int idx_P = ((i * seq_len + j) * batch + b);
                        float Pij = attention_weights_.data()[idx_P * num_heads_ + h];
                        int idx_V = (j * batch * d_k_) + (b * d_k_) + k;
                        float Vjk = V.data()[idx_V];
                        grad_P.data()[idx_P]       += g * Vjk;
                        grad_V.data()[idx_V]       += g * Pij;
                    }
                }
            }
        }

        // 3c) Softmax backward → grad_S
        Tensor grad_S({seq_len, seq_len, batch});
        grad_S.fill(0.0f);
        for (int i = 0; i < seq_len; ++i) {
            for (int b = 0; b < batch; ++b) {
                for (int j = 0; j < seq_len; ++j) {
                    int idx_P = ((i * seq_len + j) * batch + b);
                    float dSj = 0.0f;
                    for (int l = 0; l < seq_len; ++l) {
                        int idx_Pl = ((i * seq_len + l) * batch + b);
                        float gradPl = grad_P.data()[idx_Pl];
                        float Pil    = attention_weights_.data()[idx_Pl * num_heads_ + h];
                        float Pij    = attention_weights_.data()[idx_P  * num_heads_ + h];
                        if (l == j) {
                            dSj += gradPl * Pil * (1.0f - Pij);
                        } else {
                            dSj -= gradPl * Pil * Pij;
                        }
                    }
                    grad_S.data()[idx_P] = dSj;
                }
            }
        }

        // 3d) Backprop scaled dot-product → grad_Q et grad_K
        Tensor grad_Q({seq_len, batch, d_k_});
        Tensor grad_K({seq_len, batch, d_k_});
        grad_Q.fill(0.0f);
        grad_K.fill(0.0f);
        for (int i = 0; i < seq_len; ++i) {
            for (int b = 0; b < batch; ++b) {
                for (int j = 0; j < seq_len; ++j) {
                    int idx_S = ((i * seq_len + j) * batch + b);
                    float dS = grad_S.data()[idx_S];
                    for (int k = 0; k < d_k_; ++k) {
                        int idx_Q = (i * batch * d_k_) + (b * d_k_) + k;
                        int idx_K = (j * batch * d_k_) + (b * d_k_) + k;
                        float Kjk = K.data()[idx_K];
                        float Qik = Q.data()[idx_Q];
                        grad_Q.data()[idx_Q] += dS * Kjk * scale;
                        grad_K.data()[idx_K] += dS * Qik * scale;
                    }
                }
            }
        }

        // 3e) Backprop à travers Q/K/V linears
        Tensor grad_in_q(concat_cache_.shape()); grad_in_q.fill(0.0f);
        Tensor grad_in_k(concat_cache_.shape()); grad_in_k.fill(0.0f);
        Tensor grad_in_v(concat_cache_.shape()); grad_in_v.fill(0.0f);
        w_q_[h]->backward(grad_Q, grad_in_q);
        w_k_[h]->backward(grad_K, grad_in_k);
        w_v_[h]->backward(grad_V, grad_in_v);

        // 3f) Accumuler dans grad_input
        for (int idx = 0; idx < grad_input.size(); ++idx) {
            grad_input.data()[idx] +=
                grad_in_q.data()[idx] +
                grad_in_k.data()[idx] +
                grad_in_v.data()[idx];
        }
    }
}

void MultiHeadAttention::update(float lr) {
    for (int h = 0; h < num_heads_; ++h) {
        w_q_[h]->update(lr);
        w_k_[h]->update(lr);
        w_v_[h]->update(lr);
    }
    w_o_->update(lr);
}

void MultiHeadAttention::save(const std::string& path) {
    for (int h = 0; h < num_heads_; ++h) {
        w_q_[h]->save(path + "_w_q_" + std::to_string(h));
        w_k_[h]->save(path + "_w_k_" + std::to_string(h));
        w_v_[h]->save(path + "_w_v_" + std::to_string(h));
    }
    w_o_->save(path + "_w_o");
}

void MultiHeadAttention::load(const std::string& path) {
    for (int h = 0; h < num_heads_; ++h) {
        w_q_[h]->load(path + "_w_q_" + std::to_string(h));
        w_k_[h]->load(path + "_w_k_" + std::to_string(h));
        w_v_[h]->load(path + "_w_v_" + std::to_string(h));
    }
    w_o_->load(path + "_w_o");
}

