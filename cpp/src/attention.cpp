#include "attention.h"
#include "linear.h"
#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>
#include <fstream>
#include <algorithm>

MultiHeadAttention::MultiHeadAttention(int d_model, int num_heads)
    : d_model_(d_model), num_heads_(num_heads), d_k_(d_model / num_heads) {
    if (d_model % num_heads != 0) {
        throw std::invalid_argument("d_model must be divisible by num_heads");
    }
    for (int i = 0; i < num_heads; ++i) {
        w_q_.push_back(std::make_shared<Linear>(d_model, d_k_));
        w_k_.push_back(std::make_shared<Linear>(d_model, d_k_));
        w_v_.push_back(std::make_shared<Linear>(d_model, d_k_));
    }
    w_o_ = std::make_shared<Linear>(d_model, d_model);
    weights_ = Tensor({d_model, d_model}); // Placeholder for serialization
    grad_weights_ = Tensor({d_model, d_model});
    chunk_size_ = 512; // Process 512 query tokens at a time
}

void MultiHeadAttention::forward(Tensor& query, Tensor& key, Tensor& value, Tensor& output) {
    if (query.ndim() != 3 || key.ndim() != 3 || value.ndim() != 3) {
        throw std::invalid_argument("Inputs must be 3D (seq_len, batch_size, d_model)");
    }
    int seq_len = query.shape()[0];
    int batch_size = query.shape()[1];
    if (output.shape() != std::vector<int>{seq_len, batch_size, d_model_}) {
        throw std::invalid_argument("Output shape mismatch");
    }

    std::vector<Tensor> heads(num_heads_);
    attention_weights_.reshape({seq_len, seq_len, batch_size, num_heads_});

    for (int h = 0; h < num_heads_; ++h) {
        Tensor q({seq_len, batch_size, d_k_});
        Tensor k({seq_len, batch_size, d_k_});
        Tensor v({seq_len, batch_size, d_k_});
        w_q_[h]->forward(query, q);
        w_k_[h]->forward(key, k);
        w_v_[h]->forward(value, v);

        heads[h] = Tensor({seq_len, batch_size, d_k_});
        // Process attention in chunks to reduce memory usage
        for (int start = 0; start < seq_len; start += chunk_size_) {
            int end = std::min(start + chunk_size_, seq_len);
            int chunk_len = end - start;

            // Compute scores for chunk
            Tensor scores({chunk_len, seq_len, batch_size});
            Eigen::Map<Eigen::MatrixXf> q_mat(q.data().data() + start * batch_size * d_k_, chunk_len, batch_size * d_k_);
            Eigen::Map<Eigen::MatrixXf> k_mat(k.data().data(), seq_len, batch_size * d_k_);
            Eigen::Map<Eigen::MatrixXf> scores_mat(scores.data().data(), chunk_len, seq_len * batch_size);
            scores_mat = (q_mat * k_mat.transpose()) / std::sqrt(static_cast<float>(d_k_));

            // Softmax
            for (int b = 0; b < batch_size; ++b) {
                for (int i = 0; i < chunk_len; ++i) {
                    float max_score = scores[i * seq_len * batch_size].data()[0];
                    for (int j = 0; j < seq_len; ++j) {
                        max_score = std::max(max_score, scores[i * seq_len * batch_size + j * batch_size + b]);
                    }
                    float sum_exp = 0.0f;
                    for (int j = 0; j < seq_len; ++j) {
                        float exp_val = std::exp(scores[i * seq_len * batch_size + j * batch_size + b] - max_score);
                        attention_weights_[(start + i) * seq_len * batch_size * num_heads_ +
                                           j * batch_size * num_heads_ + b * num_heads_ + h] = exp_val;
                        sum_exp += exp_val;
                    }
                    for (int j = 0; j < seq_len; ++j) {
                        attention_weights_[(start + i) * seq_len * batch_size * num_heads_ +
                                           j * batch_size * num_heads_ + b * num_heads_ + h] /= sum_exp;
                    }
                }
            }

            // Apply attention weights to values
            Eigen::Map<Eigen::MatrixXf> v_mat(v.data().data(), seq_len, batch_size * d_k_);
            Eigen::Map<Eigen::MatrixXf> head_mat(heads[h].data().data() + start * batch_size * d_k_, chunk_len, batch_size * d_k_);
            Eigen::Map<Eigen::MatrixXf> attn_mat(attention_weights_.data().data() +
                                                 h * seq_len * seq_len * batch_size +
                                                 start * seq_len * batch_size, chunk_len, seq_len * batch_size);
            head_mat = attn_mat * v_mat;
        }
    }

    // Concatenate heads
    Tensor concat({seq_len, batch_size, d_model_});
    for (int h = 0; h < num_heads_; ++h) {
        for (int i = 0; i < seq_len; ++i) {
            for (int b = 0; b < batch_size; ++b) {
                for (int j = 0; j < d_k_; ++j) {
                    concat[i * batch_size * d_model_ + b * d_model_ + h * d_k_ + j] = heads[h][i * batch_size * d_k_ + b * d_k_ + j];
                }
            }
        }
    }

    w_o_->forward(concat, output);
}

void MultiHeadAttention::backward(Tensor& grad_output, Tensor& grad_input) {
    int seq_len = grad_output.shape()[0];
    int batch_size = grad_output.shape()[1];

    // Backward through output linear layer
    Tensor grad_concat({seq_len, batch_size, d_model_});
    w_o_->backward(grad_output, grad_concat);

    // Split gradient to heads
    std::vector<Tensor> grad_heads(num_heads_, Tensor({seq_len, batch_size, d_k_}));
    for (int h = 0; h < num_heads_; ++h) {
        for (int i = 0; i < seq_len; ++i) {
            for (int b = 0; b < batch_size; ++b) {
                for (int j = 0; j < d_k_; ++j) {
                    grad_heads[h][i * batch_size * d_k_ + b * d_k_ + j] =
                        grad_concat[i * batch_size * d_model_ + b * d_model_ + h * d_k_ + j];
                }
            }
        }
    }

    // Backward through attention
    Tensor grad_query({seq_len, batch_size, d_model_});
    Tensor grad_key({seq_len, batch_size, d_model_});
    Tensor grad_value({seq_len, batch_size, d_model_});
    grad_query.fill(0.0f);
    grad_key.fill(0.0f);
    grad_value.fill(0.0f);

    for (int h = 0; h < num_heads_; ++h) {
        Tensor q({seq_len, batch_size, d_k_});
        Tensor k({seq_len, batch_size, d_k_});
        w_q_[h]->forward(grad_query, q);
        w_k_[h]->forward(grad_key, k);

        for (int start = 0; start < seq_len; start += chunk_size_) {
            int end = std::min(start + chunk_size_, seq_len);
            int chunk_len = end - start;

            Tensor grad_v({chunk_len, batch_size, d_k_});
            Eigen::Map<Eigen::MatrixXf> grad_head_mat(grad_heads[h].data().data() + start * batch_size * d_k_,
                                                      chunk_len, batch_size * d_k_);
            Eigen::Map<Eigen::MatrixXf> attn_mat(attention_weights_.data().data() +
                                                 h * seq_len * seq_len * batch_size +
                                                 start * seq_len * batch_size, chunk_len, seq_len * batch_size);
            Eigen::Map<Eigen::MatrixXf> grad_v_mat(grad_v.data().data(), chunk_len, batch_size * d_k_);
            grad_v_mat = attn_mat.transpose() * grad_head_mat;

            Tensor grad_attn({chunk_len, seq_len, batch_size});
            for (int b = 0; b < batch_size; ++b) {
                for (int i = 0; i < chunk_len; ++i) {
                    float sum_grad = 0.0f;
                    for (int j = 0; j < seq_len; ++j) {
                        float attn = attention_weights_[(start + i) * seq_len * batch_size * num_heads_ +
                                                        j * batch_size * num_heads_ + b * num_heads_ + h];
                        float grad = grad_heads[h][(start + i) * batch_size * d_k_ + b * d_k_];
                        grad_attn[i * seq_len * batch_size + j * batch_size + b] = attn * (1.0f - attn) * grad;
                        sum_grad += grad_attn[i * seq_len * batch_size + j * batch_size + b];
                    }
                    for (int j = 0; j < seq_len; ++j) {
                        grad_attn[i * seq_len * batch_size + j * batch_size + b] -= attn * sum_grad;
                    }
                }
            }

            Tensor grad_q({chunk_len, batch_size, d_k_});
            Eigen::Map<Eigen::MatrixXf> grad_attn_mat(grad_attn.data().data(), chunk_len, seq_len * batch_size);
            Eigen::Map<Eigen::MatrixXf> k_mat(k.data().data(), seq_len, batch_size * d_k_);
            Eigen::Map<Eigen::MatrixXf> grad_q_mat(grad_q.data().data(), chunk_len, batch_size * d_k_);
            grad_q_mat = grad_attn_mat * k_mat / std::sqrt(static_cast<float>(d_k_));

            Tensor grad_k({seq_len, batch_size, d_k_});
            Eigen::Map<Eigen::MatrixXf> q_mat(q.data().data() + start * batch_size * d_k_, chunk_len, batch_size * d_k_);
            Eigen::Map<Eigen::MatrixXf> grad_k_mat(grad_k.data().data(), seq_len, batch_size * d_k_);
            grad_k_mat = grad_attn_mat.transpose() * q_mat / std::sqrt(static_cast<float>(d_k_));

            Tensor grad_v_full({seq_len, batch_size, d_k_});
            for (int i = 0; i < chunk_len; ++i) {
                for (int b = 0; b < batch_size; ++b) {
                    for (int j = 0; j < d_k_; ++j) {
                        grad_v_full[(start + i) * batch_size * d_k_ + b * d_k_ + j] = grad_v[i * batch_size * d_k_ + b * d_k_ + j];
                    }
                }
            }

            w_v_[h]->backward(grad_v_full, grad_value);
            Tensor grad_q_full({seq_len, batch_size, d_k_});
            for (int i = 0; i < chunk_len; ++i) {
                for (int b = 0; b < batch_size; ++b) {
                    for (int j = 0; j < d_k_; ++j) {
                        grad_q_full[(start + i) * batch_size * d_k_ + b * d_k_ + j] = grad_q[i * batch_size * d_k_ + b * d_k_ + j];
                    }
                }
            }
            w_q_[h]->backward(grad_q_full, grad_query);
            w_k_[h]->backward(grad_k, grad_key);
        }
    }

    grad_input = grad_query; // Assuming query = key = value for simplicity
}

void MultiHeadAttention::update(float lr) {
    for (int h = 0; h < num_heads_; ++h) {
        w_q_[h]->update(lr);
        w_k_[h]->update(lr);
        w_v_[h]->update(lr);
    }
    w_o_->update(lr);
}

void MultiHeadAttention::set_weights(const Tensor& weights) {
    weights_ = weights;
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
