#include "transformer.h"
#include "activation.h"
#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>
#include <fstream>
#include <vector>

Transformer::Transformer(int d_model, int num_heads, int num_layers, int d_ff)
    : d_model_(d_model), num_heads_(num_heads), num_layers_(num_layers), d_ff_(d_ff) {
    for (int i = 0; i < num_layers; ++i) {
        attention_layers_.push_back(std::make_shared<MultiHeadAttention>(d_model, num_heads));
        feed_forward_layers_.push_back(std::make_shared<Linear>(d_model, d_ff));
        ff_activations_.push_back(std::make_shared<ReLU>());
        feed_forward_layers_.push_back(std::make_shared<Linear>(d_ff, d_model));
    }
    weights_ = Tensor({num_layers, 1}); // Placeholder for serialization
    grad_weights_ = Tensor({num_layers, 1});
    // Pre-allocate reusable buffers
    temp_buffer_ = Tensor({1, 1, d_model});
    attn_output_buffer_ = Tensor({1, 1, d_model});
    ff_output_buffer_ = Tensor({1, 1, d_ff});
}

void Transformer::add_positional_encoding(Tensor& input) {
    int seq_len = input.shape()[0];
    int batch_size = input.shape()[1];
    for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < d_model_; ++j) {
            float pos = static_cast<float>(i);
            float dim = static_cast<float>(j);
            float div_term = std::pow(10000.0f, 2.0f * dim / d_model_);
            for (int b = 0; b < batch_size; ++b) {
                if (j % 2 == 0) {
                    input[i * batch_size * d_model_ + b * d_model_ + j] += std::sin(pos / div_term);
                } else {
                    input[i * batch_size * d_model_ + b * d_model_ + j] += std::cos(pos / div_term);
                }
            }
        }
    }
}

void Transformer::forward(Tensor& input, Tensor& output) {
    if (input.ndim() != 3 || input.shape()[2] != d_model_) {
        throw std::invalid_argument("Input must be 3D (seq_len, batch_size, d_model)");
    }
    int seq_len = input.shape()[0];
    int batch_size = input.shape()[1];
    if (output.shape() != std::vector<int>{seq_len, batch_size, d_model_}) {
        throw std::invalid_argument("Output shape mismatch");
    }

    // Resize reusable buffers if needed
    temp_buffer_.reshape({seq_len, batch_size, d_model_});
    attn_output_buffer_.reshape({seq_len, batch_size, d_model_});
    ff_output_buffer_.reshape({seq_len, batch_size, d_ff_});
    temp_buffer_ = input; // Copy input to temp buffer
    add_positional_encoding(temp_buffer_);

    for (int i = 0; i < num_layers_; ++i) {
        attention_layers_[i]->forward(temp_buffer_, temp_buffer_, temp_buffer_, attn_output_buffer_);
        // Add residual connection
        for (size_t j = 0; j < temp_buffer_.size(); ++j) {
            temp_buffer_[j] += attn_output_buffer_[j];
        }
        // Feed-forward
        feed_forward_layers_[i * 2]->forward(temp_buffer_, ff_output_buffer_);
        ff_activations_[i]->forward(ff_output_buffer_, ff_output_buffer_);
        feed_forward_layers_[i * 2 + 1]->forward(ff_output_buffer_, attn_output_buffer_);
        // Add residual connection
        for (size_t j = 0; j < temp_buffer_.size(); ++j) {
            temp_buffer_[j] += attn_output_buffer_[j];
        }
    }

    output = temp_buffer_;
}

void Transformer::backward(Tensor& grad_output, Tensor& grad_input) {
    int seq_len = grad_output.shape()[0];
    int batch_size = grad_output.shape()[1];
    temp_buffer_.reshape({seq_len, batch_size, d_model_});
    attn_output_buffer_.reshape({seq_len, batch_size, d_model_});
    ff_output_buffer_.reshape({seq_len, batch_size, d_ff_});
    temp_buffer_ = grad_output;

    for (int i = num_layers_ - 1; i >= 0; --i) {
        // Backward through second residual connection
        Tensor grad_ff_output({seq_len, batch_size, d_ff_});
        feed_forward_layers_[i * 2 + 1]->backward(temp_buffer_, grad_ff_output);
        ff_activations_[i]->backward(grad_ff_output, grad_ff_output);
        feed_forward_layers_[i * 2]->backward(grad_ff_output, attn_output_buffer_);
        // Add residual gradient
        for (size_t j = 0; j < temp_buffer_.size(); ++j) {
            temp_buffer_[j] += attn_output_buffer_[j];
        }
        // Backward through attention and first residual connection
        attention_layers_[i]->backward(temp_buffer_, attn_output_buffer_);
        for (size_t j = 0; j < temp_buffer_.size(); ++j) {
            temp_buffer_[j] += attn_output_buffer_[j];
        }
    }

    grad_input = temp_buffer_; // Positional encoding is not differentiable
}

void Transformer::update(float lr) {
    for (auto& layer : attention_layers_) {
        layer->update(lr);
    }
    for (auto& layer : feed_forward_layers_) {
        layer->update(lr);
    }
    for (auto& act : ff_activations_) {
        act->update(lr);
    }
}

void Transformer::set_weights(const Tensor& weights) {
    weights_ = weights;
}

void Transformer::save(const std::string& path) {
    for (size_t i = 0; i < attention_layers_.size(); ++i) {
        attention_layers_[i]->save(path + "_attention_" + std::to_string(i));
        feed_forward_layers_[i * 2]->save(path + "_ff1_" + std::to_string(i));
        feed_forward_layers_[i * 2 + 1]->save(path + "_ff2_" + std::to_string(i));
    }
}

void Transformer::load(const std::string& path) {
    for (size_t i = 0; i < attention_layers_.size(); ++i) {
        attention_layers_[i]->load(path + "_attention_" + std::to_string(i));
        feed_forward_layers_[i * 2]->load(path + "_ff1_" + std::to_string(i));
        feed_forward_layers_[i * 2 + 1]->load(path + "_ff2_" + std::to_string(i));
    }
}
