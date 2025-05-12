#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "module.h"
#include "tensor.h"
#include "linear.h"
#include "attention.h"
#include <vector>
#include <memory>

class Transformer : public Module {
public:
    Transformer(int d_model, int num_heads, int num_layers, int d_ff);
    void forward(Tensor& input, Tensor& output) override;
    void backward(Tensor& grad_output, Tensor& grad_input) override;
    void update(float lr) override;
    Tensor& get_weights() override { return weights_; }
    Tensor& get_grad_weights() override { return grad_weights_; }
    void set_weights(const Tensor& weights) override;
    void save(const std::string& path) override;
    void load(const std::string& path) override;

private:
    void add_positional_encoding(Tensor& input);
    int d_model_;
    int num_heads_;
    int num_layers_;
    int d_ff_;
    std::vector<std::shared_ptr<MultiHeadAttention>> attention_layers_;
    std::vector<std::shared_ptr<Linear>> feed_forward_layers_;
    std::vector<std::shared_ptr<ReLU>> ff_activations_;
    Tensor weights_;
    Tensor grad_weights_;
};

#endif
