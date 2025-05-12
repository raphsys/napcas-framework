#ifndef ATTENTION_H
#define ATTENTION_H

#include "tensor.h"
#include "module.h"
#include <memory>

class MultiHeadAttention : public Module {
public:
    MultiHeadAttention(int d_model, int num_heads);
    void forward(Tensor& query, Tensor& key, Tensor& value, Tensor& output) override;
    void backward(Tensor& grad_output, Tensor& grad_input) override;
    void update(float lr) override;
    Tensor& get_weights() override { return weights_; }
    Tensor& get_grad_weights() override { return grad_weights_; }
    void set_weights(const Tensor& weights) override;
    void save(const std::string& path) override;
    void load(const std::string& path) override;

private:
    int d_model_;
    int num_heads_;
    int d_k_;
    std::vector<std::shared_ptr<Linear>> w_q_;
    std::vector<std::shared_ptr<Linear>> w_k_;
    std::vector<std::shared_ptr<Linear>> w_v_;
    std::shared_ptr<Linear> w_o_;
    Tensor weights_;
    Tensor grad_weights_;
    Tensor attention_weights_; // For backward pass
};

#endif
