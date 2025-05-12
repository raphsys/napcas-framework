#ifndef POOLING_H
#define POOLING_H

#include "module.h"
#include "tensor.h"

class MaxPool2d : public Module {
public:
    MaxPool2d(int kernel_size, int stride);
    void forward(Tensor& input, Tensor& output) override;
    void backward(Tensor& grad_output, Tensor& grad_input) override;
    void update(float lr) override {}
    Tensor& get_weights() override { return weights_; }
    Tensor& get_grad_weights() override { return grad_weights_; }
    void set_weights(const Tensor& weights) override {}
    void save(const std::string& path) override {}
    void load(const std::string& path) override {}

private:
    int kernel_size_;
    int stride_;
    Tensor weights_; // Placeholder
    Tensor grad_weights_; // Placeholder
    std::vector<std::vector<int>> max_indices_; // For backward pass
};

#endif
