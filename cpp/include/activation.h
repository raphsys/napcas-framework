#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "module.h"
#include "tensor.h"

class ReLU : public Module {
public:
    ReLU();
    void forward(Tensor& input, Tensor& output) override;
    void backward(Tensor& grad_output, Tensor& grad_input) override;
    void update(float lr) override;
    Tensor& get_weights() override;
    Tensor& get_grad_weights() override;
    void set_weights(const Tensor& weights) override;
};

class Sigmoid : public Module {
public:
    Sigmoid();
    void forward(Tensor& input, Tensor& output) override;
    void backward(Tensor& grad_output, Tensor& grad_input) override;
    void update(float lr) override;
    Tensor& get_weights() override;
    Tensor& get_grad_weights() override;
    void set_weights(const Tensor& weights) override;
};

class Tanh : public Module {
public:
    Tanh();
    void forward(Tensor& input, Tensor& output) override;
    void backward(Tensor& grad_output, Tensor& grad_input) override;
    void update(float lr) override;
    Tensor& get_weights() override;
    Tensor& get_grad_weights() override;
    void set_weights(const Tensor& weights) override;
};

#endif // ACTIVATION_H
