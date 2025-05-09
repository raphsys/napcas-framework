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
};

class Sigmoid : public Module {
public:
    Sigmoid();
    void forward(Tensor& input, Tensor& output) override;
    void backward(Tensor& grad_output, Tensor& grad_input) override;
    void update(float lr) override;

private:
    Tensor output_;
};

class Tanh : public Module {
public:
    Tanh();
    void forward(Tensor& input, Tensor& output) override;
    void backward(Tensor& grad_output, Tensor& grad_input) override;
    void update(float lr) override;

private:
    Tensor output_;
};

#endif // ACTIVATION_H
