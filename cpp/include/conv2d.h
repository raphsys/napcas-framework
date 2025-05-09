#ifndef CONV2D_H
#define CONV2D_H

#include "module.h"
#include "tensor.h"

class Conv2d : public Module {
public:
    Conv2d(int in_channels, int out_channels, int kernel_size);
    void forward(Tensor& input, Tensor& output) override;
    void backward(Tensor& grad_output, Tensor& grad_input) override;
    void update(float lr) override;
    Tensor& get_weights() override;
    Tensor& get_grad_weights() override;
    void set_weights(const Tensor& weights) override;

private:
    int kernel_size_;
    Tensor weights_;
    Tensor bias_;
    Tensor grad_weights_;
    Tensor grad_bias_;
    float learning_rate_;
};

#endif // CONV2D_H

