#ifndef NAPCAS_H
#define NAPCAS_H

#include "module.h"
#include "tensor.h"

class NAPCAS : public Module {
public:
    NAPCAS(int in_features, int out_features);
    void forward(Tensor& input, Tensor& output) override;
    void backward(Tensor& grad_output, Tensor& grad_input) override;
    void update(float lr) override;
    Tensor& get_weights() override;
    Tensor& get_grad_weights() override;
    void set_weights(const Tensor& weights) override;

private:
    Tensor weights_;
    Tensor bias_;
    Tensor grad_weights_;
    Tensor grad_bias_;
    float learning_rate_;
};

#endif // NAPCAS_H
