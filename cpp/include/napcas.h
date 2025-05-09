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

private:
    Tensor weights_;
    Tensor connections_;
    Tensor threshold_;
    Tensor alpha_;
    Tensor memory_paths_;
    Tensor grad_weights_;
    Tensor grad_connections_;
    Tensor grad_threshold_;
    Tensor grad_alpha_;
    Tensor grad_memory_paths_;
    float learning_rate_;
};

#endif // NAPCAS_H
