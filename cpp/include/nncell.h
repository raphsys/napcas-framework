#ifndef NNCELL_H
#define NNCELL_H

#include "module.h"
#include "tensor.h"

class NNCell : public Module {
public:
    NNCell(int in_features, int out_features);
    void forward(Tensor& input, Tensor& output) override;
    void backward(Tensor& grad_output, Tensor& grad_input) override;
    void update(float lr) override;

private:
    Tensor weights_;
    Tensor bias_;
    Tensor grad_weights_;
    Tensor grad_bias_;
    Tensor input_;  // Pour stocker l'input durant le forward pass
    float learning_rate_;
};

#endif // NNCELL_H
