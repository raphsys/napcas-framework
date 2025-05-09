#ifndef MODULE_H
#define MODULE_H

#include "tensor.h"

class Module {
public:
    Module() = default;
    virtual ~Module() = default;
    
    virtual void forward(Tensor& input, Tensor& output) = 0;
    virtual void backward(Tensor& grad_output, Tensor& grad_input) = 0;
    virtual void update(float lr) = 0;
};

#endif // MODULE_H
