#ifndef MODULE_H
#define MODULE_H

#include "tensor.h"
#include <memory>
#include <vector>

class Module {
public:
    virtual void forward(Tensor& input, Tensor& output) = 0;
    virtual void backward(Tensor& grad_output, Tensor& grad_input) = 0;
    virtual void update(float lr) = 0;
    virtual Tensor& get_weights() = 0;
    virtual Tensor& get_grad_weights() = 0;
    virtual ~Module() = default;

protected:
    Tensor input_;
};

#endif // MODULE_H
