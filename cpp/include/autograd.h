#ifndef AUTOGRADE_H
#define AUTOGRADE_H

#include "tensor.h"

class Autograd {
public:
    static Tensor grad(Tensor& tensor);
    static void zero_grad(std::vector<Tensor>& tensors);
};

#endif // AUTOGRADE_H
