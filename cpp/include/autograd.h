#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include "tensor.h"

class Autograd {
public:
    static void zero_grad(std::vector<Tensor>& tensors);
};

#endif // AUTOGRAD_H
