#include "autograd.h"

Tensor Autograd::grad(Tensor& tensor) {
    return tensor;
}

void Autograd::zero_grad(std::vector<Tensor>& tensors) {
    for (auto& t : tensors) {
        t.zero_grad();
    }
}
