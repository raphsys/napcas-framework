#include "autograd.h"

void Autograd::zero_grad(std::vector<Tensor>& tensors) {
    for (Tensor& tensor : tensors) {
        tensor.zero_grad();
    }
}
