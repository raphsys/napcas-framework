#include "autograd.h"
#include <algorithm>

void Autograd::zero_grad(std::vector<Tensor>& tensors) {
    for (Tensor& tensor : tensors) {
        tensor.zero_grad();
    }
}

void Autograd::backward(Tensor& output, std::vector<Tensor>& inputs) {
    // Simplified automatic differentiation
    // In a real implementation, this would build a computation graph
    Tensor grad_output({output.shape()}, std::vector<float>(output.size(), 1.0f));
    for (Tensor& input : inputs) {
        Tensor grad_input(input.shape());
        // Placeholder for gradient computation
        // In practice, this would traverse the computation graph
        input.data() = grad_input.data();
    }
}
