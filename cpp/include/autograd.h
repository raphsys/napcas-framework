#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include "tensor.h"
#include <vector>
#include <memory>

/// @brief Autograd engine for automatic differentiation.
class Autograd {
public:
    /// @brief Zeros gradients for a list of tensors.
    /// @param tensors Vector of tensors to zero gradients for.
    static void zero_grad(std::vector<Tensor>& tensors);

    /// @brief Computes gradients using automatic differentiation.
    /// @param output Output tensor to compute gradients for.
    /// @param inputs Input tensors to compute gradients with respect to.
    static void backward(Tensor& output, std::vector<Tensor>& inputs);
};

#endif // AUTOGRAD_H
