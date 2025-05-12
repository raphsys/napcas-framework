#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "module.h"
#include "tensor.h"

/// @brief ReLU activation function module.
class ReLU : public Module {
public:
    ReLU() = default;
    /// @brief Applies ReLU activation: output = max(0, input).
    /// @param input Input tensor.
    /// @param output Output tensor.
    void forward(Tensor& input, Tensor& output) override;
    /// @brief Computes backward pass for ReLU.
    /// @param grad_output Gradient of the output.
    /// @param grad_input Gradient of the input.
    void backward(Tensor& grad_output, Tensor& grad_input) override;
    /// @brief Updates parameters (no-op for ReLU).
    /// @param lr Learning rate.
    void update(float lr) override;
    /// @brief Gets weights (throws error for ReLU).
    /// @return Tensor reference.
    Tensor& get_weights() override;
    /// @brief Gets gradient of weights (throws error for ReLU).
    /// @return Tensor reference.
    Tensor& get_grad_weights() override;
    /// @brief Sets weights (throws error for ReLU).
    /// @param weights Tensor of weights.
    void set_weights(const Tensor& weights) override;
};

/// @brief Sigmoid activation function module.
class Sigmoid : public Module {
public:
    Sigmoid() = default;
    /// @brief Applies Sigmoid activation: output = 1 / (1 + exp(-input)).
    /// @param input Input tensor.
    /// @param output Output tensor.
    void forward(Tensor& input, Tensor& output) override;
    /// @brief Computes backward pass for Sigmoid.
    /// @param grad_output Gradient of the output.
    /// @param grad_input Gradient of the input.
    void backward(Tensor& grad_output, Tensor& grad_input) override;
    /// @brief Updates parameters (no-op for Sigmoid).
    /// @param lr Learning rate.
    void update(float lr) override;
    /// @brief Gets weights (throws error for Sigmoid).
    /// @return Tensor reference.
    Tensor& get_weights() override;
    /// @brief Gets gradient of weights (throws error for Sigmoid).
    /// @return Tensor reference.
    Tensor& get_grad_weights() override;
    /// @brief Sets weights (throws error for Sigmoid).
    /// @param weights Tensor of weights.
    void set_weights(const Tensor& weights) override;
};

/// @brief Tanh activation function module.
class Tanh : public Module {
public:
    Tanh() = default;
    /// @brief Applies Tanh activation: output = tanh(input).
    /// @param input Input tensor.
    /// @param output Output tensor.
    void forward(Tensor& input, Tensor& output) override;
    /// @brief Computes backward pass for Tanh.
    /// @param grad_output Gradient of the output.
    /// @param grad_input Gradient of the input.
    void backward(Tensor& grad_output, Tensor& grad_input) override;
    /// @brief Updates parameters (no-op for Tanh).
    /// @param lr Learning rate.
    void update(float lr) override;
    /// @brief Gets weights (throws error for Tanh).
    /// @return Tensor reference.
    Tensor& get_weights() override;
    /// @brief Gets gradient of weights (throws error for Tanh).
    /// @return Tensor reference.
    Tensor& get_grad_weights() override;
    /// @brief Sets weights (throws error for Tanh).
    /// @param weights Tensor of weights.
    void set_weights(const Tensor& weights) override;
};

#endif // ACTIVATION_H
