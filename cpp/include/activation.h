#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "module.h"
#include "tensor.h"
#include <string>

/// @brief ReLU activation function module.
class ReLU : public Module {
public:
    ReLU() = default;
    /// @brief Applies ReLU activation: output = max(0, input).
    void forward(Tensor& input, Tensor& output) override;
    /// @brief Computes backward pass for ReLU.
    void backward(Tensor& grad_output, Tensor& grad_input) override;
    /// @brief Updates parameters (no-op for ReLU).
    void update(float lr) override;
    /// @brief Gets weights (throws error for ReLU).
    Tensor& get_weights() override;
    /// @brief Gets gradient of weights (throws error for ReLU).
    Tensor& get_grad_weights() override;
    /// @brief Sets weights (throws error for ReLU).
    void set_weights(const Tensor& weights) override;
    /// @brief No state to save for ReLU.
    void save(const std::string& path) override;
    /// @brief No state to load for ReLU.
    void load(const std::string& path) override;
};

/// @brief Sigmoid activation function module.
class Sigmoid : public Module {
public:
    Sigmoid() = default;
    /// @brief Applies Sigmoid activation: output = 1 / (1 + exp(-input)).
    void forward(Tensor& input, Tensor& output) override;
    /// @brief Computes backward pass for Sigmoid.
    void backward(Tensor& grad_output, Tensor& grad_input) override;
    /// @brief Updates parameters (no-op for Sigmoid).
    void update(float lr) override;
    /// @brief Gets weights (throws error for Sigmoid).
    Tensor& get_weights() override;
    /// @brief Gets gradient of weights (throws error for Sigmoid).
    Tensor& get_grad_weights() override;
    /// @brief Sets weights (throws error for Sigmoid).
    void set_weights(const Tensor& weights) override;
    /// @brief No state to save for Sigmoid.
    void save(const std::string& path) override;
    /// @brief No state to load for Sigmoid.
    void load(const std::string& path) override;
};

/// @brief Tanh activation function module.
class Tanh : public Module {
public:
    Tanh() = default;
    /// @brief Applies Tanh activation: output = tanh(input).
    void forward(Tensor& input, Tensor& output) override;
    /// @brief Computes backward pass for Tanh.
    void backward(Tensor& grad_output, Tensor& grad_input) override;
    /// @brief Updates parameters (no-op for Tanh).
    void update(float lr) override;
    /// @brief Gets weights (throws error for Tanh).
    Tensor& get_weights() override;
    /// @brief Gets gradient of weights (throws error for Tanh).
    Tensor& get_grad_weights() override;
    /// @brief Sets weights (throws error for Tanh).
    void set_weights(const Tensor& weights) override;
    /// @brief No state to save for Tanh.
    void save(const std::string& path) override;
    /// @brief No state to load for Tanh.
    void load(const std::string& path) override;
};

#endif // ACTIVATION_H

