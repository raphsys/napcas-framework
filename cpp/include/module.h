#ifndef MODULE_H
#define MODULE_H

#include "tensor.h"
#include <memory>
#include <vector>

/// @brief Base class for all neural network modules.
class Module {
public:
    /// @brief Performs forward pass.
    /// @param input Input tensor.
    /// @param output Output tensor.
    virtual void forward(Tensor& input, Tensor& output) = 0;
    /// @brief Performs backward pass (gradient computation).
    /// @param grad_output Gradient of the output.
    /// @param grad_input Gradient of the input.
    virtual void backward(Tensor& grad_output, Tensor& grad_input) = 0;
    /// @brief Updates parameters.
    /// @param lr Learning rate.
    virtual void update(float lr) = 0;
    /// @brief Gets weights tensor.
    /// @return Reference to weights tensor.
    virtual Tensor& get_weights() = 0;
    /// @brief Gets gradient of weights tensor.
    /// @return Reference to gradient of weights tensor.
    virtual Tensor& get_grad_weights() = 0;
    /// @brief Sets weights tensor.
    /// @param weights New weights tensor.
    virtual void set_weights(const Tensor& weights) = 0;
    /// @brief Saves module state to file.
    /// @param path File path.
    virtual void save(const std::string& path) = 0;
    /// @brief Loads module state from file.
    /// @param path File path.
    virtual void load(const std::string& path) = 0;
    virtual ~Module() = default;

protected:
    Tensor input_;
};

#endif // MODULE_H
