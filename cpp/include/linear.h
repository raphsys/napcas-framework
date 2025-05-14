#ifndef LINEAR_H
#define LINEAR_H

#include "module.h"
#include "tensor.h"
#include <string>

/// @brief Linear (fully connected) layer module.
class Linear : public Module {
public:
    /// @brief Constructs a Linear layer.
    /// @param in_features Number of input features.
    /// @param out_features Number of output features.
    Linear(int in_features, int out_features);

    /// @brief Performs forward pass (matrix multiplication + bias).
    /// @param input Input tensor (batch_size, in_features).
    /// @param output Output tensor (batch_size, out_features).
    void forward(Tensor& input, Tensor& output) override;

    /// @brief Performs backward pass (gradient computation).
    /// @param grad_output Gradient of the output.
    /// @param grad_input Gradient of the input.
    void backward(Tensor& grad_output, Tensor& grad_input) override;

    /// @brief Updates weights and biases.
    /// @param lr Learning rate.
    void update(float lr) override;

    /// @brief Gets the weight tensor.
    Tensor& get_weights() override;

    /// @brief Gets the gradient-of-weights tensor.
    Tensor& get_grad_weights() override;

    /// @brief Sets the weight tensor.
    /// @param weights New weights tensor.
    void set_weights(const Tensor& weights) override;

    /// @brief Saves layer parameters to file.
    /// @param path File path prefix.
    void save(const std::string& path);

    /// @brief Loads layer parameters from file.
    /// @param path File path prefix.
    void load(const std::string& path);

private:
    Tensor weights_;
    Tensor bias_;
    Tensor grad_weights_;
    Tensor grad_bias_;
    float learning_rate_;
};

#endif // LINEAR_H

