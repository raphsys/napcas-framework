#ifndef LINEAR_H
#define LINEAR_H

#include "module.h"
#include "tensor.h"

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
    /// @brief Gets weights tensor.
    /// @return Reference to weights tensor.
    Tensor& get_weights() override;
    /// @brief Gets gradient of weights tensor.
    /// @return Reference to gradient of weights tensor.
    Tensor& get_grad_weights() override;
    /// @brief Sets weights tensor.
    /// @param weights New weights tensor.
    void set_weights(const Tensor& weights) override;

private:
    Tensor weights_;
    Tensor bias_;
    Tensor grad_weights_;
    Tensor grad_bias_;
    float learning_rate_;
#ifdef USE_CUDA
    void forward_cuda(Tensor& input, Tensor& output);
    void backward_cuda(Tensor& grad_output, Tensor& grad_input);
#endif
};

#endif // LINEAR_H
