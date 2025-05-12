#ifndef MLP_H
#define MLP_H

#include "module.h"
#include "linear.h"
#include "activation.h"
#include <vector>
#include <memory>

/// @brief Multi-Layer Perceptron (MLP) model.
class MLP : public Module {
public:
    /// @brief Constructs an MLP.
    /// @param layers List of layer sizes (including input and output).
    /// @param activation Activation function to use between layers.
    MLP(const std::vector<int>& layers, const std::string& activation = "relu");
    /// @brief Performs forward pass through the MLP.
    /// @param input Input tensor.
    /// @param output Output tensor.
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
    /// @brief Saves MLP state to file.
    /// @param path File path.
    void save(const std::string& path) override;
    /// @brief Loads MLP state from file.
    /// @param path File path.
    void load(const std::string& path) override;

private:
    std::vector<std::shared_ptr<Linear>> linear_layers_;
    std::vector<std::shared_ptr<Module>> activation_layers_;
    Tensor weights_;
    Tensor grad_weights_;
};

#endif // MLP_H
