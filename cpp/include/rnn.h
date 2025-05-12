#ifndef RNN_H
#define RNN_H

#include "module.h"
#include "linear.h"
#include "activation.h"
#include <vector>
#include <memory>

/// @brief Recurrent Neural Network (RNN) module.
class RNN : public Module {
public:
    /// @brief Constructs an RNN.
    /// @param input_size Size of input features.
    /// @param hidden_size Size of hidden state.
    /// @param num_layers Number of RNN layers.
    RNN(int input_size, int hidden_size, int num_layers = 1);
    /// @brief Performs forward pass through the RNN.
    /// @param input Input tensor (sequence_length, batch_size, input_size).
    /// @param output Output tensor (sequence_length, batch_size, hidden_size).
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
    /// @brief Saves RNN state to file.
    /// @param path File path.
    void save(const std::string& path) override;
    /// @brief Loads RNN state from file.
    /// @param path File path.
    void load(const std::string& path) override;

private:
    int input_size_;
    int hidden_size_;
    int num_layers_;
    std::vector<std::shared_ptr<Linear>> input_to_hidden_;
    std::vector<std::shared_ptr<Linear>> hidden_to_hidden_;
    std::vector<std::shared_ptr<Tanh>> activations_;
    Tensor weights_;
    Tensor grad_weights_;
};

#endif // RNN_H
