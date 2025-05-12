#ifndef NAPCA_SIM_H
#define NAPCA_SIM_H

#include "module.h"
#include "tensor.h"
#include <vector>
#include <unordered_map>

/// @brief NAPCA_Sim module with adaptive connections and path similarity.
class NAPCA_Sim : public Module {
public:
    /// @brief Constructs a NAPCA_Sim layer.
    /// @param in_features Number of input features.
    /// @param out_features Number of output features.
    /// @param alpha Scaling factor for input transformation.
    /// @param threshold Activation threshold.
    NAPCA_Sim(int in_features, int out_features, float alpha = 0.6f, float threshold = 0.5f);
    /// @brief Performs forward pass with adaptive connections.
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
    /// @brief Computes similarity between two activation paths.
    /// @param path1 First path.
    /// @param path2 Second path.
    /// @return Similarity score (Jaccard index).
    float compute_path_similarity(const std::vector<int>& path1, const std::vector<int>& path2);
    /// @brief Updates weights based on similar paths.
    /// @param similar_pairs Pairs of similar paths.
    /// @param eta Adjustment factor.
    void update_weights_conditionally(const std::vector<std::pair<int,int>>& similar_pairs, float eta);
    /// @brief Prunes connections below a threshold.
    /// @param threshold Pruning threshold.
    void prune_connections(float threshold = 0.01f);
    /// @brief Saves module state to file.
    /// @param path File path.
    void save(const std::string& path) override;
    /// @brief Loads module state from file.
    /// @param path File path.
    void load(const std::string& path) override;

private:
    Tensor weights_;
    Tensor bias_;
    Tensor grad_weights_;
    Tensor grad_bias_;
    float learning_rate_;
    float alpha_;
    float threshold_;
    std::vector<std::vector<int>> memory_paths_;
    std::unordered_map<int, bool> active_connections_;
#ifdef USE_CUDA
    void forward_cuda(Tensor& input, Tensor& output);
    void backward_cuda(Tensor& grad_output, Tensor& grad_input);
#endif
};

#endif // NAPCA_SIM_H
