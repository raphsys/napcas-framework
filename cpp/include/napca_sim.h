#ifndef NAPCA_SIM_H
#define NAPCA_SIM_H

#include "module.h"
#include "tensor.h"
#include <vector>
#include <unordered_map>

class NAPCA_Sim : public Module {
public:
    NAPCA_Sim(int in_features, int out_features, float alpha=0.6f, float threshold=0.5f);
    void forward(Tensor& input, Tensor& output) override;
    void backward(Tensor& grad_output, Tensor& grad_input) override;
    void update(float lr) override;
    Tensor& get_weights() override;
    Tensor& get_grad_weights() override;
    void set_weights(const Tensor& weights) override;
    
    // Nouvelles fonctionnalités spécifiques à NAP-CA Sim
    float compute_path_similarity(const std::vector<int>& path1, const std::vector<int>& path2);
    void update_weights_conditionally(const std::vector<std::pair<int,int>>& similar_pairs, float eta);
    void prune_connections(float threshold=0.01f);

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
};

#endif // NAPCA_SIM_H
