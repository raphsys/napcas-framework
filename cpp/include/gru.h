#ifndef GRU_H
#define GRU_H

#include "module.h"
#include "tensor.h"
#include "linear.h"
#include "activation.h"
#include <vector>
#include <memory>

class GRU : public Module {
public:
    GRU(int input_size, int hidden_size, int num_layers);
    void forward(Tensor& input, Tensor& output) override;
    void backward(Tensor& grad_output, Tensor& grad_input) override;
    void update(float lr) override;
    Tensor& get_weights() override { return weights_; }
    Tensor& get_grad_weights() override { return grad_weights_; }
    void set_weights(const Tensor& weights) override;
    void save(const std::string& path) override;
    void load(const std::string& path) override;

private:
    int input_size_;
    int hidden_size_;
    int num_layers_;
    std::vector<std::shared_ptr<Linear>> input_to_hidden_;
    std::vector<std::shared_ptr<Linear>> hidden_to_hidden_;
    std::vector<std::shared_ptr<Sigmoid>> gate_activations_;
    std::vector<std::shared_ptr<Tanh>> tanh_activations_;
    Tensor weights_;
    Tensor grad_weights_;
    std::vector<Tensor> hidden_states_;
};

#endif
