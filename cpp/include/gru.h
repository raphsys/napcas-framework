// File: cpp/include/gru.h

#ifndef GRU_H
#define GRU_H

#include "module.h"
#include "tensor.h"
#include "linear.h"
#include "activation.h"
#include <vector>
#include <memory>

/// @brief Gated Recurrent Unit (GRU) récurrent.
class GRU : public Module {
public:
    GRU(int input_size, int hidden_size, int num_layers);

    /// Passe avant sur toute la séquence.
    /// input.shape  = [seq_len, batch_size, input_size]
    /// output.shape = [seq_len, batch_size, hidden_size]
    void forward(Tensor& input, Tensor& output) override;

    /// Rétro-propagation (BPTT).
    void backward(Tensor& grad_output, Tensor& grad_input) override;

    /// Mise à jour SGD des linears internes.
    void update(float lr) override;

    Tensor& get_weights() override { return weights_; }
    Tensor& get_grad_weights() override { return grad_weights_; }
    void set_weights(const Tensor& weights) override { weights_ = weights; }

    void save(const std::string& path) override;
    void load(const std::string& path) override;

private:
    int input_size_;
    int hidden_size_;
    int num_layers_;

    std::vector<std::shared_ptr<Linear>>  input_to_hidden_;
    std::vector<std::shared_ptr<Linear>>  hidden_to_hidden_;
    std::vector<std::shared_ptr<Sigmoid>> gate_activations_;
    std::vector<std::shared_ptr<Tanh>>    tanh_activations_;

    Tensor weights_, grad_weights_;
    std::vector<Tensor> hidden_states_;  // h_t pour chaque couche
};

#endif // GRU_H

