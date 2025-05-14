// File: cpp/include/lstm.h

#ifndef LSTM_H
#define LSTM_H

#include "module.h"
#include "tensor.h"
#include "linear.h"
#include "activation.h"
#include <vector>
#include <memory>

/// @brief Long Short-Term Memory (LSTM) récurrent.
class LSTM : public Module {
public:
    LSTM(int input_size, int hidden_size, int num_layers);

    /// Passe avant sur toute la séquence.
    /// input.shape  = [T, B, D_in]
    /// output.shape = [T, B, D_h]
    void forward(Tensor& input, Tensor& output) override;

    /// Rétro-propagation complète (BPTT).
    void backward(Tensor& grad_output, Tensor& grad_input) override;

    /// Mise à jour SGD des linears internes.
    void update(float lr) override;

    Tensor& get_weights() override { return weights_; }
    Tensor& get_grad_weights() override { return grad_weights_; }
    void set_weights(const Tensor& w) override { weights_ = w; }

    void save(const std::string& path) override;
    void load(const std::string& path) override;

private:
    int input_size_, hidden_size_, num_layers_;

    // Pour chaque couche l :
    //   Linear(input→4*hidden) + Linear(hidden→4*hidden)
    std::vector<std::shared_ptr<Linear>> input_to_hidden_, hidden_to_hidden_;
    // activations pour i,f,o et g
    std::vector<std::shared_ptr<Sigmoid>> gate_act_;
    std::vector<std::shared_ptr<Tanh>>    g_act_;

    Tensor weights_, grad_weights_; // placeholders

    // Caches [timestep][layer]
    std::vector<std::vector<Tensor>> x_cache_, h_prev_cache_, c_prev_cache_;
    std::vector<std::vector<Tensor>> i_cache_, f_cache_, g_cache_, o_cache_;
    std::vector<std::vector<Tensor>> c_cache_, h_cache_;
};

#endif // LSTM_H

