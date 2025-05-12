#include "lstm.h"
#include <Eigen/Dense>
#include <stdexcept>
#include <fstream>

LSTM::LSTM(int input_size, int hidden_size, int num_layers)
    : input_size_(input_size), hidden_size_(hidden_size), num_layers_(num_layers) {
    for (int i = 0; i < num_layers; ++i) {
        input_to_hidden_.push_back(std::make_shared<Linear>(i == 0 ? input_size : hidden_size, hidden_size * 4));
        hidden_to_hidden_.push_back(std::make_shared<Linear>(hidden_size, hidden_size * 4));
        gate_activations_.push_back(std::make_shared<Sigmoid>());
        tanh_activations_.push_back(std::make_shared<Tanh>());
    }
    weights_ = Tensor({num_layers, 1});
    grad_weights_ = Tensor({num_layers, 1});
}

void LSTM::forward(Tensor& input, Tensor& output) {
    if (input.ndim() != 3) {
        throw std::invalid_argument("Input must be 3D (seq_len, batch_size, input_size)");
    }
    int seq_len = input.shape()[0];
    int batch_size = input.shape()[1];
    hidden_states_.clear();
    cell_states_.clear();

    for (int l = 0; l < num_layers_; ++l) {
        Tensor h({batch_size, hidden_size_});
        Tensor c({batch_size, hidden_size_});
        h.fill(0.0f);
        c.fill(0.0f);
        hidden_states_.push_back(h);
        cell_states_.push_back(c);
    }

    for (int t = 0; t < seq_len; ++t) {
        Tensor input_t({batch_size, input_size_});
        for (int b = 0; b < batch_size; ++b) {
            for (int i = 0; i < input_size_; ++i) {
                input_t[b * input_size_ + i] = input[t * batch_size * input_size_ + b * input_size_ + i];
            }
        }

        Tensor next_h = input_t;
        for (int l = 0; l < num_layers_; ++l) {
            Tensor h = hidden_states_[l];
            Tensor c = cell_states_[l];
            Tensor gates({batch_size, hidden_size_ * 4});
            input_to_hidden_[l]->forward(l == 0 ? input_t : next_h, gates);
            hidden_to_hidden_[l]->forward(h, gates);

            Tensor i_gate({batch_size, hidden_size_});
            Tensor f_gate({batch_size, hidden_size_});
            Tensor c_gate({batch_size, hidden_size_});
            Tensor o_gate({batch_size, hidden_size_});
            for (int b = 0; b < batch_size; ++b) {
                for (int j = 0; j < hidden_size_; ++j) {
                    i_gate[b * hidden_size_ + j] = gates[b * hidden_size_ * 4 + j];
                    f_gate[b * hidden_size_ + j] = gates[b * hidden_size_ * 4 + hidden_size_ + j];
                    c_gate[b * hidden_size_ + j] = gates[b * hidden_size_ * 4 + 2 * hidden_size_ + j];
                    o_gate[b * hidden_size_ + j] = gates[b * hidden_size_ * 4 + 3 * hidden_size_ + j];
                }
            }

            gate_activations_[l]->forward(i_gate, i_gate);
            gate_activations_[l]->forward(f_gate, f_gate);
            tanh_activations_[l]->forward(c_gate, c_gate);
            gate_activations_[l]->forward(o_gate, o_gate);

            for (int b = 0; b < batch_size; ++b) {
                for (int j = 0; j < hidden_size_; ++j) {
                    c[b * hidden_size_ + j] = f_gate[b * hidden_size_ + j] * c[b * hidden_size_ + j] +
                                              i_gate[b * hidden_size_ + j] * c_gate[b * hidden_size_ + j];
                    h[b * hidden_size_ + j] = o_gate[b * hidden_size_ + j] * std::tanh(c[b * hidden_size_ + j]);
                }
            }

            hidden_states_[l] = h;
            cell_states_[l] = c;
            next_h = h;
        }

        for (int b = 0; b < batch_size; ++b) {
            for (int i = 0; i < hidden_size_; ++i) {
                output[t * batch_size * hidden_size_ + b * hidden_size_ + i] = hidden_states_[num_layers_ - 1][b * hidden_size_ + i];
            }
        }
    }
}

void LSTM::backward(Tensor& grad_output, Tensor& grad_input) {
    int seq_len = grad_output.shape()[0];
    int batch_size = grad_output.shape()[1];
    grad_input.fill(0.0f);

    std::vector<Tensor> grad_hidden(num_layers_, Tensor({batch_size, hidden_size_}));
    std::vector<Tensor> grad_cell(num_layers_, Tensor({batch_size, hidden_size_}));
    for (int l = 0; l < num_layers_; ++l) {
        grad_hidden[l].fill(0.0f);
        grad_cell[l].fill(0.0f);
    }

    for (int t = seq_len - 1; t >= 0; --t) {
        Tensor grad_output_t({batch_size, hidden_size_});
        for (int b = 0; b < batch_size; ++b) {
            for (int i = 0; i < hidden_size_; ++i) {
                grad_output_t[b * hidden_size_ + i] = grad_output[t * batch_size * hidden_size_ + b * hidden_size_ + i];
            }
        }

        Tensor grad_next_h = grad_output_t;
        for (int l = num_layers_ - 1; l >= 0; --l) {
            Tensor h = hidden_states_[l];
            Tensor c = cell_states_[l];
            Tensor grad_h = grad_hidden[l] + grad_next_h;
            Tensor grad_c = grad_cell[l];

            Tensor gates({batch_size, hidden_size_ * 4});
            input_to_hidden_[l]->forward(l == 0 ? Tensor({batch_size, input_size_}) : hidden_states_[l-1], gates);
            hidden_to_hidden_[l]->forward(h, gates);

            Tensor i_gate({batch_size, hidden_size_});
            Tensor f_gate({batch_size, hidden_size_});
            Tensor c_gate({batch_size, hidden_size_});
            Tensor o_gate({batch_size, hidden_size_});
            for (int b = 0; b < batch_size; ++b) {
                for (int j = 0; j < hidden_size_; ++j) {
                    i_gate[b * hidden_size_ + j] = gates[b * hidden_size_ * 4 + j];
                    f_gate[b * hidden_size_ + j] = gates[b * hidden_size_ * 4 + hidden_size_ + j];
                    c_gate[b * hidden_size_ + j] = gates[b * hidden_size_ * 4 + 2 * hidden_size_ + j];
                    o_gate[b * hidden_size_ + j] = gates[b * hidden_size_ * 4 + 3 * hidden_size_ + j];
                }
            }

            Tensor grad_o({batch_size, hidden_size_});
            for (int b = 0; b < batch_size; ++b) {
                for (int j = 0; j < hidden_size_; ++j) {
                    grad_o[b * hidden_size_ + j] = grad_h[b * hidden_size_ + j] * std::tanh(c[b * hidden_size_ + j]);
                    grad_c[b * hidden_size_ + j] += grad_h[b * hidden_size_ + j] * o_gate[b * hidden_size_ + j] *
                                                    (1.0f - std::tanh(c[b * hidden_size_ + j]) * std::tanh(c[b * hidden_size_ + j]));
                }
            }

            Tensor grad_i({batch_size, hidden_size_});
            Tensor grad_f({batch_size, hidden_size_});
            Tensor grad_c_tilde({batch_size, hidden_size_});
            for (int b = 0; b < batch_size; ++b) {
                for (int j = 0; j < hidden_size_; ++j) {
                    grad_i[b * hidden_size_ + j] = grad_c[b * hidden_size_ + j] * c_gate[b * hidden_size_ + j];
                    grad_f[b * hidden_size_ + j] = grad_c[b * hidden_size_ + j] * cell_states_[l][b * hidden_size_ + j];
                    grad_c_tilde[b * hidden_size_ + j] = grad_c[b * hidden_size_ + j] * i_gate[b * hidden_size_ + j];
                }
            }

            gate_activations_[l]->backward(grad_i, grad_i);
            gate_activations_[l]->backward(grad_f, grad_f);
            tanh_activations_[l]->backward(grad_c_tilde, grad_c_tilde);
            gate_activations_[l]->backward(grad_o, grad_o);

            Tensor grad_gates({batch_size, hidden_size_ * 4});
            for (int b = 0; b < batch_size; ++b) {
                for (int j = 0; j < hidden_size_; ++j) {
                    grad_gates[b * hidden_size_ * 4 + j] = grad_i[b * hidden_size_ + j];
                    grad_gates[b * hidden_size_ * 4 + hidden_size_ + j] = grad_f[b * hidden_size_ + j];
                    grad_gates[b * hidden_size_ * 4 + 2 * hidden_size_ + j] = grad_c_tilde[b * hidden_size_ + j];
                    grad_gates[b * hidden_size_ * 4 + 3 * hidden_size_ + j] = grad_o[b * hidden_size_ + j];
                }
            }

            hidden_to_hidden_[l]->backward(grad_gates, grad_hidden[l]);
            input_to_hidden_[l]->backward(grad_gates, grad_next_h);

            grad_cell[l] = grad_c * f_gate;
        }

        if (t > 0) {
            for (int b = 0; b < batch_size; ++b) {
                for (int i = 0; i < input_size_; ++i) {
                    grad_input[t * batch_size * input_size_ + b * input_size_ + i] = grad_next_h[b * input_size_ + i];
                }
            }
        }
    }
}

void LSTM::update(float lr) {
    for (size_t i = 0; i < input_to_hidden_.size(); ++i) {
        input_to_hidden_[i]->update(lr);
        hidden_to_hidden_[i]->update(lr);
    }
}

void LSTM::set_weights(const Tensor& weights) {
    weights_ = weights;
}

void LSTM::save(const std::string& path) {
    for (size_t i = 0; i < input_to_hidden_.size(); ++i) {
        input_to_hidden_[i]->save(path + "_input_to_hidden_" + std::to_string(i));
        hidden_to_hidden_[i]->save(path + "_hidden_to_hidden_" + std::to_string(i));
    }
}

void LSTM::load(const std::string& path) {
    for (size_t i = 0; i < input_to_hidden_.size(); ++i) {
        input_to_hidden_[i]->load(path + "_input_to_hidden_" + std::to_string(i));
        hidden_to_hidden_[i]->load(path + "_hidden_to_hidden_" + std::to_string(i));
    }
}
