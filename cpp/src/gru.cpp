// File: cpp/src/gru.cpp

#include "gru.h"
#include <Eigen/Dense>
#include <stdexcept>
#include <algorithm>

GRU::GRU(int input_size, int hidden_size, int num_layers)
    : input_size_(input_size),
      hidden_size_(hidden_size),
      num_layers_(num_layers)
{
    for (int l = 0; l < num_layers_; ++l) {
        int in_feats = (l == 0 ? input_size_ : hidden_size_);
        input_to_hidden_.push_back(
            std::make_shared<Linear>(in_feats, hidden_size_ * 3));
        hidden_to_hidden_.push_back(
            std::make_shared<Linear>(hidden_size_, hidden_size_ * 3));
        gate_activations_.push_back(std::make_shared<Sigmoid>()); // r et z
        tanh_activations_.push_back(std::make_shared<Tanh>());    // n
    }
    weights_      = Tensor({num_layers_, 1});
    grad_weights_ = Tensor({num_layers_, 1});
}

void GRU::forward(Tensor& input, Tensor& output) {
    if (input.ndim() != 3) {
        throw std::invalid_argument("Input must be 3D (seq_len, batch, input_size)");
    }
    int seq_len    = input.shape()[0];
    int batch_size = input.shape()[1];

    // Initialisation des hidden states
    hidden_states_.assign(num_layers_, Tensor({batch_size, hidden_size_}));
    for (auto& h : hidden_states_) h.fill(0.0f);

    // Préparation de la sortie
    output = Tensor({seq_len, batch_size, hidden_size_});

    for (int t = 0; t < seq_len; ++t) {
        // Extraction de input_t
        Tensor input_t({batch_size, input_size_});
        for (int b = 0; b < batch_size; ++b) {
            for (int d = 0; d < input_size_; ++d) {
                input_t.data()[b * input_size_ + d] =
                  input.data()[t * batch_size * input_size_ + b * input_size_ + d];
            }
        }

        Tensor prev_h = input_t;
        for (int l = 0; l < num_layers_; ++l) {
            // Récupération du hidden state de la couche l
            Tensor h = hidden_states_[l];

            // 1) Calcul des 3 gates concaténées
            Tensor gates({batch_size, hidden_size_ * 3});
            input_to_hidden_[l]->forward(prev_h, gates);
            hidden_to_hidden_[l]->forward(h,      gates);

            // 2) Séparation en r, z, n
            Tensor r({batch_size, hidden_size_});
            Tensor z({batch_size, hidden_size_});
            Tensor n({batch_size, hidden_size_});
            for (int b = 0; b < batch_size; ++b) {
                for (int j = 0; j < hidden_size_; ++j) {
                    int base = b * hidden_size_ * 3 + j;
                    r.data()[b*hidden_size_ + j] = gates.data()[base];
                    z.data()[b*hidden_size_ + j] = gates.data()[base + hidden_size_];
                    n.data()[b*hidden_size_ + j] = gates.data()[base + 2*hidden_size_];
                }
            }

            // 3) Activations non-linéaires
            gate_activations_[l]->forward(r, r);
            gate_activations_[l]->forward(z, z);
            tanh_activations_[l]->forward(n, n);

            // 4) Mise à jour du hidden state
            Tensor h_new({batch_size, hidden_size_});
            for (int idx = 0; idx < batch_size * hidden_size_; ++idx) {
                float h_prev = h.data()[idx];
                float reset  = r.data()[idx];
                float cand   = n.data()[idx];
                float update = z.data()[idx];
                // h_new = z * h_prev + (1 - z) * cand
                h_new.data()[idx] = update * h_prev + (1.0f - update) * cand;
            }

            hidden_states_[l] = h_new;
            prev_h = h_new;
        }

        // Ecriture dans la sortie
        for (int b = 0; b < batch_size; ++b) {
            for (int j = 0; j < hidden_size_; ++j) {
                output.data()[t * batch_size * hidden_size_ + b * hidden_size_ + j] =
                  prev_h.data()[b * hidden_size_ + j];
            }
        }
    }
}

void GRU::backward(Tensor& grad_output, Tensor& grad_input) {
    if (grad_output.ndim() != 3) {
        throw std::invalid_argument("grad_output must be 3D");
    }
    int seq_len    = grad_output.shape()[0];
    int batch_size = grad_output.shape()[1];

    // Préparer grad_input
    grad_input = Tensor({seq_len, batch_size, input_size_});
    grad_input.fill(0.0f);

    // Gradients des hidden states
    std::vector<Tensor> grad_hidden(num_layers_, Tensor({batch_size, hidden_size_}));
    for (auto& gh : grad_hidden) gh.fill(0.0f);

    // BPTT
    for (int t = seq_len - 1; t >= 0; --t) {
        // 1) Extraction de grad_output_t
        Tensor grad_out_t({batch_size, hidden_size_});
        for (int b = 0; b < batch_size; ++b) {
            for (int j = 0; j < hidden_size_; ++j) {
                grad_out_t.data()[b*hidden_size_+j] =
                  grad_output.data()[t*batch_size*hidden_size_ + b*hidden_size_ + j];
            }
        }

        Tensor grad_next_h = grad_out_t;
        for (int l = num_layers_ - 1; l >= 0; --l) {
            // 2) Somme élément par élément au lieu de operator+
            Tensor grad_h({batch_size, hidden_size_});
            for (int idx = 0; idx < batch_size * hidden_size_; ++idx) {
                grad_h.data()[idx] = grad_hidden[l].data()[idx] + grad_next_h.data()[idx];
            }

            // 3) Recalcul simplifié des gates pour ce timestep/layer
            Tensor gates({batch_size, hidden_size_ * 3});
            Tensor prev_h = (l == 0
                ? Tensor(std::vector<int>{batch_size, input_size_})
                : hidden_states_[l-1]);
            if (l == 0) prev_h.fill(0.0f);
            input_to_hidden_[l]->forward(prev_h, gates);
            hidden_to_hidden_[l]->forward(hidden_states_[l], gates);

            // 4) Extraire r, z, n comme en forward
            Tensor r({batch_size, hidden_size_}), z({batch_size, hidden_size_}), n({batch_size, hidden_size_});
            for (int b = 0; b < batch_size; ++b) {
                for (int j = 0; j < hidden_size_; ++j) {
                    int base = b*hidden_size_*3 + j;
                    r.data()[b*hidden_size_+j] = gates.data()[base];
                    z.data()[b*hidden_size_+j] = gates.data()[base+hidden_size_];
                    n.data()[b*hidden_size_+j] = gates.data()[base+2*hidden_size_];
                }
            }
            gate_activations_[l]->forward(r, r);
            gate_activations_[l]->forward(z, z);
            tanh_activations_[l]->forward(n, n);

            // 5) Calcul des gradients de z et n
            Tensor grad_z({batch_size, hidden_size_}), grad_n({batch_size, hidden_size_});
            for (int idx = 0; idx < batch_size * hidden_size_; ++idx) {
                float cur_h   = hidden_states_[l].data()[idx];
                float prev_hv = (l==0 ? 0.0f : hidden_states_[l-1].data()[idx]);
                grad_z.data()[idx] = grad_h.data()[idx] * (cur_h - n.data()[idx]);
                grad_n.data()[idx] = grad_h.data()[idx] * (1.0f - z.data()[idx]);
            }
            gate_activations_[l]->backward(grad_z, grad_z);
            tanh_activations_[l]->backward(grad_n, grad_n);

            // 6) Construire grad_gates (r ignoré ici)
            Tensor grad_gates({batch_size, hidden_size_ * 3});
            for (int b = 0; b < batch_size; ++b) {
                for (int j = 0; j < hidden_size_; ++j) {
                    int base = b*hidden_size_*3 + j;
                    grad_gates.data()[base]                     = 0.0f;
                    grad_gates.data()[base + hidden_size_]     = grad_z.data()[b*hidden_size_+j];
                    grad_gates.data()[base + 2*hidden_size_]   = grad_n.data()[b*hidden_size_+j];
                }
            }

            // 7) Backprop à travers les Linears
            Tensor dhp({batch_size, hidden_size_});
            Tensor dx ({batch_size, l==0 ? input_size_ : hidden_size_});
            hidden_to_hidden_[l]->backward(grad_gates, dhp);
            input_to_hidden_[l]->backward(grad_gates, dx);

            // 8) Propager vers la couche/couche-temps précédent(e)
            grad_hidden[l] = dhp;
            grad_next_h    = dx;
        }

        // 9) Écriture dans grad_input[t]
        for (int b = 0; b < batch_size; ++b) {
            for (int d = 0; d < input_size_; ++d) {
                grad_input.data()[t*batch_size*input_size_ + b*input_size_ + d] =
                  grad_next_h.data()[b*input_size_ + d];
            }
        }
    }
}

void GRU::update(float lr) {
    for (size_t i = 0; i < input_to_hidden_.size(); ++i) {
        input_to_hidden_[i]->update(lr);
        hidden_to_hidden_[i]->update(lr);
    }
}

void GRU::save(const std::string& path) {
    for (size_t i = 0; i < input_to_hidden_.size(); ++i) {
        input_to_hidden_[i]->save(path + "_input_to_hidden_" + std::to_string(i));
        hidden_to_hidden_[i]->save(path + "_hidden_to_hidden_" + std::to_string(i));
    }
}

void GRU::load(const std::string& path) {
    for (size_t i = 0; i < input_to_hidden_.size(); ++i) {
        input_to_hidden_[i]->load(path + "_input_to_hidden_" + std::to_string(i));
        hidden_to_hidden_[i]->load(path + "_hidden_to_hidden_" + std::to_string(i));
    }
}

