// File: cpp/src/lstm.cpp

#include "lstm.h"
#include <stdexcept>
#include <cmath>      // std::tanh
#include <algorithm>
#include <fstream>

LSTM::LSTM(int input_size, int hidden_size, int num_layers)
    : input_size_(input_size),
      hidden_size_(hidden_size),
      num_layers_(num_layers)
{
    for (int l = 0; l < num_layers_; ++l) {
        int in_feats = (l == 0 ? input_size_ : hidden_size_);
        input_to_hidden_.push_back(
            std::make_shared<Linear>(in_feats, hidden_size_ * 4));
        hidden_to_hidden_.push_back(
            std::make_shared<Linear>(hidden_size_, hidden_size_ * 4));
        gate_act_.push_back(std::make_shared<Sigmoid>());
        g_act_.push_back(std::make_shared<Tanh>());
    }
    // placeholders (pas de vrais poids globaux)
    weights_      = Tensor({num_layers_, 1});
    grad_weights_ = Tensor({num_layers_, 1});
}

void LSTM::forward(Tensor& input, Tensor& output) {
    if (input.ndim() != 3) {
        throw std::invalid_argument("LSTM input must be 3D (T, B, D_in)");
    }
    int T = input.shape()[0];
    int B = input.shape()[1];

    // préparer les caches
    x_cache_      .assign(T, std::vector<Tensor>(num_layers_));
    h_prev_cache_ .assign(T, std::vector<Tensor>(num_layers_));
    c_prev_cache_ .assign(T, std::vector<Tensor>(num_layers_));
    i_cache_      .assign(T, std::vector<Tensor>(num_layers_));
    f_cache_      .assign(T, std::vector<Tensor>(num_layers_));
    g_cache_      .assign(T, std::vector<Tensor>(num_layers_));
    o_cache_      .assign(T, std::vector<Tensor>(num_layers_));
    c_cache_      .assign(T, std::vector<Tensor>(num_layers_));
    h_cache_      .assign(T, std::vector<Tensor>(num_layers_));

    // états initiaux
    std::vector<Tensor> h_prev(num_layers_), c_prev(num_layers_);
    for (int l = 0; l < num_layers_; ++l) {
        h_prev[l] = Tensor({B, hidden_size_});
        c_prev[l] = Tensor({B, hidden_size_});
        h_prev[l].fill(0.0f);
        c_prev[l].fill(0.0f);
    }

    // préparer la sortie
    output = Tensor({T, B, hidden_size_});

    for (int t = 0; t < T; ++t) {
        // extraire x_t
        Tensor x_t({B, input_size_});
        for (int b = 0; b < B; ++b) {
            for (int d = 0; d < input_size_; ++d) {
                x_t.data()[b * input_size_ + d] =
                    input.data()[t * B * input_size_ + b * input_size_ + d];
            }
        }

        Tensor layer_in = x_t;
        for (int l = 0; l < num_layers_; ++l) {
            // caches
            x_cache_[t][l]      = layer_in;
            h_prev_cache_[t][l] = h_prev[l];
            c_prev_cache_[t][l] = c_prev[l];

            // 1) gates concat
            Tensor gates({B, hidden_size_ * 4});
            input_to_hidden_[l]->forward(layer_in, gates);
            hidden_to_hidden_[l]->forward(h_prev[l], gates);

            // 2) split gates
            Tensor i({B, hidden_size_}), f({B, hidden_size_});
            Tensor g({B, hidden_size_}), o({B, hidden_size_});
            for (int b = 0; b < B; ++b) {
                for (int j = 0; j < hidden_size_; ++j) {
                    int base = b * hidden_size_ * 4 + j;
                    i.data()[b * hidden_size_ + j] =
                        gates.data()[base];
                    f.data()[b * hidden_size_ + j] =
                        gates.data()[base + hidden_size_];
                    g.data()[b * hidden_size_ + j] =
                        gates.data()[base + 2 * hidden_size_];
                    o.data()[b * hidden_size_ + j] =
                        gates.data()[base + 3 * hidden_size_];
                }
            }

            // 3) activations
            gate_act_[l]->forward(i, i);
            gate_act_[l]->forward(f, f);
            g_act_[l]->forward(g, g);
            gate_act_[l]->forward(o, o);

            // 4) new cell & hidden
            Tensor c_new({B, hidden_size_}), h_new({B, hidden_size_});
            for (int idx = 0; idx < B * hidden_size_; ++idx) {
                float ci = i.data()[idx];
                float cf = f.data()[idx];
                float cg = g.data()[idx];
                float co = o.data()[idx];
                float cp = c_prev[l].data()[idx];
                float cv = cf * cp + ci * cg;
                c_new.data()[idx] = cv;
                h_new.data()[idx] = co * std::tanh(cv);
            }

            // cacher
            i_cache_[t][l] = i;
            f_cache_[t][l] = f;
            g_cache_[t][l] = g;
            o_cache_[t][l] = o;
            c_cache_[t][l] = c_new;
            h_cache_[t][l] = h_new;

            // préparer pour la couche suivante
            layer_in = h_new;
            h_prev[l] = h_new;
            c_prev[l] = c_new;
        }

        // écrire output[t]
        for (int b = 0; b < B; ++b) {
            for (int j = 0; j < hidden_size_; ++j) {
                output.data()[t * B * hidden_size_ + b * hidden_size_ + j] =
                    layer_in.data()[b * hidden_size_ + j];
            }
        }
    }
}

void LSTM::backward(Tensor& grad_output, Tensor& grad_input) {
    int T = grad_output.shape()[0];
    int B = grad_output.shape()[1];

    // préparer grad_input
    grad_input = Tensor({T, B, input_size_});
    grad_input.fill(0.0f);

    // next-layer gradients
    std::vector<Tensor> dh_next(num_layers_), dc_next(num_layers_);
    for (int l = 0; l < num_layers_; ++l) {
        dh_next[l] = Tensor({B, hidden_size_}); dh_next[l].fill(0.0f);
        dc_next[l] = Tensor({B, hidden_size_}); dc_next[l].fill(0.0f);
    }

    // BPTT
    for (int t = T - 1; t >= 0; --t) {
        // grad from output
        Tensor dh = Tensor({B, hidden_size_});
        for (int b = 0; b < B; ++b) {
            for (int j = 0; j < hidden_size_; ++j) {
                dh.data()[b * hidden_size_ + j] =
                    grad_output.data()[t * B * hidden_size_ + b * hidden_size_ + j]
                    + dh_next.back().data()[b * hidden_size_ + j];
            }
        }

        // layer-wise backprop
        for (int l = num_layers_ - 1; l >= 0; --l) {
            auto& x_t   = x_cache_[t][l];
            auto& hp    = h_prev_cache_[t][l];
            auto& cp    = c_prev_cache_[t][l];
            auto& i     = i_cache_[t][l];
            auto& f     = f_cache_[t][l];
            auto& g     = g_cache_[t][l];
            auto& o     = o_cache_[t][l];
            auto& c_t   = c_cache_[t][l];

            // accumulate dh_next
            for (int idx = 0; idx < B * hidden_size_; ++idx) {
                dh.data()[idx] += dh_next[l].data()[idx];
            }

            // grad o
            Tensor do_({B, hidden_size_});
            for (int idx = 0; idx < B * hidden_size_; ++idx) {
                do_.data()[idx] = dh.data()[idx] * std::tanh(c_t.data()[idx]);
            }

            // grad c
            Tensor dc = dc_next[l];
            for (int idx = 0; idx < B * hidden_size_; ++idx) {
                float dht = dh.data()[idx] * o.data()[idx];
                dc.data()[idx] += dht * (1.0f - std::tanh(c_t.data()[idx]) * std::tanh(c_t.data()[idx]));
            }

            // gate grads
            Tensor di({B, hidden_size_}), df({B, hidden_size_}),
                   dg({B, hidden_size_}), dcprev({B, hidden_size_});
            for (int idx = 0; idx < B * hidden_size_; ++idx) {
                di.data()[idx]     = dc.data()[idx] * g.data()[idx];
                df.data()[idx]     = dc.data()[idx] * cp.data()[idx];
                dg.data()[idx]     = dc.data()[idx] * i.data()[idx];
                dcprev.data()[idx] = dc.data()[idx] * f.data()[idx];
            }

            // activation backward
            for (int idx = 0; idx < B * hidden_size_; ++idx) {
                float iv = i.data()[idx], fv = f.data()[idx],
                      gv = g.data()[idx], ov = o.data()[idx];
                di.data()[idx] *= iv * (1.0f - iv);
                df.data()[idx] *= fv * (1.0f - fv);
                dg.data()[idx] *= (1.0f - gv * gv);
                do_.data()[idx] *= ov * (1.0f - ov);
            }

            // concat gate grads
            Tensor dG({B, hidden_size_ * 4});
            for (int b = 0; b < B; ++b) {
                for (int j = 0; j < hidden_size_; ++j) {
                    int base = b * hidden_size_ * 4 + j;
                    dG.data()[base]                     = di.data()[b * hidden_size_ + j];
                    dG.data()[base + hidden_size_]     = df.data()[b * hidden_size_ + j];
                    dG.data()[base + 2 * hidden_size_] = dg.data()[b * hidden_size_ + j];
                    dG.data()[base + 3 * hidden_size_] = do_.data()[b * hidden_size_ + j];
                }
            }

            // backprop linears
            Tensor dhp({B, hidden_size_}), dx({B, (l==0?input_size_:hidden_size_)});
            hidden_to_hidden_[l]->backward(dG, dhp);
            input_to_hidden_[l]->backward(dG, dx);

            // prepare next
            dh_next[l] = dhp;
            dc_next[l] = dcprev;
            dh = dx;  // feed into lower layer or into input grad
        }

        // write into grad_input[t]
        for (int b = 0; b < B; ++b) {
            for (int d = 0; d < input_size_; ++d) {
                grad_input.data()[t * B * input_size_ + b * input_size_ + d] =
                    dh.data()[b * input_size_ + d];
            }
        }
    }
}

void LSTM::update(float lr) {
    for (int l = 0; l < num_layers_; ++l) {
        input_to_hidden_[l]->update(lr);
        hidden_to_hidden_[l]->update(lr);
    }
}

void LSTM::save(const std::string& path) {
    for (int l = 0; l < num_layers_; ++l) {
        input_to_hidden_[l]->save(path + "_in2h_" + std::to_string(l));
        hidden_to_hidden_[l]->save(path + "_h2h_" + std::to_string(l));
    }
}

void LSTM::load(const std::string& path) {
    for (int l = 0; l < num_layers_; ++l) {
        input_to_hidden_[l]->load(path + "_in2h_" + std::to_string(l));
        hidden_to_hidden_[l]->load(path + "_h2h_" + std::to_string(l));
    }
}

