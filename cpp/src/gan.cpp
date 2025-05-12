#include "gan.h"
#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>
#include <fstream>

GAN::GAN(const std::vector<int>& generator_layers, const std::vector<int>& discriminator_layers) {
    for (size_t i = 1; i < generator_layers.size(); ++i) {
        generator_.push_back(std::make_shared<Linear>(generator_layers[i-1], generator_layers[i]));
        if (i < generator_layers.size() - 1) {
            generator_.push_back(std::make_shared<ReLU>());
        } else {
            generator_.push_back(std::make_shared<Sigmoid>());
        }
    }
    for (size_t i = 1; i < discriminator_layers.size(); ++i) {
        discriminator_.push_back(std::make_shared<Linear>(discriminator_layers[i-1], discriminator_layers[i]));
        if (i < discriminator_layers.size() - 1) {
            discriminator_.push_back(std::make_shared<ReLU>());
        } else {
            discriminator_.push_back(std::make_shared<Sigmoid>());
        }
    }
    weights_ = Tensor({static_cast<int>(generator_.size() + discriminator_.size()), 1});
    grad_weights_ = Tensor({static_cast<int>(generator_.size() + discriminator_.size()), 1});
}

void GAN::forward(Tensor& input, Tensor& output) {
    Tensor temp = input;
    for (auto& layer : generator_) {
        Tensor next;
        layer->forward(temp, next);
        temp = next;
    }
    output = temp;
}

void GAN::backward(Tensor& grad_output, Tensor& grad_input) {
    Tensor grad_temp = grad_output;
    for (auto it = generator_.rbegin(); it != generator_.rend(); ++it) {
        Tensor grad_next;
        (*it)->backward(grad_temp, grad_next);
        grad_temp = grad_next;
    }
    grad_input = grad_temp;
}

void GAN::train_step(Tensor& real_data, Tensor& noise, float lr, Tensor& gen_loss, Tensor& disc_loss) {
    int batch_size = real_data.shape()[0];
    Tensor fake_data({batch_size, real_data.shape()[1]});
    forward(noise, fake_data);

    // Discriminator loss
    Tensor real_labels({batch_size, 1}, std::vector<float>(batch_size, 1.0f));
    Tensor fake_labels({batch_size, 1}, std::vector<float>(batch_size, 0.0f));

    Tensor disc_real_out({batch_size, 1});
    Tensor disc_fake_out({batch_size, 1});
    Tensor disc_temp = real_data;
    for (auto& layer : discriminator_) {
        Tensor next;
        layer->forward(disc_temp, next);
        disc_temp = next;
    }
    disc_real_out = disc_temp;

    disc_temp = fake_data;
    for (auto& layer : discriminator_) {
        Tensor next;
        layer->forward(disc_temp, next);
        disc_temp = next;
    }
    disc_fake_out = disc_temp;

    // Binary cross-entropy loss
    float d_loss_real = 0.0f;
    float d_loss_fake = 0.0f;
    for (int i = 0; i < batch_size; ++i) {
        d_loss_real -= std::log(disc_real_out[i] + 1e-10f);
        d_loss_fake -= std::log(1.0f - disc_fake_out[i] + 1e-10f);
    }
    disc_loss[0] = (d_loss_real + d_loss_fake) / batch_size;

    // Discriminator backward
    Tensor grad_disc_real({batch_size, 1});
    Tensor grad_disc_fake({batch_size, 1});
    for (int i = 0; i < batch_size; ++i) {
        grad_disc_real[i] = -1.0f / (disc_real_out[i] + 1e-10f);
        grad_disc_fake[i] = 1.0f / (1.0f - disc_fake_out[i] + 1e-10f);
    }
    Tensor grad_disc_temp = grad_disc_real;
    for (auto it = discriminator_.rbegin(); it != discriminator_.rend(); ++it) {
        Tensor grad_next;
        (*it)->backward(grad_disc_temp, grad_next);
        grad_disc_temp = grad_next;
    }
    grad_disc_temp = grad_disc_fake;
    for (auto it = discriminator_.rbegin(); it != discriminator_.rend(); ++it) {
        Tensor grad_next;
        (*it)->backward(grad_disc_temp, grad_next);
        grad_disc_temp = grad_next;
    }

    // Generator loss
    Tensor gen_labels({batch_size, 1}, std::vector<float>(batch_size, 1.0f));
    Tensor gen_out({batch_size, 1});
    disc_temp = fake_data;
    for (auto& layer : discriminator_) {
        Tensor next;
        layer->forward(disc_temp, next);
        disc_temp = next;
    }
    gen_out = disc_temp;

    float g_loss = 0.0f;
    for (int i = 0; i < batch_size; ++i) {
        g_loss -= std::log(gen_out[i] + 1e-10f);
    }
    gen_loss[0] = g_loss / batch_size;

    // Generator backward
    Tensor grad_gen({batch_size, 1});
    for (int i = 0; i < batch_size; ++i) {
        grad_gen[i] = -1.0f / (gen_out[i] + 1e-10f);
    }
    grad_disc_temp = grad_gen;
    for (auto it = discriminator_.rbegin(); it != discriminator_.rend(); ++it) {
        Tensor grad_next;
        (*it)->backward(grad_disc_temp, grad_next);
        grad_disc_temp = grad_next;
    }
    backward(grad_disc_temp, grad_disc_temp);
}

void GAN::update(float lr) {
    for (auto& layer : generator_) {
        layer->update(lr);
    }
    for (auto& layer : discriminator_) {
        layer->update(lr);
    }
}

void GAN::set_weights(const Tensor& weights) {
    weights_ = weights;
}

void GAN::save(const std::string& path) {
    for (size_t i = 0; i < generator_.size(); ++i) {
        generator_[i]->save(path + "_generator_" + std::to_string(i));
    }
    for (size_t i = 0; i < discriminator_.size(); ++i) {
        discriminator_[i]->save(path + "_discriminator_" + std::to_string(i));
    }
}

void GAN::load(const std::string& path) {
    for (size_t i = 0; i < generator_.size(); ++i) {
        generator_[i]->load(path + "_generator_" + std::to_string(i));
    }
    for (size_t i = 0; i < discriminator_.size(); ++i) {
        discriminator_[i]->load(path + "_discriminator_" + std::to_string(i));
    }
}
