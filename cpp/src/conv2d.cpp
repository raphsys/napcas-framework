#include "conv2d.h"
#include <random>
#include <cmath>

Conv2d::Conv2d(int in_channels, int out_channels, int kernel_size)
    : kernel_size_(kernel_size),
      weights_(std::vector<int>{out_channels, in_channels, kernel_size, kernel_size}, std::vector<float>(out_channels * in_channels * kernel_size * kernel_size, 0.0f)),
      bias_(std::vector<int>{out_channels}, std::vector<float>(out_channels, 0.0f)),
      grad_weights_(std::vector<int>{out_channels, in_channels, kernel_size, kernel_size}, std::vector<float>(out_channels * in_channels * kernel_size * kernel_size, 0.0f)),
      grad_bias_(std::vector<int>{out_channels}, std::vector<float>(out_channels, 0.0f)),
      learning_rate_(0.01f) {

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(0.0f, 0.01f);

    for (int i = 0; i < weights_.size(); ++i) {
        weights_[i] = dist(gen);
    }
}

void Conv2d::forward(Tensor& input, Tensor& output) {
    // Implémentation de la convolution 2D
}

void Conv2d::backward(Tensor& grad_output, Tensor& grad_input) {
    // Rétropropagation
}

void Conv2d::update(float lr) {
    // Mise à jour des poids
}
