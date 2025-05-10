#include "conv2d.h"
#include <Eigen/Dense>
#include <stdexcept>
#include <vector>

Conv2d::Conv2d(int in_channels, int out_channels, int kernel_size)
    : kernel_size_(kernel_size), learning_rate_(0.0f) {
    if (kernel_size % 2 == 0) {
        throw std::invalid_argument("Kernel size must be odd");
    }
    std::vector<int> weight_shape = {out_channels, in_channels, kernel_size, kernel_size};
    std::vector<int> bias_shape = {out_channels};
    weights_ = Tensor(weight_shape);
    bias_ = Tensor(bias_shape);
    grad_weights_ = Tensor(weight_shape);
    grad_bias_ = Tensor(bias_shape);

    // Initialisation des poids (exemple : initialisation aléatoire simple)
    for (int i = 0; i < weights_.size(); ++i) {
        weights_[i] = static_cast<float>(rand()) / RAND_MAX - 0.5f;
    }
    for (int i = 0; i < bias_.size(); ++i) {
        bias_[i] = 0.0f;
    }
}

void Conv2d::forward(Tensor& input, Tensor& output) {
    if (input.ndim() != 4) {
        throw std::invalid_argument("Input must be 4D (batch, channels, height, width)");
    }
    int batch_size = input.shape()[0];
    int in_channels = input.shape()[1];
    int height = input.shape()[2];
    int width = input.shape()[3];
    int out_channels = weights_.shape()[0];
    int out_height = height - kernel_size_ + 1;
    int out_width = width - kernel_size_ + 1;

    if (output.shape() != std::vector<int>{batch_size, out_channels, out_height, out_width}) {
        throw std::invalid_argument("Output shape mismatch");
    }

    // Conversion des tenseurs en matrices Eigen pour le calcul
    Eigen::Map<Eigen::MatrixXf> weights_mat(weights_.data().data(), out_channels, in_channels * kernel_size_ * kernel_size_);
    Eigen::Map<Eigen::MatrixXf> output_mat(output.data().data(), batch_size * out_height * out_width, out_channels);

    // Extraction des patches de l'entrée (im2col)
    std::vector<float> input_patches(batch_size * out_height * out_width * in_channels * kernel_size_ * kernel_size_);
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < out_height; ++h) {
            for (int w = 0; w < out_width; ++w) {
                for (int c = 0; c < in_channels; ++c) {
                    for (int kh = 0; kh < kernel_size_; ++kh) {
                        for (int kw = 0; kw < kernel_size_; ++kw) {
                            int idx = b * out_height * out_width * in_channels * kernel_size_ * kernel_size_ +
                                     (h * out_width + w) * in_channels * kernel_size_ * kernel_size_ +
                                     c * kernel_size_ * kernel_size_ + kh * kernel_size_ + kw;
                            int input_idx = b * in_channels * height * width +
                                           c * height * width +
                                           (h + kh) * width + (w + kw);
                            input_patches[idx] = input[input_idx];
                        }
                    }
                }
            }
        }
    }

    Eigen::Map<Eigen::MatrixXf> input_patches_mat(input_patches.data(),
        batch_size * out_height * out_width, in_channels * kernel_size_ * kernel_size_);
    output_mat = input_patches_mat * weights_mat.transpose();

    // Ajout du biais
    Eigen::Map<Eigen::VectorXf> bias_vec(bias_.data().data(), bias_.shape()[0]);
    for (int i = 0; i < batch_size * out_height * out_width; ++i) {
        output_mat.row(i) += bias_vec.transpose();
    }
}

void Conv2d::backward(Tensor& grad_output, Tensor& grad_input) {
    if (grad_output.ndim() != 4 || grad_input.ndim() != 4) {
        throw std::invalid_argument("Gradients must be 4D");
    }
    int batch_size = grad_input.shape()[0];
    int in_channels = grad_input.shape()[1];
    int height = grad_input.shape()[2];
    int width = grad_input.shape()[3];
    int out_channels = weights_.shape()[0];
    int out_height = height - kernel_size_ + 1;
    int out_width = width - kernel_size_ + 1;

    if (grad_output.shape() != std::vector<int>{batch_size, out_channels, out_height, out_width}) {
        throw std::invalid_argument("Gradient output shape mismatch");
    }

    // Conversion des tenseurs en matrices Eigen
    Eigen::Map<Eigen::MatrixXf> grad_output_mat(grad_output.data().data(), batch_size * out_height * out_width, out_channels);
    Eigen::Map<Eigen::MatrixXf> grad_input_mat(grad_input.data().data(), batch_size * height * width, in_channels);
    Eigen::Map<Eigen::MatrixXf> grad_weights_mat(grad_weights_.data().data(), out_channels, in_channels * kernel_size_ * kernel_size_);
    Eigen::Map<Eigen::VectorXf> grad_bias_vec(grad_bias_.data().data(), grad_bias_.shape()[0]);

    // Calcul du gradient des poids et du biais
    std::vector<float> input_patches(batch_size * out_height * out_width * in_channels * kernel_size_ * kernel_size_, 0.0f);
    Eigen::Map<Eigen::MatrixXf> input_patches_mat(input_patches.data(),
        batch_size * out_height * out_width, in_channels * kernel_size_ * kernel_size_);
    Eigen::Map<Eigen::MatrixXf> weights_mat(weights_.data().data(), out_channels, in_channels * kernel_size_ * kernel_size_);

    // Gradient du biais
    grad_bias_vec = grad_output_mat.colwise().sum();

    // Gradient des poids
    grad_weights_mat = grad_output_mat.transpose() * input_patches_mat;

    // Gradient de l'entrée
    grad_input_mat.setZero();
    Eigen::MatrixXf grad_input_patches = grad_output_mat * weights_mat;
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < out_height; ++h) {
            for (int w = 0; w < out_width; ++w) {
                for (int c = 0; c < in_channels; ++c) {
                    for (int kh = 0; kh < kernel_size_; ++kh) {
                        for (int kw = 0; kw < kernel_size_; ++kw) {
                            int idx = b * out_height * out_width * in_channels * kernel_size_ * kernel_size_ +
                                     (h * out_width + w) * in_channels * kernel_size_ * kernel_size_ +
                                     c * kernel_size_ * kernel_size_ + kh * kernel_size_ + kw;
                            int grad_idx = b * in_channels * height * width +
                                          c * height * width +
                                          (h + kh) * width + (w + kw);
                            grad_input_mat(grad_idx, c) += grad_input_patches(idx, c);
                        }
                    }
                }
            }
        }
    }
}

void Conv2d::update(float lr) {
    learning_rate_ = lr;
    for (int i = 0; i < weights_.size(); ++i) {
        weights_[i] -= lr * grad_weights_[i];
    }
    for (int i = 0; i < bias_.size(); ++i) {
        bias_[i] -= lr * grad_bias_[i];
    }
}

Tensor& Conv2d::get_weights() { return weights_; }
Tensor& Conv2d::get_grad_weights() { return grad_weights_; }

void Conv2d::set_weights(const Tensor& weights) {
    this->weights_ = weights;
}
