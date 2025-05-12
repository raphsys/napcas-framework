#include "conv2d.h"
#include <Eigen/Dense>
#include <stdexcept>
#include <fstream>
#include <vector>

Conv2d::Conv2d(int in_channels, int out_channels, int kernel_size, int stride, int padding)
    : in_channels_(in_channels), out_channels_(out_channels), kernel_size_(kernel_size),
      stride_(stride), padding_(padding) {
    weights_ = Tensor({out_channels, in_channels, kernel_size, kernel_size});
    bias_ = Tensor({out_channels});
    grad_weights_ = Tensor({out_channels, in_channels, kernel_size, kernel_size});
    grad_bias_ = Tensor({out_channels});
    weights_.fill(0.01f); // Simple initialization
    bias_.fill(0.0f);
}

void Conv2d::im2col(const Tensor& input, Eigen::MatrixXf& col, int out_height, int out_width) {
    int batch_size = input.shape()[0];
    int in_height = input.shape()[2];
    int in_width = input.shape()[3];
    int patch_size = in_channels_ * kernel_size_ * kernel_size_;

    col.resize(patch_size, out_height * out_width * batch_size);

    std::vector<float> padded_input;
    if (padding_ > 0) {
        int padded_height = in_height + 2 * padding_;
        int padded_width = in_width + 2 * padding_;
        padded_input.resize(batch_size * in_channels_ * padded_height * padded_width, 0.0f);
        for (int b = 0; b < batch_size; ++b) {
            for (int c = 0; c < in_channels_; ++c) {
                for (int h = 0; h < in_height; ++h) {
                    for (int w = 0; w < in_width; ++w) {
                        int src_idx = b * in_channels_ * in_height * in_width +
                                      c * in_height * in_width + h * in_width + w;
                        int dst_idx = b * in_channels_ * padded_height * padded_width +
                                      c * padded_height * padded_width + (h + padding_) * padded_width + (w + padding_);
                        padded_input[dst_idx] = input[src_idx];
                    }
                }
            }
        }
    } else {
        padded_input = input.data();
    }

    int col_idx = 0;
    for (int b = 0; b < batch_size; ++b) {
        for (int oh = 0; oh < out_height; ++oh) {
            for (int ow = 0; ow < out_width; ++ow) {
                for (int c = 0; c < in_channels_; ++c) {
                    for (int kh = 0; kh < kernel_size_; ++kh) {
                        for (int kw = 0; kw < kernel_size_; ++kw) {
                            int h = oh * stride_ + kh;
                            int w = ow * stride_ + kw;
                            int padded_height = in_height + 2 * padding_;
                            int padded_width = in_width + 2 * padding_;
                            int idx = b * in_channels_ * padded_height * padded_width +
                                      c * padded_height * padded_width + h * padded_width + w;
                            col(col_idx++, oh * out_width * batch_size + ow * batch_size + b) = padded_input[idx];
                        }
                    }
                }
            }
        }
    }
}

void Conv2d::forward(Tensor& input, Tensor& output) {
    if (input.ndim() != 4) {
        throw std::invalid_argument("Input must be 4D (batch_size, in_channels, height, width)");
    }
    int batch_size = input.shape()[0];
    int in_height = input.shape()[2];
    int in_width = input.shape()[3];
    int out_height = (in_height + 2 * padding_ - kernel_size_) / stride_ + 1;
    int out_width = (in_width + 2 * padding_ - kernel_size_) / stride_ + 1;
    if (output.shape() != std::vector<int>{batch_size, out_channels_, out_height, out_width}) {
        throw std::invalid_argument("Output shape mismatch");
    }

    // im2col transformation
    Eigen::MatrixXf col;
    im2col(input, col, out_height, out_width);

    // Reshape weights to (out_channels, in_channels * kernel_size * kernel_size)
    Eigen::Map<Eigen::MatrixXf> weight_mat(weights_.data().data(), out_channels_,
                                           in_channels_ * kernel_size_ * kernel_size_);
    Eigen::MatrixXf output_mat(out_channels_, out_height * out_width * batch_size);
    output_mat = weight_mat * col;

    // Add bias and reshape to output tensor
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < out_channels_; ++c) {
            for (int h = 0; h < out_height; ++h) {
                for (int w = 0; w < out_width; ++w) {
                    int idx = b * out_channels_ * out_height * out_width +
                              c * out_height * out_width + h * out_width + w;
                    output[idx] = output_mat(c, h * out_width * batch_size + w * batch_size + b) + bias_[c];
                }
            }
        }
    }

    // Cache for backward pass
    input_cache_ = input;
    col_cache_ = col;
}

void Conv2d::backward(Tensor& grad_output, Tensor& grad_input) {
    int batch_size = grad_output.shape()[0];
    int out_height = grad_output.shape()[2];
    int out_width = grad_output.shape()[3];
    int in_height = input_cache_.shape()[2];
    int in_width = input_cache_.shape()[3];

    // Reshape grad_output to (out_channels, out_height * out_width * batch_size)
    Eigen::MatrixXf grad_output_mat(out_channels_, out_height * out_width * batch_size);
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < out_channels_; ++c) {
            for (int h = 0; h < out_height; ++h) {
                for (int w = 0; w < out_width; ++w) {
                    int idx = b * out_channels_ * out_height * out_width +
                              c * out_height * out_width + h * out_width + w;
                    grad_output_mat(c, h * out_width * batch_size + w * batch_size + b) = grad_output[idx];
                }
            }
        }
    }

    // Compute gradients for weights
    Eigen::Map<Eigen::MatrixXf> grad_weight_mat(grad_weights_.data().data(), out_channels_,
                                                in_channels_ * kernel_size_ * kernel_size_);
    grad_weight_mat = grad_output_mat * col_cache_.transpose();

    // Compute gradients for bias
    for (int c = 0; c < out_channels_; ++c) {
        grad_bias_[c] = grad_output_mat.row(c).sum();
    }

    // Compute gradients for input
    grad_input.fill(0.0f);
    Eigen::Map<Eigen::MatrixXf> weight_mat(weights_.data().data(), out_channels_,
                                           in_channels_ * kernel_size_ * kernel_size_);
    Eigen::MatrixXf grad_col = weight_mat.transpose() * grad_output_mat;

    // col2im to compute grad_input
    int padded_height = in_height + 2 * padding_;
    int padded_width = in_width + 2 * padding_;
    std::vector<float> grad_padded(batch_size * in_channels_ * padded_height * padded_width, 0.0f);
    int col_idx = 0;
    for (int b = 0; b < batch_size; ++b) {
        for (int oh = 0; oh < out_height; ++oh) {
            for (int ow = 0; ow < out_width; ++ow) {
                for (int c = 0; c < in_channels_; ++c) {
                    for (int kh = 0; kh < kernel_size_; ++kh) {
                        for (int kw = 0; kw < kernel_size_; ++kw) {
                            int h = oh * stride_ + kh;
                            int w = ow * stride_ + kw;
                            int idx = b * in_channels_ * padded_height * padded_width +
                                      c * padded_height * padded_width + h * padded_width + w;
                            grad_padded[idx] += grad_col(col_idx++, oh * out_width * batch_size + ow * batch_size + b);
                        }
                    }
                }
            }
        }
    }

    // Extract grad_input from padded gradient
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < in_channels_; ++c) {
            for (int h = 0; h < in_height; ++h) {
                for (int w = 0; w < in_width; ++w) {
                    int src_idx = b * in_channels_ * padded_height * padded_width +
                                  c * padded_height * padded_width + (h + padding_) * padded_width + (w + padding_);
                    int dst_idx = b * in_channels_ * in_height * in_width +
                                  c * in_height * in_width + h * in_width + w;
                    grad_input[dst_idx] = grad_padded[src_idx];
                }
            }
        }
    }
}

void Conv2d::update(float lr) {
    for (size_t i = 0; i < weights_.size(); ++i) {
        weights_[i] -= lr * grad_weights_[i];
    }
    for (size_t i = 0; i < bias_.size(); ++i) {
        bias_[i] -= lr * grad_bias_[i];
    }
}

void Conv2d::set_weights(const Tensor& weights) {
    if (weights.shape() != std::vector<int>{out_channels_, in_channels_, kernel_size_, kernel_size_}) {
        throw std::invalid_argument("Weight shape mismatch");
    }
    weights_ = weights;
}

void Conv2d::save(const std::string& path) {
    weights_.save(path + "_weights.tensor");
    bias_.save(path + "_bias.tensor");
}

void Conv2d::load(const std::string& path) {
    weights_.load(path + "_weights.tensor");
    bias_.load(path + "_bias.tensor");
}
