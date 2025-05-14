// File: cpp/src/pooling.cpp

#include "pooling.h"
#include <stdexcept>
#include <algorithm>
#include <limits>
#include <vector>

MaxPool2d::MaxPool2d(int kernel_size, int stride)
    : kernel_size_(kernel_size), stride_(stride) {
    weights_ = Tensor({1});
    grad_weights_ = Tensor({1});
}

void MaxPool2d::forward(Tensor& input, Tensor& output) {
    if (input.ndim() != 4) {
        throw std::invalid_argument("Input must be 4D (batch_size, channels, height, width)");
    }
    int batch_size = input.shape()[0];
    int channels   = input.shape()[1];
    int in_height  = input.shape()[2];
    int in_width   = input.shape()[3];
    int out_height = (in_height - kernel_size_) / stride_ + 1;
    int out_width  = (in_width  - kernel_size_) / stride_ + 1;

    if (output.shape() != std::vector<int>{batch_size, channels, out_height, out_width}) {
        throw std::invalid_argument("Output shape mismatch");
    }

    max_indices_.clear();
    max_indices_.resize(batch_size * channels * out_height * out_width);

    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < channels; ++c) {
            for (int oh = 0; oh < out_height; ++oh) {
                for (int ow = 0; ow < out_width; ++ow) {
                    float max_val = -std::numeric_limits<float>::infinity();
                    int   max_idx = 0;
                    for (int kh = 0; kh < kernel_size_; ++kh) {
                        for (int kw = 0; kw < kernel_size_; ++kw) {
                            int ih = oh * stride_ + kh;
                            int iw = ow * stride_ + kw;
                            if (ih < in_height && iw < in_width) {
                                float val = input[
                                    b * channels * in_height * in_width +
                                    c * in_height   * in_width   +
                                    ih * in_width   + iw
                                ];
                                if (val > max_val) {
                                    max_val = val;
                                    max_idx = kh * kernel_size_ + kw;
                                }
                            }
                        }
                    }
                    output[
                        b * channels * out_height * out_width +
                        c * out_height   * out_width   +
                        oh * out_width   + ow
                    ] = max_val;
                    max_indices_[
                        b * channels * out_height * out_width +
                        c * out_height   * out_width   +
                        oh * out_width   + ow
                    ] = { max_idx };
                }
            }
        }
    }
}

void MaxPool2d::backward(Tensor& grad_output, Tensor& grad_input) {
    if (grad_output.shape() != grad_input.shape()) {
        throw std::invalid_argument("Gradient shapes must match");
    }
    // On remet à zéro le gradient de l’entrée
    grad_input.fill(0.0f);

    int batch_size  = grad_output.shape()[0];
    int channels    = grad_output.shape()[1];
    int out_height  = grad_output.shape()[2];
    int out_width   = grad_output.shape()[3];
    int in_height   = grad_input.shape()[2];
    int in_width    = grad_input.shape()[3];

    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < channels; ++c) {
            for (int oh = 0; oh < out_height; ++oh) {
                for (int ow = 0; ow < out_width; ++ow) {
                    int idx     = b * channels * out_height * out_width +
                                  c * out_height   * out_width   +
                                  oh * out_width   + ow;
                    int max_loc = max_indices_[idx][0];
                    int kh      = max_loc / kernel_size_;
                    int kw      = max_loc % kernel_size_;
                    int ih      = oh * stride_ + kh;
                    int iw      = ow * stride_ + kw;
                    if (ih < in_height && iw < in_width) {
                        grad_input[
                            b * channels * in_height * in_width +
                            c * in_height   * in_width   +
                            ih * in_width   + iw
                        ] += grad_output[idx];
                    }
                }
            }
        }
    }
}

