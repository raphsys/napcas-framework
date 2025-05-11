#include "tensor.h"
#include <stdexcept>

Tensor::Tensor(const std::vector<int>& shape, const std::vector<float>& data) : shape_(shape) {
    int expected_size = 1;
    for (int dim : shape) {
        if (dim <= 0) {
            throw std::invalid_argument("Shape dimensions must be positive");
        }
        expected_size *= dim;
    }
    if (!data.empty() && data.size() != expected_size) {
        throw std::invalid_argument("Data size does not match shape");
    }
    data_.resize(expected_size);
    if (!data.empty()) {
        data_ = data;
    } else {
        std::fill(data_.begin(), data_.end(), 0.0f);
    }
}

void Tensor::fill(float value) {
    std::fill(data_.begin(), data_.end(), value);
}

void Tensor::zero_grad() {
    fill(0.0f);
}

