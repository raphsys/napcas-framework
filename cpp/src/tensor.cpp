#include "tensor.h"
#include <numeric>
#include <algorithm>

// Constructeur avec une forme donnée
Tensor::Tensor(const std::vector<int>& shape)
    : shape_(shape), data_(std::vector<float>(std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()), 0.0f)) {}

// Constructeur avec une forme et des données
Tensor::Tensor(const std::vector<int>& shape, const std::vector<float>& data)
    : shape_(shape), data_(data) {
    if (data_.size() != std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>())) {
        throw std::invalid_argument("Data size does not match tensor shape");
    }
}

// Méthode pour redimensionner le tenseur
void Tensor::reshape(const std::vector<int>& new_shape) {
    if (std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<int>()) != data_.size()) {
        throw std::invalid_argument("New shape size does not match data size");
    }
    shape_ = new_shape;
}

// Méthode pour remplir le tenseur avec une valeur donnée
void Tensor::fill(float value) {
    std::fill(data_.begin(), data_.end(), value);
}

// Méthode pour remettre à zéro les gradients
void Tensor::zero_grad() {
    std::fill(data_.begin(), data_.end(), 0.0f);
}
