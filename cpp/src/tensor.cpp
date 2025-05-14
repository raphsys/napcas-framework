// File: cpp/src/tensor.cpp
#include "tensor.h"
#include <fstream>
#include <algorithm>
#include <stdexcept>

Tensor::Tensor(const std::vector<int>& shape, const std::vector<float>& data)
    : shape_(shape)
{
    // Calcul du nombre total d'éléments
    int total = 1;
    for (int d : shape_) {
        if (d <= 0) throw std::invalid_argument("Tensor shape dimensions must be positive");
        total *= d;
    }

    // Allocation et initialisation
    data_.resize(total);
    if (!data.empty()) {
        if (static_cast<int>(data.size()) != total) {
            throw std::invalid_argument("Data size does not match tensor shape");
        }
        data_ = data;
    } else {
        fill(0.0f);
    }
}

void Tensor::fill(float value) {
    std::fill(data_.begin(), data_.end(), value);
}

void Tensor::zero_grad() {
    fill(0.0f);
}

void Tensor::reshape(const std::vector<int>& new_shape) {
    // Vérification et calcul du nouveau total
    int new_total = 1;
    for (int d : new_shape) {
        if (d <= 0) throw std::invalid_argument("Tensor reshape dimensions must be positive");
        new_total *= d;
    }
    if (new_total != size()) {
        throw std::invalid_argument("Reshape total elements mismatch");
    }
    shape_ = new_shape;
}

void Tensor::save(const std::string& path) const {
    std::ofstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open file for writing: " + path);

    // Écriture du nombre de dimensions et de la forme
    size_t dims = shape_.size();
    file.write(reinterpret_cast<const char*>(&dims), sizeof(dims));
    file.write(reinterpret_cast<const char*>(shape_.data()), dims * sizeof(int));

    // Écriture des données brutes
    file.write(reinterpret_cast<const char*>(data_.data()), data_.size() * sizeof(float));
    file.close();
}

void Tensor::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open file for reading: " + path);

    // Lecture du nombre de dimensions et de la forme
    size_t dims;
    file.read(reinterpret_cast<char*>(&dims), sizeof(dims));
    shape_.resize(dims);
    file.read(reinterpret_cast<char*>(shape_.data()), dims * sizeof(int));

    // Calcul du total et lecture des données
    int total = 1;
    for (int d : shape_) {
        if (d <= 0) throw std::runtime_error("Invalid tensor shape in file: " + path);
        total *= d;
    }
    data_.resize(total);
    file.read(reinterpret_cast<char*>(data_.data()), total * sizeof(float));
    file.close();
}

#ifdef USE_CUDA
#include "gpu_utils.h"

void Tensor::to_cuda() {
    if (cuda_data_) return;
    cuda_data_ = GPUUtils::allocate_cuda_memory(size());
    GPUUtils::copy_to_cuda(data_.data(), cuda_data_, size());
}

void Tensor::to_cpu() {
    if (!cuda_data_) return;
    GPUUtils::copy_from_cuda(cuda_data_, data_.data(), size());
    GPUUtils::free_cuda_memory(cuda_data_);
    cuda_data_ = nullptr;
}
#endif

