#include "tensor.h"
#include "gpu_utils.h"
#include <fstream>
#include <stdexcept>
#include <numeric>

Tensor::Tensor(const std::vector<int>& shape, const std::vector<float>& data)
    : shape_(shape) {
    int size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    if (!data.empty() && static_cast<int>(data.size()) != size) {
        throw std::invalid_argument("Data size does not match shape");
    }
    data_.resize(size);
    if (!data.empty()) {
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
    int new_size = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<int>());
    if (new_size != size()) {
        throw std::invalid_argument("New shape size does not match tensor size");
    }
    shape_ = new_shape;
}

void Tensor::save(const std::string& path) const {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + path);
    }
    size_t shape_size = shape_.size();
    file.write(reinterpret_cast<const char*>(&shape_size), sizeof(shape_size));
    file.write(reinterpret_cast<const char*>(shape_.data()), shape_size * sizeof(int));
    file.write(reinterpret_cast<const char*>(data_.data()), data_.size() * sizeof(float));
    file.close();
}

void Tensor::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for reading: " + path);
    }
    size_t shape_size;
    file.read(reinterpret_cast<char*>(&shape_size), sizeof(shape_size));
    shape_.resize(shape_size);
    file.read(reinterpret_cast<char*>(shape_.data()), shape_size * sizeof(int));
    int size = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<int>());
    data_.resize(size);
    file.read(reinterpret_cast<char*>(data_.data()), size * sizeof(float));
    file.close();
}

#ifdef USE_CUDA
void Tensor::to_cuda() {
    if (cuda_data_) {
        return;
    }
    cuda_data_ = GPUUtils::allocate_cuda_memory(size());
    GPUUtils::copy_to_cuda(data_.data(), cuda_data_, size());
}

void Tensor::to_cpu() {
    if (!cuda_data_) {
        return;
    }
    GPUUtils::copy_from_cuda(cuda_data_, data_.data(), size());
    GPUUtils::free_cuda_memory(cuda_data_);
    cuda_data_ = nullptr;
}
#endif
