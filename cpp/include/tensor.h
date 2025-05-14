#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <stdexcept>
#include <numeric>
#include <functional>
#include <string>

#ifdef USE_CUDA
#include "gpu_utils.h"
#endif

/// @brief Tensor class for multi-dimensional arrays.
class Tensor {
public:
    Tensor() = default;
    /// @brief Constructs a tensor with given shape and optional initial data.
    /// @param shape Shape of the tensor.
    /// @param data Optional initial data; if empty, tensor is zero-initialized.
    Tensor(const std::vector<int>& shape, const std::vector<float>& data = {});

    /// @brief Total number of elements.
    int size() const { return static_cast<int>(data_.size()); }
    /// @brief Number of dimensions.
    int ndim() const { return static_cast<int>(shape_.size()); }

    /// @brief Shape vector.
    const std::vector<int>& shape() const { return shape_; }

    /// @brief Raw data access.
    std::vector<float>& data() { return data_; }
    const std::vector<float>& data() const { return data_; }

    /// @brief Fill every element with a value.
    void fill(float value);
    /// @brief Zero the tensor (alias for fill(0)).
    void zero_grad();

    /// @brief Element access.
    float& operator[](int i) { return data_.at(i); }
    const float& operator[](int i) const { return data_.at(i); }

    /// @brief Reshape the tensor (must preserve total size).
    void reshape(const std::vector<int>& new_shape);

    /// @brief Save tensor to binary file.
    void save(const std::string& path) const;
    /// @brief Load tensor from binary file.
    void load(const std::string& path);

#ifdef USE_CUDA
    /// @brief Transfer data to GPU.
    void to_cuda();
    /// @brief Transfer data back to CPU.
    void to_cpu();
#endif

private:
    std::vector<int> shape_;
    std::vector<float> data_;
#ifdef USE_CUDA
    float* cuda_data_ = nullptr;
#endif
};

#endif // TENSOR_H

