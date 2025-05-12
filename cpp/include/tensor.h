#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <stdexcept>

/// @brief Tensor class for multi-dimensional arrays.
class Tensor {
public:
    Tensor() = default;
    /// @brief Constructs a tensor with given shape and optional data.
    /// @param shape Shape of the tensor.
    /// @param data Optional initial data.
    Tensor(const std::vector<int>& shape, const std::vector<float>& data = {});
    /// @brief Gets the total number of elements.
    /// @return Size of the tensor.
    int size() const { return data_.size(); }
    /// @brief Gets the shape of the tensor.
    /// @return Reference to shape vector.
    const std::vector<int>& shape() const { return shape_; }
    /// @brief Gets the tensor data.
    /// @return Reference to data vector.
    std::vector<float>& data() { return data_; }
    /// @brief Gets the tensor data (const).
    /// @return Const reference to data vector.
    const std::vector<float>& data() const { return data_; }
    /// @brief Fills the tensor with a value.
    /// @param value Value to fill.
    void fill(float value);
    /// @brief Zeros the gradient.
    void zero_grad();
    /// @brief Accesses element at index.
    /// @param i Index.
    /// @return Reference to element.
    float& operator[](int i) { return data_[i]; }
    /// @brief Accesses element at index (const).
    /// @param i Index.
    /// @return Const reference to element.
    const float& operator[](int i) const { return data_[i]; }
    /// @brief Gets the number of dimensions.
    /// @return Number of dimensions.
    int ndim() const { return shape_.size(); }
    /// @brief Reshapes the tensor.
    /// @param new_shape New shape.
    void reshape(const std::vector<int>& new_shape);
    /// @brief Saves tensor to file.
    /// @param path File path.
    void save(const std::string& path) const;
    /// @brief Loads tensor from file.
    /// @param path File path.
    void load(const std::string& path);
#ifdef USE_CUDA
    /// @brief Moves tensor to GPU.
    void to_cuda();
    /// @brief Moves tensor to CPU.
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
