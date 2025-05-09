#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <memory>
#include <stdexcept>

class Tensor {
public:
    Tensor() = default; // Constructeur par défaut
    Tensor(const std::vector<int>& shape);
    Tensor(const std::vector<int>& shape, const std::vector<float>& data);

    int size() const { return static_cast<int>(data_.size()); }
    int ndim() const { return static_cast<int>(shape_.size()); }
    const std::vector<int>& shape() const { return shape_; }

    float& operator[](int index) { return data_[index]; }
    const float& operator[](int index) const { return data_[index]; }

    void reshape(const std::vector<int>& new_shape);
    void fill(float value);
    void zero_grad();

private:
    std::vector<int> shape_;
    std::vector<float> data_;
};

#endif // TENSOR_H
