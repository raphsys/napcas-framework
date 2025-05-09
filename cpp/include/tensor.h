#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <stdexcept>

class Tensor {
public:
    Tensor() = default;
    Tensor(const std::vector<int>& shape, const std::vector<float>& data = {});
    int size() const { return data_.size(); }
    const std::vector<int>& shape() const { return shape_; }
    std::vector<float>& data() { return data_; }
    const std::vector<float>& data() const { return data_; }
    void fill(float value);
    void zero_grad();
    float& operator[](int i) { return data_[i]; }
    const float& operator[](int i) const { return data_[i]; }
    int ndim() const { return shape_.size(); }

private:
    std::vector<int> shape_;
    std::vector<float> data_;
};

#endif // TENSOR_H
