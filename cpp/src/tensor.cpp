#include "napcas/tensor.h"
#include <Eigen/Dense>
#include <cstring>
#include <stdexcept>
#include <numeric>
#include <iostream>

namespace napcas {

namespace {
    void default_deleter(void* ptr) {
        std::free(ptr);
    }

    size_t compute_numel(const std::vector<std::size_t>& shape) {
        return std::accumulate(shape.begin(), shape.end(), 1UL, std::multiplies<>());
    }

    std::vector<std::ptrdiff_t> compute_strides_generic(const std::vector<std::size_t>& shape) {
        std::vector<std::ptrdiff_t> strides(shape.size());
        std::ptrdiff_t stride = 1;
        for (int i = shape.size() - 1; i >= 0; --i) {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    }
}

// ===================== Constructeurs =====================

Tensor::Tensor()
    : dtype_(DType::Float32), device_(DeviceType::CPU, 0), storage_(nullptr, default_deleter) {}

Tensor::Tensor(const std::vector<std::size_t>& shape, DType dtype, Device device)
    : shape_(shape), dtype_(dtype), device_(device), storage_(nullptr, default_deleter) {
    compute_strides();
    size_t size_bytes = compute_numel(shape_) * dtype_size(dtype_);
    void* raw = device_malloc(size_bytes, device_);
    storage_ = std::unique_ptr<void, void(*)(void*)>(raw, default_deleter);
}

template<typename Scalar>
Tensor::Tensor(const std::vector<std::size_t>& shape,
               const std::vector<Scalar>& data,
               DType dtype,
               Device device)
    : shape_(shape), dtype_(dtype), device_(device), storage_(nullptr, default_deleter) {
    compute_strides();
    size_t expected = compute_numel(shape);
    if (data.size() != expected) throw std::runtime_error("Mismatch in shape and data size");
    size_t size_bytes = expected * dtype_size(dtype_);
    void* raw = device_malloc(size_bytes, device_);
    std::memcpy(raw, data.data(), size_bytes);
    storage_ = std::unique_ptr<void, void(*)(void*)>(raw, default_deleter);
}

Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)), strides_(std::move(other.strides_)),
      dtype_(other.dtype_), device_(other.device_), storage_(std::move(other.storage_)),
      grad_ptr_(std::move(other.grad_ptr_)), grad_fn_(std::move(other.grad_fn_)),
      requires_grad_flag_(other.requires_grad_flag_) {}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        shape_ = std::move(other.shape_);
        strides_ = std::move(other.strides_);
        dtype_ = other.dtype_;
        device_ = other.device_;
        storage_ = std::move(other.storage_);
        grad_ptr_ = std::move(other.grad_ptr_);
        grad_fn_ = std::move(other.grad_fn_);
        requires_grad_flag_ = other.requires_grad_flag_;
    }
    return *this;
}


// ===================== Métadonnées =====================

std::size_t Tensor::numel() const noexcept {
    return compute_numel(shape_);
}

bool Tensor::is_contiguous() const noexcept {
    return strides_ == compute_strides_generic(shape_);
}

void Tensor::compute_strides() {
    strides_ = compute_strides_generic(shape_);
}

void Tensor::check_device_consistency(const Tensor& other) const {
    if (device_ != other.device_)
        throw std::runtime_error("Device mismatch");
}

void Tensor::check_shape_broadcast(const Tensor& other) const {
    if (shape_ != other.shape_)
        throw std::runtime_error("Shape mismatch: broadcasting not yet supported");
}

// ===================== Manipulations =====================

Tensor Tensor::clone() const {
    Tensor out(shape_, dtype_, device_);
    std::memcpy(out.storage_.get(), storage_.get(), numel() * dtype_size(dtype_));
    return out;
}

Tensor Tensor::detach() const {
    Tensor out = clone();
    out.requires_grad_flag_ = false;
    return out;
}

Tensor Tensor::astype(DType new_dtype) const {
    Tensor out(shape_, new_dtype, device_);
    if (dtype_ != new_dtype) throw std::runtime_error("astype not implemented yet");
    std::memcpy(out.storage_.get(), storage_.get(), numel() * dtype_size(dtype_));
    return out;
}

Tensor Tensor::to(Device new_device) const {
    if (new_device == device_) return clone();
    Tensor out(shape_, dtype_, new_device);
    std::memcpy(out.storage_.get(), storage_.get(), numel() * dtype_size(dtype_));
    return out;
}

Tensor Tensor::reshape(const std::vector<std::size_t>& new_shape) const {
    if (compute_numel(new_shape) != numel()) throw std::runtime_error("Invalid reshape");
    Tensor out = clone();
    out.shape_ = new_shape;
    out.compute_strides();
    return out;
}

Tensor Tensor::view(const std::vector<std::size_t>& new_shape) const {
    return reshape(new_shape);
}

Tensor Tensor::permute(const std::vector<int>& dims) const {
    if (dims.size() != shape_.size()) throw std::runtime_error("Invalid permutation");
    Tensor out = clone();
    std::vector<std::size_t> new_shape(shape_.size());
    std::vector<std::ptrdiff_t> new_strides(strides_.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        new_shape[i]   = shape_[dims[i]];
        new_strides[i] = strides_[dims[i]];
    }
    out.shape_   = new_shape;
    out.strides_ = new_strides;
    return out;
}

Tensor Tensor::transpose(int dim0, int dim1) const {
    std::vector<int> idx(shape_.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::swap(idx[dim0], idx[dim1]);
    return permute(idx);
}

Tensor Tensor::squeeze(int dim) const {
    Tensor out = clone();
    if (dim >= 0 && shape_[dim] == 1) {
        out.shape_.erase(out.shape_.begin() + dim);
        out.compute_strides();
    }
    return out;
}

Tensor Tensor::unsqueeze(int dim) {
    Tensor out = clone();
    out.shape_.insert(out.shape_.begin() + dim, 1);
    out.compute_strides();
    return out;
}

Tensor Tensor::contiguous() const {
    if (is_contiguous()) return clone();
    return clone();  // view-compatible placeholder
}

// ===================== Initialisateurs =====================

Tensor Tensor::zeros(const std::vector<std::size_t>& shape,
                     DType dtype,
                     Device device) {
    Tensor out(shape, dtype, device);
    std::memset(out.storage_.get(), 0, out.numel() * dtype_size(dtype));
    return out;
}

Tensor Tensor::ones(const std::vector<std::size_t>& shape,
                    DType dtype,
                    Device device) {
    Tensor out(shape, dtype, device);
    if (dtype == DType::Float32) {
        float* ptr = static_cast<float*>(out.storage_.get());
        std::fill(ptr, ptr + out.numel(), 1.0f);
    }
    return out;
}

// ===================== Opérations élémentaires =====================

Tensor Tensor::operator+(const Tensor& rhs) const {
    check_device_consistency(rhs);
    check_shape_broadcast(rhs);
    Tensor out(shape_, dtype_, device_);
    float* a = static_cast<float*>(storage_.get());
    float* b = static_cast<float*>(rhs.storage_.get());
    float* c = static_cast<float*>(out.storage_.get());
    for (size_t i = 0; i < numel(); ++i) c[i] = a[i] + b[i];
    return out;
}

Tensor Tensor::operator-(const Tensor& rhs) const {
    check_device_consistency(rhs);
    check_shape_broadcast(rhs);
    Tensor out(shape_, dtype_, device_);
    float* a = static_cast<float*>(storage_.get());
    float* b = static_cast<float*>(rhs.storage_.get());
    float* c = static_cast<float*>(out.storage_.get());
    for (size_t i = 0; i < numel(); ++i) c[i] = a[i] - b[i];
    return out;
}

Tensor Tensor::operator*(const Tensor& rhs) const {
    check_device_consistency(rhs);
    check_shape_broadcast(rhs);
    Tensor out(shape_, dtype_, device_);
    float* a = static_cast<float*>(storage_.get());
    float* b = static_cast<float*>(rhs.storage_.get());
    float* c = static_cast<float*>(out.storage_.get());
    for (size_t i = 0; i < numel(); ++i) c[i] = a[i] * b[i];
    return out;
}

Tensor Tensor::operator/(const Tensor& rhs) const {
    check_device_consistency(rhs);
    check_shape_broadcast(rhs);
    Tensor out(shape_, dtype_, device_);
    float* a = static_cast<float*>(storage_.get());
    float* b = static_cast<float*>(rhs.storage_.get());
    float* c = static_cast<float*>(out.storage_.get());
    for (size_t i = 0; i < numel(); ++i) c[i] = a[i] / b[i];
    return out;
}

Tensor Tensor::matmul(const Tensor& rhs) const {
    if (shape_.size() != 2 || rhs.shape_.size() != 2)
        throw std::runtime_error("matmul: only 2D supported");
    if (shape_[1] != rhs.shape_[0])
        throw std::runtime_error("matmul: shape mismatch");

    size_t m = shape_[0], k = shape_[1], n = rhs.shape_[1];
    Tensor out({m, n}, dtype_, device_);
    Eigen::Map<Eigen::MatrixXf> A(static_cast<float*>(storage_.get()), m, k);
    Eigen::Map<Eigen::MatrixXf> B(static_cast<float*>(rhs.storage_.get()), k, n);
    Eigen::Map<Eigen::MatrixXf> C(static_cast<float*>(out.storage_.get()), m, n);
    C.noalias() = A * B;
    return out;
}

// ===================== Affichage =====================

void Tensor::print_summary() const {
    std::cout << "Tensor(";
    std::cout << "shape=[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        std::cout << shape_[i];
        if (i < shape_.size() - 1) std::cout << ", ";
    }
    std::cout << "], dtype=";
    switch (dtype_) {
        case DType::Float32: std::cout << "float32"; break;
        default: std::cout << "unknown"; break;
    }
    std::cout << ", device=";
    switch (device_.type) {
        case DeviceType::CPU: std::cout << "cpu"; break;
        default: std::cout << "unknown"; break;
    }
    std::cout << ")" << std::endl;
}

void Tensor::print_shape() const {
    std::cout << "[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        std::cout << shape_[i];
        if (i < shape_.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

// ===================== Autograd access & backward =====================

Tensor& Tensor::grad() {
    if (!grad_ptr_) {
        // initialize grad to ones of same shape
        grad_ptr_.reset(new Tensor(Tensor::ones(shape_, dtype_, device_)));
    }
    return *grad_ptr_;
}

const Tensor& Tensor::grad() const {
    if (!grad_ptr_) {
        throw std::runtime_error("Gradient has not been initialized");
    }
    return *grad_ptr_;
}

void Tensor::backward() {
    if (!requires_grad_flag_) {
        throw std::runtime_error(
            "Cannot call backward() on a tensor that does not require grad");
    }
    // seed gradient if not set
    if (!grad_ptr_) {
        grad_ptr_.reset(new Tensor(Tensor::ones(shape_, dtype_, device_)));
    }
    if (grad_fn_) {
        grad_fn_->backward();
    }
}

// Instantiate template constructor
template Tensor::Tensor(const std::vector<std::size_t>&, const std::vector<float>&, DType, Device);

} // namespace napcas

