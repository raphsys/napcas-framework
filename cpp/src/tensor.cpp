// cpp/src/tensor.cpp

#include "napcas/tensor.h"
#include "napcas/grad_fn.h"
#include <unordered_set>
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
        for (int i = int(shape.size()) - 1; i >= 0; --i) {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    }
}

// ===================== Constructeurs =====================

Tensor::Tensor()
    : dtype_(DType::Float32),
      device_(DeviceType::CPU, 0),
      storage_(nullptr, default_deleter),
      requires_grad_flag_(false)
{}

Tensor::Tensor(const std::vector<std::size_t>& shape,
               DType dtype,
               Device device)
    : shape_(shape),
      dtype_(dtype),
      device_(device),
      storage_(nullptr, default_deleter),
      requires_grad_flag_(false)
{
    compute_strides();
    size_t size_bytes = compute_numel(shape_) * dtype_size(dtype_);
    void* raw = device_malloc(size_bytes, device_);
    storage_.reset(raw);
}

template<typename Scalar>
Tensor::Tensor(const std::vector<std::size_t>& shape,
               const std::vector<Scalar>& data,
               DType dtype,
               Device device)
    : shape_(shape),
      dtype_(dtype),
      device_(device),
      storage_(nullptr, default_deleter),
      requires_grad_flag_(false)
{
    compute_strides();
    size_t expected = compute_numel(shape);
    if (data.size() != expected)
        throw std::runtime_error("Mismatch in shape and data size");
    size_t size_bytes = expected * dtype_size(dtype_);
    void* raw = device_malloc(size_bytes, device_);
    std::memcpy(raw, data.data(), size_bytes);
    storage_.reset(raw);
}

Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)),
      strides_(std::move(other.strides_)),
      dtype_(other.dtype_),
      device_(other.device_),
      storage_(std::move(other.storage_)),
      grad_ptr_(std::move(other.grad_ptr_)),
      grad_fn_(std::move(other.grad_fn_)),
      requires_grad_flag_(other.requires_grad_flag_)
{}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        shape_               = std::move(other.shape_);
        strides_             = std::move(other.strides_);
        dtype_               = other.dtype_;
        device_              = other.device_;
        storage_             = std::move(other.storage_);
        grad_ptr_            = std::move(other.grad_ptr_);
        grad_fn_             = std::move(other.grad_fn_);
        requires_grad_flag_  = other.requires_grad_flag_;
    }
    return *this;
}

// ===================== Autograd setup =====================

void Tensor::set_grad_fn(std::shared_ptr<GradFn> fn) {
    grad_fn_ = std::move(fn);
    requires_grad_flag_ = true;
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
    std::memcpy(out.storage_.get(), storage_.get(),
                numel() * dtype_size(dtype_));
    return out;
}

Tensor Tensor::detach() const {
    Tensor out = clone();
    out.requires_grad_flag_ = false;
    return out;
}

Tensor Tensor::astype(DType new_dtype) const {
    Tensor out(shape_, new_dtype, device_);
    if (dtype_ != new_dtype)
        throw std::runtime_error("astype not implemented yet");
    std::memcpy(out.storage_.get(), storage_.get(),
                numel() * dtype_size(dtype_));
    return out;
}

Tensor Tensor::to(Device new_device) const {
    if (new_device == device_)
        return clone();
    Tensor out(shape_, dtype_, new_device);
    std::memcpy(out.storage_.get(), storage_.get(),
                numel() * dtype_size(dtype_));
    return out;
}

// -- reshape with backward tracking --

Tensor Tensor::reshape(const std::vector<std::size_t>& new_shape) const {
    if (compute_numel(new_shape) != numel()) {
        throw std::runtime_error("Invalid reshape");
    }
    Tensor out = clone();
    std::vector<std::size_t> old_shape = shape_;
    out.shape_ = new_shape;
    out.compute_strides();
    if (requires_grad_flag_) {
        out.requires_grad_flag_ = true;
        out.set_grad_fn(
            std::make_shared<ReshapeBackward>(
                const_cast<Tensor*>(this),
                &out,
                std::move(old_shape)
            )
        );
    }
    return out;
}

Tensor Tensor::view(const std::vector<std::size_t>& new_shape) const {
    return reshape(new_shape);
}

// -- permute with backward tracking --

Tensor Tensor::permute(const std::vector<int>& dims) const {
    if (dims.size() != shape_.size())
        throw std::runtime_error("Invalid permutation");
    Tensor out = clone();
    std::vector<std::size_t> new_shape(shape_.size());
    std::vector<std::ptrdiff_t> new_strides(strides_.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        new_shape[i]   = shape_[dims[i]];
        new_strides[i] = strides_[dims[i]];
    }
    out.shape_   = new_shape;
    out.strides_ = new_strides;
    if (requires_grad_flag_) {
        // compute inverse permutation
        std::vector<int> inv(dims.size());
        for (size_t i = 0; i < dims.size(); ++i)
            inv[dims[i]] = int(i);
        out.requires_grad_flag_ = true;
        out.set_grad_fn(
            std::make_shared<PermuteBackward>(
                const_cast<Tensor*>(this),
                &out,
                std::move(inv)
            )
        );
    }
    return out;
}

// -- transpose via permute (with backward tracking) --

Tensor Tensor::transpose(int dim0, int dim1) const {
    if (dim0 < 0 || dim1 < 0 ||
        dim0 >= int(shape_.size()) || dim1 >= int(shape_.size())) {
        throw std::runtime_error("transpose: invalid dimensions");
    }
    std::vector<int> perm(shape_.size());
    std::iota(perm.begin(), perm.end(), 0);
    std::swap(perm[dim0], perm[dim1]);
    Tensor out = permute(perm);
    // permute already attached PermuteBackward, no extra attach needed
    return out;
}

// -- squeeze with backward tracking --

Tensor Tensor::squeeze(int dim) const {
    if (dim < 0 || dim >= int(shape_.size()) || shape_[dim] != 1) {
        return clone();
    }
    Tensor out = clone();
    out.shape_.erase(out.shape_.begin() + dim);
    out.compute_strides();
    if (requires_grad_flag_) {
        out.requires_grad_flag_ = true;
        out.set_grad_fn(
            std::make_shared<SqueezeBackward>(
                const_cast<Tensor*>(this),
                &out,
                dim
            )
        );
    }
    return out;
}

// -- unsqueeze with backward tracking --

Tensor Tensor::unsqueeze(int dim) const {
    if (dim < 0 || dim > int(shape_.size())) {
        throw std::runtime_error("unsqueeze: invalid dimension");
    }
    Tensor out = clone();
    out.shape_.insert(out.shape_.begin() + dim, 1);
    out.compute_strides();
    if (requires_grad_flag_) {
        out.requires_grad_flag_ = true;
        out.set_grad_fn(
            std::make_shared<UnsqueezeBackward>(
                const_cast<Tensor*>(this),
                &out,
                dim
            )
        );
    }
    return out;
}

Tensor Tensor::contiguous() const {
    if (is_contiguous())
        return clone();
    return clone();  // placeholder
}

// ===================== Initialisateurs =====================

Tensor Tensor::zeros(const std::vector<std::size_t>& shape,
                     DType dtype,
                     Device device) {
    Tensor out(shape, dtype, device);
    std::memset(out.storage_.get(), 0,
                out.numel() * dtype_size(dtype));
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
    size_t N = numel();
    for (size_t i = 0; i < N; ++i) c[i] = a[i] + b[i];
    if (requires_grad_flag_ || rhs.requires_grad_flag_) {
        out.requires_grad_flag_ = true;
        out.set_grad_fn(
            std::make_shared<AddBackward>(
                const_cast<Tensor*>(this),
                const_cast<Tensor*>(&rhs),
                &out
            )
        );
    }
    return out;
}

Tensor Tensor::operator-(const Tensor& rhs) const {
    check_device_consistency(rhs);
    check_shape_broadcast(rhs);
    Tensor out(shape_, dtype_, device_);
    float* a = static_cast<float*>(storage_.get());
    float* b = static_cast<float*>(rhs.storage_.get());
    float* c = static_cast<float*>(out.storage_.get());
    size_t N = numel();
    for (size_t i = 0; i < N; ++i) c[i] = a[i] - b[i];
    if (requires_grad_flag_ || rhs.requires_grad_flag_) {
        out.requires_grad_flag_ = true;
        out.set_grad_fn(
            std::make_shared<SubBackward>(
                const_cast<Tensor*>(this),
                const_cast<Tensor*>(&rhs),
                &out
            )
        );
    }
    return out;
}

Tensor Tensor::operator*(const Tensor& rhs) const {
    check_device_consistency(rhs);
    check_shape_broadcast(rhs);
    Tensor out(shape_, dtype_, device_);
    float* a = static_cast<float*>(storage_.get());
    float* b = static_cast<float*>(rhs.storage_.get());
    float* c = static_cast<float*>(out.storage_.get());
    size_t N = numel();
    for (size_t i = 0; i < N; ++i) c[i] = a[i] * b[i];
    if (requires_grad_flag_ || rhs.requires_grad_flag_) {
        out.requires_grad_flag_ = true;
        out.set_grad_fn(
            std::make_shared<MulBackward>(
                const_cast<Tensor*>(this),
                const_cast<Tensor*>(&rhs),
                &out
            )
        );
    }
    return out;
}

Tensor Tensor::operator/(const Tensor& rhs) const {
    check_device_consistency(rhs);
    check_shape_broadcast(rhs);
    Tensor out(shape_, dtype_, device_);
    float* a = static_cast<float*>(storage_.get());
    float* b = static_cast<float*>(rhs.storage_.get());
    float* c = static_cast<float*>(out.storage_.get());
    size_t N = numel();
    for (size_t i = 0; i < N; ++i) c[i] = a[i] / b[i];
    if (requires_grad_flag_ || rhs.requires_grad_flag_) {
        out.requires_grad_flag_ = true;
        out.set_grad_fn(
            std::make_shared<DivBackward>(
                const_cast<Tensor*>(this),
                const_cast<Tensor*>(&rhs),
                &out
            )
        );
    }
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
    if (requires_grad_flag_ || rhs.requires_grad_flag_) {
        out.requires_grad_flag_ = true;
        out.set_grad_fn(
            std::make_shared<MatMulBackward>(
                const_cast<Tensor*>(this),
                const_cast<Tensor*>(&rhs),
                &out
            )
        );
    }
    return out;
}

// ===================== Affichage =====================

void Tensor::print_summary() const {
    std::cout << "Tensor(";
    std::cout << "shape=[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        std::cout << shape_[i];
        if (i + 1 < shape_.size()) std::cout << ", ";
    }
    std::cout << "], dtype=";
    switch (dtype_) {
        case DType::Float32: std::cout << "float32"; break;
        case DType::Int32:   std::cout << "int32";   break;
    }
    std::cout << ", device=";
    switch (device_.type) {
        case DeviceType::CPU:  std::cout << "cpu";  break;
        case DeviceType::CUDA: std::cout << "cuda"; break;
    }
    std::cout << ")" << std::endl;
}

void Tensor::print_shape() const {
    std::cout << "[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        std::cout << shape_[i];
        if (i + 1 < shape_.size()) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

// ===================== Autograd access & backward =====================

Tensor& Tensor::grad() {
    if (!grad_ptr_) {
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
            "Cannot call backward() on a tensor that does not require grad"
        );
    }
    if (!grad_ptr_) {
        grad_ptr_.reset(new Tensor(Tensor::ones(shape_, dtype_, device_)));
    }
    std::vector<std::shared_ptr<GradFn>> stack;
    std::unordered_set<GradFn*> visited;
    if (grad_fn_) {
        stack.push_back(grad_fn_);
        visited.insert(grad_fn_.get());
    }
    while (!stack.empty()) {
        auto fn = stack.back(); stack.pop_back();
        fn->backward();
        for (Tensor* inp : fn->prev()) {
            if (inp->grad_fn_) {
                auto prev_fn = inp->grad_fn_;
                if (visited.insert(prev_fn.get()).second) {
                    stack.push_back(prev_fn);
                }
            }
        }
    }
}

// ===================== Data access =====================

template<typename T>
T* Tensor::data() {
    return static_cast<T*>(storage_.get());
}

template<typename T>
const T* Tensor::data() const {
    return static_cast<const T*>(storage_.get());
}

template float*       Tensor::data<float>();
template const float* Tensor::data<float>() const;

// Instantiate template constructor
template Tensor::Tensor(const std::vector<std::size_t>&,
                        const std::vector<float>&,
                        DType, Device);

// ===================== Copy & assignment =====================

Tensor::Tensor(const Tensor& other)
  : shape_(other.shape_),
    strides_(other.strides_),
    dtype_(other.dtype_),
    device_(other.device_),
    storage_(nullptr, default_deleter),
    requires_grad_flag_(other.requires_grad_flag_)
{
    size_t bytes = numel() * dtype_size(dtype_);
    void* raw = device_malloc(bytes, device_);
    std::memcpy(raw, other.storage_.get(), bytes);
    storage_.reset(raw);

    if (other.grad_ptr_) {
        grad_ptr_ = std::make_shared<Tensor>(*other.grad_ptr_);
    }
    grad_fn_ = other.grad_fn_;
}

Tensor& Tensor::operator=(const Tensor& other) {
    if (this == &other) return *this;
    shape_              = other.shape_;
    strides_            = other.strides_;
    dtype_              = other.dtype_;
    device_             = other.device_;
    requires_grad_flag_ = other.requires_grad_flag_;

    size_t bytes = numel() * dtype_size(dtype_);
    void* raw = device_malloc(bytes, device_);
    std::memcpy(raw, other.storage_.get(), bytes);
    storage_.reset(raw);

    if (other.grad_ptr_) {
        grad_ptr_ = std::make_shared<Tensor>(*other.grad_ptr_);
    } else {
        grad_ptr_.reset();
    }
    grad_fn_ = other.grad_fn_;
    return *this;
}

} // namespace napcas

