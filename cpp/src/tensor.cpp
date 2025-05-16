// tensor.cpp
#include "napcas/tensor.h"
#include <cstring>
#include <algorithm>
#include <queue>

namespace napcas {

// ------ Constructeurs ------

Tensor::Tensor(const std::vector<std::size_t>& shape,
               DType dtype,
               Device device)
    : shape_(shape), dtype_(dtype), device_(device) {
    if (shape_.empty())
        throw Error("Tensor: empty shape");
    std::size_t bytes = napcas::numel(shape_) * sizeof(float);
    storage_ = Storage(::operator new(bytes),
                       [](void* p){ ::operator delete(p); });
    strides_.resize(shape_.size());
    std::ptrdiff_t stride = 1;
    for (int i = int(shape_.size()) - 1; i >= 0; --i) {
        strides_[i] = stride;
        stride *= static_cast<std::ptrdiff_t>(shape_[i]);
    }
}

template<typename Scalar>
Tensor::Tensor(const std::vector<std::size_t>& shape,
               const std::vector<Scalar>& data,
               Device device)
    : Tensor(shape,
             std::is_same<Scalar,float>::value ? DType::Float32 : DType::Float64,
             device) {
    if (data.size() != napcas::numel(shape))
        throw Error("Tensor: data size != numel(shape)");
    std::memcpy(storage_.get(), data.data(), data.size() * sizeof(Scalar));
}

// ------ Transformations ------

Tensor Tensor::to(Device new_device) const {
    if (new_device == device_) return *this;
    Tensor out(*this);
    out.device_ = new_device;
    // TODO: GPU<->CPU copy
    return out;
}

Tensor Tensor::astype(DType new_type) const {
    if (new_type == dtype_) return *this;
    Tensor out(shape_, new_type, device_);
    if (dtype_ == DType::Float32 && new_type == DType::Float64) {
        const float* src = data<float>();
        double*       dst = out.data<double>();
        std::transform(src, src + numel(), dst,
                       [](float v){ return static_cast<double>(v); });
    } else if (dtype_ == DType::Float64 && new_type == DType::Float32) {
        const double* src = data<double>();
        float*        dst = out.data<float>();
        std::transform(src, src + numel(), dst,
                       [](double v){ return static_cast<float>(v); });
    } else {
        throw Error("astype: unsupported cast");
    }
    return out;
}

Tensor Tensor::reshape(const std::vector<std::size_t>& new_shape) const {
    if (napcas::numel(new_shape) != napcas::numel(shape_))
        throw Error("reshape: incompatible numel");
    Tensor out(*this);
    out.shape_ = new_shape;
    out.strides_.resize(new_shape.size());
    std::ptrdiff_t stride = 1;
    for (int i = int(new_shape.size()) - 1; i >= 0; --i) {
        out.strides_[i] = stride;
        stride *= static_cast<std::ptrdiff_t>(new_shape[i]);
    }
    return out;
}

Tensor Tensor::contiguous() const {
    return *this;
}

// ------ Opérations binaires + Autograd ------

template<typename Functor>
Tensor Tensor::binary_op(const Tensor& rhs,
                         Functor fn,
                         std::function<void(const Tensor&)> backward_fn) const {
    // Forward
    Tensor out(shape_, dtype_, device_);
    if (dtype_ == DType::Float32) {
        const auto* a = data<float>();
        const auto* b = rhs.data<float>();
              auto* c = out.data<float>();
        std::transform(a, a + numel(), b, c, fn);
    } else {
        const auto* a = data<double>();
        const auto* b = rhs.data<double>();
              auto* c = out.data<double>();
        std::transform(a, a + numel(), b, c, fn);
    }

    // Autograd
    if (requires_grad_flag_ || rhs.requires_grad_flag_) {
        out.requires_grad_(true);
        struct OpFn : Function {
            std::function<void(const Tensor&)> backward_fn_;
            OpFn(std::function<void(const Tensor&)> bf)
              : backward_fn_(std::move(bf)) {}
            void backward(const Tensor& grad_output) override {
                backward_fn_(grad_output);
            }
        };
        auto fn_ptr = std::make_shared<OpFn>(backward_fn);
        if (requires_grad_flag_)     fn_ptr->next_functions_.push_back(grad_fn_);
        if (rhs.requires_grad_flag_) fn_ptr->next_functions_.push_back(rhs.grad_fn_);
        out.grad_fn_ = fn_ptr;
    }
    return out;
}

Tensor Tensor::operator+(const Tensor& rhs) const {
    return binary_op(rhs, std::plus<>{}, [&](const Tensor& g){
        if (requires_grad_flag_)    *grad_ptr_ = *grad_ptr_ + g;
        if (rhs.requires_grad_flag_) *rhs.grad_ptr_ = *rhs.grad_ptr_ + g;
    });
}
Tensor Tensor::operator-(const Tensor& rhs) const {
    return binary_op(rhs, std::minus<>{}, [&](const Tensor& g){
        if (requires_grad_flag_)    *grad_ptr_ = *grad_ptr_ + g;
        if (rhs.requires_grad_flag_) *rhs.grad_ptr_ = *rhs.grad_ptr_ - g;
    });
}
Tensor Tensor::operator*(const Tensor& rhs) const {
    return binary_op(rhs, std::multiplies<>{}, [&](const Tensor& g){
        if (requires_grad_flag_)    *grad_ptr_ = *grad_ptr_ + (g * rhs);
        if (rhs.requires_grad_flag_) *rhs.grad_ptr_ = *rhs.grad_ptr_ + (g * *this);
    });
}
Tensor Tensor::operator/(const Tensor& rhs) const {
    return binary_op(rhs, std::divides<>{}, [&](const Tensor& g){
        if (requires_grad_flag_)    *grad_ptr_ = *grad_ptr_ + (g / rhs);
        if (rhs.requires_grad_flag_) *rhs.grad_ptr_ = *rhs.grad_ptr_ - (g * *this) / (rhs * rhs);
    });
}

// ------ Matmul + Autograd ------

Tensor Tensor::matmul(const Tensor& rhs) const {
    if (ndim()!=2 || rhs.ndim()!=2 || size(1)!=rhs.size(0))
        throw Error("matmul: shape mismatch");
    Tensor out({ size(0), rhs.size(1) }, dtype_, device_);
    if (dtype_ == DType::Float32) {
        Eigen::Map<const Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>>
            A(data<float>(), size(0), size(1));
        Eigen::Map<const Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>>
            B(rhs.data<float>(), rhs.size(0), rhs.size(1));
        Eigen::Map<Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>>
            C(out.data<float>(), out.size(0), out.size(1));
        C.noalias() = A * B;
    } else {
        Eigen::Map<const Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>>
            A(data<double>(), size(0), size(1));
        Eigen::Map<const Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>>
            B(rhs.data<double>(), rhs.size(0), rhs.size(1));
        Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>>
            C(out.data<double>(), out.size(0), out.size(1));
        C.noalias() = A * B;
    }
    return binary_op(rhs, std::plus<>{}, [&](const Tensor& g){
        if (requires_grad_flag_)    *grad_ptr_ = *grad_ptr_ + g.matmul(rhs);
        if (rhs.requires_grad_flag_) *rhs.grad_ptr_ = *rhs.grad_ptr_ + (*this).matmul(g);
    });
}

// ------ backward global ------

void Tensor::backward() {
    if (!requires_grad_flag_) requires_grad_(true);
    *grad_ptr_ = Tensor(shape_, dtype_, device_);
    if (numel()==1) (*grad_ptr_).data<float>()[0] = 1.0f;

    std::queue<std::shared_ptr<Function>> q;
    if (grad_fn_) q.push(grad_fn_);
    while (!q.empty()) {
        auto fn = q.front(); q.pop();
        fn->backward(*grad_ptr_);
        for (auto& nxt : fn->next_functions_)
            if (nxt) q.push(nxt);
    }
}

// Explicit instantiation
template Tensor::Tensor<float>(const std::vector<std::size_t>&, const std::vector<float>&, Device);
template Tensor::Tensor<double>(const std::vector<std::size_t>&, const std::vector<double>&, Device);

} // namespace napcas :contentReference[oaicite:10]{index=10}

