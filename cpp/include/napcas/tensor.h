#pragma once

#include <vector>
#include <memory>
#include <cstddef>
#include <string>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <initializer_list>
#include "napcas/common.h"
#include "napcas/device.h"
#include "napcas/grad_fn.h"

namespace napcas {

// Forward‐declare pour éviter d’inclure grad_fn.h ici
class Function;

class Tensor {
public:
    // ----- Constructeurs -----
    Tensor();
    Tensor(const std::vector<std::size_t>& shape,
           DType dtype = DType::Float32,
           Device device = Device{DeviceType::CPU, 0});

    template<typename Scalar>
    Tensor(const std::vector<std::size_t>& shape,
           const std::vector<Scalar>& data,
           DType dtype = DType::Float32,
           Device device = Device{DeviceType::CPU, 0});

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&&) noexcept;
    Tensor& operator=(Tensor&&) noexcept;

    // ----- Accès aux métadonnées -----
    const std::vector<std::size_t>& shape()   const noexcept { return shape_; }
    const std::vector<std::ptrdiff_t>& strides() const noexcept { return strides_; }
    DType   dtype()   const noexcept { return dtype_; }
    Device  device()  const noexcept { return device_; }
    std::size_t ndim()   const noexcept { return shape_.size(); }
    std::size_t numel()  const noexcept;
    bool    is_contiguous() const noexcept;

    // ----- Autograd interface -----
    // Active ou désactive le calcul du gradient
    void    requires_grad_(bool flag) noexcept { requires_grad_flag_ = flag; }
    bool    requires_grad() const noexcept    { return requires_grad_flag_; }

    // Accès au gradient (initialisé à ones() si nécessaire)
    Tensor&       grad();
    const Tensor& grad() const;

    // Lance la rétropropagation
    void    backward();

    // ----- Accès aux données -----
    template<typename T>       T* data();
    template<typename T> const T* data() const;

    // ----- Transformations -----
    Tensor clone()   const;
    Tensor detach()  const;
    Tensor to(Device new_device) const;
    Tensor astype(DType new_dtype) const;
    Tensor reshape(const std::vector<std::size_t>& new_shape) const;
    Tensor view   (const std::vector<std::size_t>& new_shape) const;
    Tensor permute(const std::vector<int>& dims) const;
    Tensor transpose(int dim0, int dim1) const;
    Tensor squeeze(int dim = -1) const;
    Tensor unsqueeze(int dim);
    Tensor contiguous() const;

    // ----- Initialisateurs statiques -----
    static Tensor zeros(const std::vector<std::size_t>& shape,
                        DType dtype = DType::Float32,
                        Device device = Device{DeviceType::CPU, 0});
    static Tensor ones (const std::vector<std::size_t>& shape,
                        DType dtype = DType::Float32,
                        Device device = Device{DeviceType::CPU, 0});

    // ----- Opérations élémentaires -----
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;
    Tensor matmul(const Tensor& other) const;

    // ----- Debug / affichage -----
    void print_shape()  const;
    void print_summary() const;

private:
    std::vector<std::size_t>    shape_;
    std::vector<std::ptrdiff_t> strides_;
    DType     dtype_;
    Device    device_;
    std::unique_ptr<void, void(*)(void*)> storage_;

    // Autograd
    std::shared_ptr<Tensor> grad_ptr_;
    std::shared_ptr<GradFn> grad_fn_;
    bool requires_grad_flag_ = false;

    // Utilitaires internes
    void compute_strides();
    void check_device_consistency(const Tensor& other) const;
    void check_shape_broadcast   (const Tensor& other) const;
};

} // namespace napcas

