#ifndef GPU_UTILS_H
#define GPU_UTILS_H

#include "tensor.h"

/// @brief GPU utilities for tensor operations.
class GPUUtils {
public:
    #ifdef USE_CUDA
    /// @brief Allocates memory on GPU.
    /// @param size Number of elements.
    /// @return Pointer to GPU memory.
    static float* allocate_cuda_memory(int size);
    /// @brief Frees GPU memory.
    /// @param ptr Pointer to GPU memory.
    static void free_cuda_memory(float* ptr);
    /// @brief Copies data from CPU to GPU.
    /// @param src Source data (CPU).
    /// @param dst Destination data (GPU).
    /// @param size Number of elements.
    static void copy_to_cuda(const float* src, float* dst, int size);
    /// @brief Copies data from GPU to CPU.
    /// @param src Source data (GPU).
    /// @param dst Destination data (CPU).
    /// @param size Number of elements.
    static void copy_from_cuda(const float* src, float* dst, int size);
    #endif
    #ifdef USE_OPENCL
    // OpenCL utilities (similar structure)
    #endif
};

#endif // GPU_UTILS_H
