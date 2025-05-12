#include "gpu_utils.h"
#include <stdexcept>

#ifdef USE_CUDA
float* GPUUtils::allocate_cuda_memory(int size) {
    float* ptr;
    cudaError_t err = cudaMalloc(&ptr, size * sizeof(float));
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA memory allocation failed");
    }
    return ptr;
}

void GPUUtils::free_cuda_memory(float* ptr) {
    cudaFree(ptr);
}

void GPUUtils::copy_to_cuda(const float* src, float* dst, int size) {
    cudaMemcpy(dst, src, size * sizeof(float), cudaMemcpyHostToDevice);
}

void GPUUtils::copy_from_cuda(const float* src, float* dst, int size) {
    cudaMemcpy(dst, src, size * sizeof(float), cudaMemcpyDeviceToHost);
}
#endif
