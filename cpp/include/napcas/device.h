#pragma once

#include "napcas/common.h"
#include <cstdlib>
#include <stdexcept>

namespace napcas {

inline void* device_malloc(std::size_t bytes, const Device& device) {
    if (device.type == DeviceType::CPU)
        return std::malloc(bytes);
#ifdef USE_CUDA
    else if (device.type == DeviceType::CUDA) {
        void* ptr = nullptr;
        cudaMalloc(&ptr, bytes);
        return ptr;
    }
#endif
    throw std::runtime_error("device_malloc: unsupported device");
}

inline void device_free(void* ptr, const Device& device) {
    if (!ptr) return;
    if (device.type == DeviceType::CPU)
        std::free(ptr);
#ifdef USE_CUDA
    else if (device.type == DeviceType::CUDA)
        cudaFree(ptr);
#endif
    else
        throw std::runtime_error("device_free: unsupported device");
}

} // namespace napcas

