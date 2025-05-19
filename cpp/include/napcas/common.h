#pragma once

#include <string>
#include <stdexcept>
#include <sstream>
#include <iostream>

namespace napcas {

// === Types de données ===
enum class DType {
    Float32,
    Int32
};

inline std::string dtype_to_string(DType dtype) {
    switch (dtype) {
        case DType::Float32: return "float32";
        case DType::Int32: return "int32";
        default: return "unknown";
    }
}

inline std::size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::Float32: return sizeof(float);
        case DType::Int32: return sizeof(int);
        default: throw std::runtime_error("Unknown dtype");
    }
}

// === Types de devices ===
enum class DeviceType {
    CPU,
    CUDA
};

struct Device {
    DeviceType type;
    int index = 0;

    Device(DeviceType t = DeviceType::CPU, int idx = 0) : type(t), index(idx) {}

    std::string to_string() const {
        std::ostringstream oss;
        oss << (type == DeviceType::CPU ? "cpu" : "cuda") << ":" << index;
        return oss.str();
    }

    bool operator==(const Device& other) const {
        return type == other.type && index == other.index;
    }

    bool operator!=(const Device& other) const {
        return !(*this == other);
    }
};

// === Classe de base pour autograd (pré-déclarée ici) ===
class Function {
public:
    virtual ~Function() = default;
};

} // namespace napcas
