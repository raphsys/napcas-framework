#pragma once

#include <string>
#include <stdexcept>
#include <sstream>
#include <iostream>
#include <random>    // pour uniform_real

namespace napcas {

// === Types de données ===
enum class DType {
    Float32,
    Int32
};

inline std::string dtype_to_string(DType dtype) {
    switch (dtype) {
        case DType::Float32: return "float32";
        case DType::Int32:   return "int32";
        default:             return "unknown";
    }
}

inline std::size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::Float32: return sizeof(float);
        case DType::Int32:   return sizeof(int);
        default:             throw std::runtime_error("Unknown dtype");
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

    Device(DeviceType t = DeviceType::CPU, int idx = 0)
      : type(t), index(idx) {}

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

// === Génération uniforme ===
/// Renvoie un nombre uniforme dans [a,b]
template<typename T>
T uniform_real(T a, T b) {
    static thread_local std::mt19937_64 gen{std::random_device{}()};
    std::uniform_real_distribution<T> dist(a, b);
    return dist(gen);
}

// === Classe de base pour autograd ===
class Function {
public:
    virtual ~Function() = default;
};

} // namespace napcas

