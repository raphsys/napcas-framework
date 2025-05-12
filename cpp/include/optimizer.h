#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <vector>
#include <memory>
#include "module.h"
#include "tensor.h"

/// @brief Stochastic Gradient Descent (SGD) optimizer.
class SGD {
public:
    /// @brief Constructs an SGD optimizer.
    /// @param lr Learning rate.
    SGD(float lr = 0.01f);
    /// @brief Constructs an SGD optimizer with modules.
    /// @param modules List of modules to optimize.
    /// @param lr Learning rate.
    SGD(const std::vector<std::shared_ptr<Module>>& modules, float lr = 0.01f);
    /// @brief Performs an optimization step.
    void step();
    /// @brief Saves optimizer state to file.
    /// @param path File path.
    void save(const std::string& path);
    /// @brief Loads optimizer state from file.
    /// @param path File path.
    void load(const std::string& path);

private:
    float learning_rate_;
    std::vector<std::shared_ptr<Module>> modules_;
};

/// @brief Adam optimizer.
class Adam {
public:
    /// @brief Constructs an Adam optimizer.
    /// @param lr Learning rate.
    /// @param beta1 First moment decay rate.
    /// @param beta2 Second moment decay rate.
    /// @param epsilon Small value for numerical stability.
    Adam(float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f);
    /// @brief Constructs an Adam optimizer with modules.
    /// @param modules List of modules to optimize.
    /// @param lr Learning rate.
    /// @param beta1 First moment decay rate.
    /// @param beta2 Second moment decay rate.
    /// @param epsilon Small value for numerical stability.
    Adam(const std::vector<std::shared_ptr<Module>>& modules, float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f);
    /// @brief Performs an optimization step.
    void step();
    /// @brief Saves optimizer state to file.
    /// @param path File path.
    void save(const std::string& path);
    /// @brief Loads optimizer state from file.
    /// @param path File path.
    void load(const std::string& path);

private:
    float learning_rate_;
    float beta1_, beta2_, epsilon_;
    int t_;
    std::vector<std::shared_ptr<Module>> modules_;
    std::vector<Tensor> m_; // First moment
    std::vector<Tensor> v_; // Second moment
};

#endif // OPTIMIZER_H
