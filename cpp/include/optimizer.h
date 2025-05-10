#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <vector>
#include <memory>
#include "module.h"
#include "tensor.h"

class SGD {
public:
    SGD(float lr = 0.01f);
    SGD(const std::vector<std::shared_ptr<Module>>& modules, float lr = 0.01f);
    void step();

private:
    float learning_rate_;
    std::vector<std::shared_ptr<Module>> modules_;
};

class Adam {
public:
    Adam(float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f);
    Adam(const std::vector<std::shared_ptr<Module>>& modules, float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f);
    void step();

private:
    float learning_rate_;
    float beta1_, beta2_, epsilon_;
    int t_;
    std::vector<std::shared_ptr<Module>> modules_;
    std::vector<Tensor> m_; // Premier moment (moyenne)
    std::vector<Tensor> v_; // Second moment (variance)
};

#endif // OPTIMIZER_H
