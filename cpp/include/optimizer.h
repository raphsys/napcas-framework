#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <memory>
#include <vector>
#include "module.h"
#include "tensor.h"

class SGD {
public:
    SGD(float lr = 0.01f);
    SGD(const std::vector<std::shared_ptr<Module>>& modules, float lr = 0.01f);
    void step();

private:
    std::vector<std::shared_ptr<Module>> modules_;
    float lr_;
};

class Adam {
public:
    Adam(float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f);
    Adam(const std::vector<std::shared_ptr<Module>>& modules, float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f);
    void step();

private:
    std::vector<std::shared_ptr<Module>> modules_;
    float lr_;
    float beta1_;
    float beta2_;
    float epsilon_;
    std::vector<Tensor> m_; // Premier moment (moyenne)
    std::vector<Tensor> v_; // Second moment (variance)
    int t_; // Compteur d'itérations
};

#endif // OPTIMIZER_H
