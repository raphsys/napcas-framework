#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "module.h"
#include "tensor.h"

class SGD {
public:
    SGD();
    SGD(const std::vector<Module*>& modules);
    void step(Module* module);
};

class Adam {
public:
    Adam();
    Adam(const std::vector<Module*>& modules);
    void step(Module* module);
};

#endif // OPTIMIZER_H
