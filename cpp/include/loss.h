#ifndef LOSS_H
#define LOSS_H

#include "tensor.h"

class MSELoss {
public:
    MSELoss() = default; // Explicit declaration
    float forward(Tensor& y_pred, Tensor& y_true);
    Tensor backward(Tensor& y_pred, Tensor& y_true);
};

class CrossEntropyLoss {
public:
    CrossEntropyLoss() = default; // Explicit declaration
    float forward(Tensor& y_pred, Tensor& y_true);
    Tensor backward(Tensor& y_pred, Tensor& y_true);
};

#endif // LOSS_H
