#include "loss.h"
#include <cmath>

float MSELoss::forward(Tensor& y_pred, Tensor& y_true) {
    float loss = 0.0f;
    for (int i = 0; i < y_pred.size(); ++i) {
        loss += std::pow(y_pred[i] - y_true[i], 2);
    }
    return loss / y_pred.size();
}

Tensor MSELoss::backward(Tensor& y_pred, Tensor& y_true) {
    Tensor grad(y_pred.shape());
    for (int i = 0; i < y_pred.size(); ++i) {
        grad[i] = 2.0f * (y_pred[i] - y_true[i]) / y_pred.size();
    }
    return grad;
}

float CrossEntropyLoss::forward(Tensor& y_pred, Tensor& y_true) {
    // À implémenter
    return 0.0f;
}

Tensor CrossEntropyLoss::backward(Tensor& y_pred, Tensor& y_true) {
    // À implémenter
    return Tensor(y_pred.shape());
}
