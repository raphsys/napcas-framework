#include "loss.h"
#include <cmath>
#include <stdexcept>
#include <numeric>

// MSELoss

float MSELoss::forward(Tensor& y_pred, Tensor& y_true) {
    if (y_pred.shape() != y_true.shape()) {
        throw std::invalid_argument("Prediction and target shapes must match");
    }
    float loss = 0.0f;
    for (int i = 0; i < y_pred.size(); ++i) {
        float diff = y_pred[i] - y_true[i];
        loss += diff * diff;
    }
    return loss / y_pred.size();
}

Tensor MSELoss::backward(Tensor& y_pred, Tensor& y_true) {
    if (y_pred.shape() != y_true.shape()) {
        throw std::invalid_argument("Prediction and target shapes must match");
    }
    Tensor grad(y_pred.shape());
    for (int i = 0; i < y_pred.size(); ++i) {
        grad[i] = 2.0f * (y_pred[i] - y_true[i]) / y_pred.size();
    }
    return grad;
}

// CrossEntropyLoss

float CrossEntropyLoss::forward(Tensor& y_pred, Tensor& y_true) {
    if (y_pred.shape() != y_true.shape()) {
        throw std::invalid_argument("Prediction and target shapes must match");
    }
    float loss = 0.0f;
    float sum_exp = std::accumulate(y_pred.data().begin(), y_pred.data().end(), 0.0f,
        [](float sum, float x) { return sum + std::exp(x); });
    for (int i = 0; i < y_pred.size(); ++i) {
        float softmax = std::exp(y_pred[i]) / sum_exp;
        if (y_true[i] > 0.5f) {
            loss -= std::log(std::max(softmax, 1e-7f));
        }
    }
    return loss;
}

Tensor CrossEntropyLoss::backward(Tensor& y_pred, Tensor& y_true) {
    if (y_pred.shape() != y_true.shape()) {
        throw std::invalid_argument("Prediction and target shapes must match");
    }
    Tensor grad(y_pred.shape());
    float sum_exp = std::accumulate(y_pred.data().begin(), y_pred.data().end(), 0.0f,
        [](float sum, float x) { return sum + std::exp(x); });
    for (int i = 0; i < y_pred.size(); ++i) {
        float softmax = std::exp(y_pred[i]) / sum_exp;
        grad[i] = (y_true[i] > 0.5f) ? (softmax - 1.0f) : softmax;
    }
    return grad;
}
