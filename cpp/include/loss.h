#ifndef LOSS_H
#define LOSS_H

#include "tensor.h"

/// @brief Mean Squared Error (MSE) loss function.
class MSELoss {
public:
    MSELoss() = default;
    /// @brief Computes MSE loss.
    /// @param y_pred Predicted values.
    /// @param y_true True values.
    /// @return Loss value.
    float forward(Tensor& y_pred, Tensor& y_true);
    /// @brief Computes gradient of MSE loss.
    /// @param y_pred Predicted values.
    /// @param y_true True values.
    /// @return Gradient tensor.
    Tensor backward(Tensor& y_pred, Tensor& y_true);
};

/// @brief Cross Entropy loss function.
class CrossEntropyLoss {
public:
    CrossEntropyLoss() = default;
    /// @brief Computes Cross Entropy loss.
    /// @param y_pred Predicted logits.
    /// @param y_true True labels (one-hot or indices).
    /// @return Loss value.
    float forward(Tensor& y_pred, Tensor& y_true);
    /// @brief Computes gradient of Cross Entropy loss.
    /// @param y_pred Predicted logits.
    /// @param y_true True labels.
    /// @return Gradient tensor.
    Tensor backward(Tensor& y_pred, Tensor& y_true);
};

#endif // LOSS_H
