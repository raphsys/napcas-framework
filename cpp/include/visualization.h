#ifndef VISUALIZATION_H
#define VISUALIZATION_H

#include "tensor.h"
#include <string>

/// @brief Visualization utilities for neural network analysis.
class Visualization {
public:
    /// @brief Plots a tensor (e.g., weights or gradients) using Matplotlib.
    /// @param tensor Tensor to plot.
    /// @param title Plot title.
    /// @param output_path Output file path.
    static void plot_tensor(const Tensor& tensor, const std::string& title, const std::string& output_path);
    /// @brief Logs performance metrics to TensorBoard.
    /// @param metric_name Name of the metric.
    /// @param value Metric value.
    /// @param step Training step.
    static void log_to_tensorboard(const std::string& metric_name, float value, int step);
    /// @brief Plots training curves (loss, accuracy).
    /// @param losses List of loss values.
    /// @param accuracies List of accuracy values.
    /// @param output_path Output file path.
    static void plot_training_curves(const std::vector<float>& losses, const std::vector<float>& accuracies, const std::string& output_path);
};

#endif // VISUALIZATION_H
