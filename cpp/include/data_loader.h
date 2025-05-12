#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <vector>
#include <string>
#include "tensor.h"

/// @brief Data loader for loading and batching datasets.
class DataLoader {
public:
    /// @brief Constructs a DataLoader from a dataset path.
    /// @param dataset_path Path to dataset (CSV, HDF5, or directory for MNIST/CIFAR-10).
    /// @param batch_size Batch size.
    /// @param augment Whether to apply data augmentation.
    DataLoader(const std::string& dataset_path, int batch_size, bool augment = false);
    /// @brief Gets the next batch of data.
    /// @return Pair of input and target tensors.
    std::pair<Tensor, Tensor> next();

private:
    std::vector<std::vector<float>> data_; // Raw data
    std::vector<Tensor> inputs_;           // Input tensors
    std::vector<Tensor> targets_;          // Target tensors
    int batch_size_;
    int current_index_;
    bool augment_;
    void load_csv(const std::string& filename);
    void load_hdf5(const std::string& filename);
    void load_mnist(const std::string& directory);
    void load_cifar10(const std::string& directory);
    void augment_data(Tensor& input);
};

#endif // DATA_LOADER_H
