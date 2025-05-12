#include "data_loader.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <hdf5.h>
#include <random>

DataLoader::DataLoader(const std::string& dataset_path, int batch_size, bool augment)
    : batch_size_(batch_size), current_index_(0), augment_(augment) {
    if (dataset_path.ends_with(".csv")) {
        load_csv(dataset_path);
    } else if (dataset_path.ends_with(".h5")) {
        load_hdf5(dataset_path);
    } else if (dataset_path.find("mnist") != std::string::npos) {
        load_mnist(dataset_path);
    } else if (dataset_path.find("cifar10") != std::string::npos) {
        load_cifar10(dataset_path);
    } else {
        throw std::runtime_error("Unsupported dataset format: " + dataset_path);
    }
}

std::pair<Tensor, Tensor> DataLoader::next() {
    if (current_index_ >= static_cast<int>(inputs_.size())) {
        current_index_ = 0; // Reset for looping
    }

    int batch_end = std::min(current_index_ + batch_size_, static_cast<int>(inputs_.size()));
    std::vector<Tensor> batch_inputs(inputs_.begin() + current_index_, inputs_.begin() + batch_end);
    std::vector<Tensor> batch_targets(targets_.begin() + current_index_, targets_.begin() + batch_end);

    if (augment_) {
        for (auto& input : batch_inputs) {
            augment_data(input);
        }
    }

    current_index_ = batch_end;
    return {batch_inputs[0], batch_targets[0]}; // Simplified; in practice, concatenate tensors
}

void DataLoader::load_csv(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    std::string line;
    std::getline(file, line); // Skip header
    while (std::getline(file, line)) {
        std::vector<float> row;
        std::stringstream ss(line);
        std::string value;
        while (std::getline(ss, value, ',')) {
            row.push_back(std::stof(value));
        }
        data_.push_back(row);
        inputs_.push_back(Tensor({static_cast<int>(row.size()) - 1}, std::vector<float>(row.begin(), row.end() - 1)));
        targets_.push_back(Tensor({1}, {row.back()}));
    }
    file.close();
}

void DataLoader::load_hdf5(const std::string& filename) {
    // Implement HDF5 loading using HDF5 C++ API
    // Example: Load datasets 'data' and 'labels'
}

void DataLoader::load_mnist(const std::string& directory) {
    // Implement MNIST loading (e.g., from binary files)
    // Example: Load images and labels
}

void DataLoader::load_cifar10(const std::string& directory) {
    // Implement CIFAR-10 loading (e.g., from binary files)
    // Example: Load images and labels
}

void DataLoader::augment_data(Tensor& input) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> noise(0.0f, 0.01f);
    for (int i = 0; i < input.size(); ++i) {
        input[i] += noise(gen); // Add Gaussian noise
    }
}
