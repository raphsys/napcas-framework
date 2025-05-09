#include "data_loader.h"
#include <fstream>
#include <sstream>
#include <stdexcept>

DataLoader::DataLoader(const std::string& filename, int batch_size)
    : batch_size_(batch_size), current_index_(0) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    std::string line;
    while (std::getline(file, line)) {
        std::vector<float> row;
        std::stringstream ss(line);
        std::string value;
        while (std::getline(ss, value, ',')) {
            row.push_back(std::stof(value));
        }
        data_.push_back(row);
    }
    file.close();
}

std::pair<Tensor, Tensor> DataLoader::next() {
    if (current_index_ >= data_.size()) {
        current_index_ = 0; // Reset pour boucler
    }

    int batch_end = std::min(current_index_ + batch_size_, static_cast<int>(data_.size()));
    std::vector<float> inputs;
    std::vector<float> targets;
    std::vector<int> input_shape = {batch_end - current_index_, static_cast<int>(data_[0].size()) - 1};
    std::vector<int> target_shape = {batch_end - current_index_, 1};

    for (int i = current_index_; i < batch_end; ++i) {
        for (size_t j = 0; j < data_[i].size() - 1; ++j) {
            inputs.push_back(data_[i][j]);
        }
        targets.push_back(data_[i].back());
    }

    current_index_ = batch_end;
    return {Tensor(input_shape, inputs), Tensor(target_shape, targets)};
}
