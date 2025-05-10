#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <vector>
#include <string>
#include "tensor.h"

class DataLoader {
public:
    DataLoader(const std::string& dataset_path, int batch_size);
    std::pair<Tensor, Tensor> next();

private:
    std::vector<std::vector<float>> data_; // Stocke les données brutes
    std::vector<Tensor> inputs_;           // Stocke les tenseurs d'entrée
    std::vector<Tensor> targets_;          // Stocke les tenseurs cible
    int batch_size_;
    int current_index_;
};

#endif // DATA_LOADER_H
