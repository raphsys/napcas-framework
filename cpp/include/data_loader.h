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
    std::vector<Tensor> inputs_;
    std::vector<Tensor> targets_;
    int batch_size_;
    int index_;
};

#endif // DATA_LOADER_H
