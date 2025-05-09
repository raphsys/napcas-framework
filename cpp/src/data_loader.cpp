#include "data_loader.h"
#include <fstream>
#include <sstream>

DataLoader::DataLoader(const std::string& dataset_path, int batch_size)
    : batch_size_(batch_size), index_(0) {
    // Chargement des données depuis un fichier CSV ou autre format
}

std::pair<Tensor, Tensor> DataLoader::next() {
    // Exemple avec des shapes par défaut - adaptez selon vos besoins
    return {Tensor({1}), Tensor({1})};  // Crée des tensors 1D de taille 1
}
