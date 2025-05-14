// File: cpp/include/data_loader.h

#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <vector>
#include <string>
#include "tensor.h"
#include <H5Cpp.h>  // HDF5 C++ API

/// @brief DataLoader capable de charger des données depuis :
///        - un fichier HDF5 (.h5, .hdf5) contenant les datasets "features" et "labels"
///        - un fichier CSV
///        - un dossier d’images
class DataLoader {
public:
    /// @param dataset_path Chemin vers .h5/.hdf5, .csv ou dossier d’images
    /// @param batch_size   Taille des mini-batchs
    /// @param augment      Si true, ajoute un léger bruit gaussien aux features
    DataLoader(const std::string& dataset_path,
               int batch_size,
               bool augment = false);

    /// @brief Retourne le batch suivant (inputs, targets)
    std::pair<Tensor, Tensor> next();

private:
    std::vector<Tensor> inputs_;    ///< Tenseurs d’entrée
    std::vector<Tensor> targets_;   ///< Tenseurs de sortie

    int batch_size_;
    int current_index_;
    bool augment_;

    void load_hdf5(const std::string& filename);
    void load_csv(const std::string& filename);
    void load_image_folder(const std::string& directory);
    void augment_data(Tensor& input);
};

#endif // DATA_LOADER_H

