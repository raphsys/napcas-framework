// File: cpp/src/data_loader.cpp
#include "data_loader.h"
#include <stdexcept>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <random>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace fs = std::filesystem;

DataLoader::DataLoader(const std::string& dataset_path,
                       int batch_size,
                       bool augment)
    : batch_size_(batch_size),
      current_index_(0),
      augment_(augment)
{
    // Détection du type de source
    if (dataset_path.size() >= 3 &&
        (dataset_path.substr(dataset_path.size()-3) == ".h5" ||
         dataset_path.substr(dataset_path.size()-5) == ".hdf5")) 
    {
        load_hdf5(dataset_path);
    }
    else if (dataset_path.size() >= 4 &&
             dataset_path.substr(dataset_path.size()-4) == ".csv")
    {
        load_csv(dataset_path);
    }
    else if (fs::is_directory(dataset_path))
    {
        load_image_folder(dataset_path);
    }
    else
    {
        throw std::runtime_error("Unsupported DataLoader path: " + dataset_path);
    }

    if (inputs_.empty()) {
        throw std::runtime_error("No data loaded by DataLoader");
    }
}

std::pair<Tensor, Tensor> DataLoader::next() {
    if (current_index_ >= static_cast<int>(inputs_.size())) {
        current_index_ = 0;  // restart
    }
    int end = std::min(current_index_ + batch_size_,
                       static_cast<int>(inputs_.size()));

    // Construction des shapes batched
    auto in0 = inputs_[current_index_];
    auto out0 = targets_[current_index_];
    std::vector<int> shape_in = in0.shape();
    std::vector<int> shape_out = out0.shape();
    shape_in.insert(shape_in.begin(), end - current_index_);
    shape_out.insert(shape_out.begin(), end - current_index_);

    // Concaténation des données
    std::vector<float> data_in, data_out;
    data_in.reserve((end - current_index_) * in0.size());
    data_out.reserve((end - current_index_) * out0.size());
    for (int i = current_index_; i < end; ++i) {
        auto& din  = inputs_[i].data();
        auto& dout = targets_[i].data();
        data_in.insert(data_in.end(), din.begin(), din.end());
        data_out.insert(data_out.end(), dout.begin(), dout.end());
    }
    current_index_ = end;

    Tensor batch_in(shape_in, data_in);
    Tensor batch_out(shape_out, data_out);
    if (augment_) augment_data(batch_in);
    return {batch_in, batch_out};
}

void DataLoader::load_hdf5(const std::string& filename) {
    try {
        H5::H5File file(filename, H5F_ACC_RDONLY);

        // Ouverture des datasets
        H5::DataSet ds_in  = file.openDataSet("features");
        H5::DataSet ds_out = file.openDataSet("labels");

        // Espaces de données
        H5::DataSpace sp_in  = ds_in.getSpace();
        H5::DataSpace sp_out = ds_out.getSpace();

        // Dimensions inputs
        int rank_in = sp_in.getSimpleExtentNdims();
        std::vector<hsize_t> dims_in(rank_in);
        sp_in.getSimpleExtentDims(dims_in.data());
        size_t N = dims_in[0];
        size_t sample_size_in = 1;
        for (int i = 1; i < rank_in; ++i) sample_size_in *= dims_in[i];

        // Lecture brute inputs
        std::vector<float> raw_in(N * sample_size_in);
        ds_in.read(raw_in.data(), H5::PredType::NATIVE_FLOAT);

        // Dimensions labels
        int rank_out = sp_out.getSimpleExtentNdims();
        std::vector<hsize_t> dims_out(rank_out);
        sp_out.getSimpleExtentDims(dims_out.data());
        size_t sample_size_out = 1;
        for (int i = 1; i < rank_out; ++i) sample_size_out *= dims_out[i];

        // Lecture brute labels
        std::vector<float> raw_out(N * sample_size_out);
        ds_out.read(raw_out.data(), H5::PredType::NATIVE_FLOAT);

        // Construction des tenseurs par échantillon
        std::vector<int> shape_in, shape_out;
        for (int i = 1; i < rank_in;  ++i) shape_in.push_back((int)dims_in[i]);
        if (rank_out > 1) {
            for (int i = 1; i < rank_out; ++i) shape_out.push_back((int)dims_out[i]);
        } else {
            shape_out.push_back(1);
        }

        for (size_t i = 0; i < N; ++i) {
            std::vector<float> slice_in(raw_in.begin()  + i*sample_size_in,
                                        raw_in.begin()  + (i+1)*sample_size_in);
            std::vector<float> slice_out(raw_out.begin() + i*sample_size_out,
                                         raw_out.begin() + (i+1)*sample_size_out);
            inputs_.emplace_back(Tensor(shape_in, slice_in));
            targets_.emplace_back(Tensor(shape_out, slice_out));
        }
    }
    catch (H5::Exception& err) {
        throw std::runtime_error("HDF5 error: " + std::string(err.getCDetailMsg()));
    }
}

void DataLoader::load_csv(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) throw std::runtime_error("Cannot open CSV: " + filename);
    std::string line;
    // Sauter éventuel header
    if (std::getline(file, line) && line.find(',') == std::string::npos) {
        file.clear();
        file.seekg(0);
    }
    while (std::getline(file, line)) {
        std::vector<float> vals;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            vals.push_back(std::stof(cell));
        }
        if (vals.size() < 2) continue;
        std::vector<float> feat(vals.begin(), vals.end()-1);
        float lbl = vals.back();
        inputs_.emplace_back(Tensor({(int)feat.size()}, feat));
        targets_.emplace_back(Tensor({1}, std::vector<float>{lbl}));
    }
}

void DataLoader::load_image_folder(const std::string& directory) {
    for (auto& entry : fs::directory_iterator(directory)) {
        if (!entry.is_regular_file()) continue;
        const auto& path = entry.path().string();
        int w,h,n;
        unsigned char* img = stbi_load(path.c_str(), &w,&h,&n,1);
        if (!img) continue;
        std::vector<float> pix(w*h);
        for (int i = 0; i < w*h; ++i) pix[i] = img[i]/255.0f;
        stbi_image_free(img);
        inputs_.emplace_back(Tensor({1,h,w}, pix));
        targets_.emplace_back(Tensor({1}, std::vector<float>{0.0f}));
    }
}

void DataLoader::augment_data(Tensor& input) {
    std::mt19937 gen(std::random_device{}());
    std::normal_distribution<float> dist(0.0f, 0.01f);
    for (int i = 0; i < input.size(); ++i) {
        input[i] += dist(gen);
    }
}

