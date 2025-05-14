#ifndef CONV2D_H
#define CONV2D_H

#include "module.h"
#include "tensor.h"
#include <Eigen/Dense>
#include <string>

/// @brief Couche de convolution 2D optimisée (im2col + GEMM).
class Conv2d : public Module {
public:
    /// @param in_channels  Nombre de canaux en entrée.
    /// @param out_channels Nombre de canaux en sortie.
    /// @param kernel_size  Taille du noyau (impair de préférence).
    /// @param stride       Pas de déplacement (défaut : 1).
    /// @param padding      Padding autour de l’entrée (défaut : 0).
    Conv2d(int in_channels,
           int out_channels,
           int kernel_size,
           int stride = 1,
           int padding = 0);

    void forward(Tensor& input, Tensor& output) override;
    void backward(Tensor& grad_output, Tensor& grad_input) override;
    void update(float lr) override;

    Tensor& get_weights() override { return weights_; }
    Tensor& get_grad_weights() override { return grad_weights_; }
    void set_weights(const Tensor& weights) override;

    void save(const std::string& path) override;
    void load(const std::string& path) override;

private:
    int in_channels_;
    int out_channels_;
    int kernel_size_;
    int stride_;
    int padding_;

    Tensor weights_;
    Tensor bias_;
    Tensor grad_weights_;
    Tensor grad_bias_;

    Tensor input_cache_;           ///< copie de l’entrée du dernier forward
    Eigen::MatrixXf col_cache_;    ///< résultat de im2col pour backward

    /// @brief Découpe l’entrée en colonnes (patch_size × (batch*out_H*out_W))
    /// @param input      Tenseur (batch, in_channels, H, W)
    /// @param col        Matrice de sortie
    /// @param out_height Hauteur de sortie
    /// @param out_width  Largeur de sortie
    void im2col(const Tensor& input,
                Eigen::MatrixXf& col,
                int out_height,
                int out_width);
};

#endif // CONV2D_H

