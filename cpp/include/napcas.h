#ifndef NAPCAS_H
#define NAPCAS_H

#include "tensor.h"
#include <string>
#include <vector>

/**
 * NAPCAS : couche linéaire masquée avec biais « alpha ».
 * - input_size  : nombre de caractéristiques d’entrée
 * - output_size : nombre de neurones de sortie
 *
 * La méthode forward ci-dessous accepte désormais des mini-batchs :
 *   input   [B, input_size ]
 *   output  [B, output_size]
 */
class NAPCAS {
public:
    NAPCAS(int input_size, int output_size);

    void forward (Tensor& input, Tensor& output);          // B × N ➜ B × M
    void backward(Tensor& grad_output, Tensor& grad_input);
    void update  (float lr);

    void set_weights(const Tensor& weights);
    void save(const std::string& path);
    void load(const std::string& path);

protected:
    int input_size_;
    int output_size_;

    Tensor weights_;        // [M, N]
    Tensor alpha_;          // [M]
    Tensor grad_weights_;   // [M, N]
    Tensor grad_alpha_;     // [M]
    Tensor masked_weights_; // [M, N]
    Tensor input_;          // cache pour backward (1 × N)

    void compute_mask();

private:
    /** Ancienne implémentation « batch 1 » — ré-utilisée en interne */
    void forward_single(Tensor& input_row, Tensor& output_row);
};

#endif /* NAPCAS_H */

