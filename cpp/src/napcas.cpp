#include "napcas.h"
#include <Eigen/Dense>
#include <stdexcept>
#include <vector>

// =============================================================================
//  Constructeur
// =============================================================================
NAPCAS::NAPCAS(int input_size, int output_size)
    : input_size_(input_size),
      output_size_(output_size),
      weights_({output_size_, input_size_}),
      alpha_  ({output_size_}),
      grad_weights_({output_size_, input_size_}),
      grad_alpha_  ({output_size_}) {

    weights_.fill(0.01f);
    alpha_.fill(1.0f);
}

// -----------------------------------------------------------------------------
//  Masque : α ⊙ W
// -----------------------------------------------------------------------------
void NAPCAS::compute_mask() {
    masked_weights_ = Tensor(weights_.shape());
    for (int i = 0; i < output_size_; ++i)
        for (int j = 0; j < input_size_; ++j)
            masked_weights_[i * input_size_ + j] =
                weights_[i * input_size_ + j] * (alpha_[i] > 0.5f ? 1.f : 0.f);
}

// =============================================================================
//  Forward (B × in) → (B × out)
// =============================================================================
void NAPCAS::forward(Tensor &input, Tensor &output)
{
    const auto in_shape = input.shape();
    if (in_shape.size() != 2 || in_shape[1] != input_size_)
        throw std::invalid_argument("Input shape must be {B, input_size}");

    const int B = in_shape[0];

    // -- Assurer la bonne taille du tenseur de sortie -------------------------
    const auto out_shape = output.shape();
    const int expected_elems = B * output_size_;

    if (out_shape != std::vector<int>{B, output_size_}) {
        // Cas 1 : tenseur plat de la bonne taille
        if (out_shape == std::vector<int>{expected_elems}) {
            output.reshape({B, output_size_});
        }
        // Cas 2 : tenseur vide → on l’alloue
        else if (output.size() == 0) {
            output = Tensor({B, output_size_});
        }
        // Sinon : erreur
        else {
            throw std::invalid_argument("Output shape must be {B, output_size}");
        }
    }

    compute_mask();

    Eigen::Map<Eigen::MatrixXf> X (input.data().data(),      B, input_size_);
    Eigen::Map<Eigen::MatrixXf> W (masked_weights_.data().data(),
                                   output_size_, input_size_);
    Eigen::Map<Eigen::MatrixXf> Y (output.data().data(),     B, output_size_);

    Y = X * W.transpose();
    input_ = input;          // cache pour backward
}

// =============================================================================
//  Backward
// =============================================================================
void NAPCAS::backward(Tensor &grad_output, Tensor &grad_input)
{
    const auto gout_shape = grad_output.shape();
    if (gout_shape.size() != 2 || gout_shape[1] != output_size_)
        throw std::invalid_argument("grad_output shape mismatch");

    const int B = gout_shape[0];

    // Redimensionner grad_input si nécessaire
    if (grad_input.shape() != std::vector<int>{B, input_size_}) {
        if (grad_input.size() == B * input_size_)
            grad_input.reshape({B, input_size_});
        else if (grad_input.size() == 0)
            grad_input = Tensor({B, input_size_});
        else
            throw std::invalid_argument("grad_input shape mismatch");
    }

    compute_mask();

    Eigen::Map<Eigen::MatrixXf> dY (grad_output.data().data(), B, output_size_);
    Eigen::Map<Eigen::MatrixXf> dX (grad_input.data().data(),  B, input_size_);
    Eigen::Map<Eigen::MatrixXf> W  (masked_weights_.data().data(),
                                    output_size_, input_size_);
    Eigen::Map<Eigen::MatrixXf> X  (input_.data().data(),      B, input_size_);
    Eigen::Map<Eigen::MatrixXf> gW (grad_weights_.data().data(),
                                    output_size_, input_size_);

    dX = dY * W;
    gW = dY.transpose() * X / static_cast<float>(B);

    // dL/dα
    for (int i = 0; i < output_size_; ++i) {
        float sum = 0.f;
        for (int b = 0; b < B; ++b)
            for (int j = 0; j < input_size_; ++j)
                sum += dY(b, i) * weights_[i * input_size_ + j];
        grad_alpha_[i] = sum / static_cast<float>(B);
    }
}

// =============================================================================
//  SGD update simple
// =============================================================================
void NAPCAS::update(float lr)
{
    for (size_t k = 0; k < weights_.size(); ++k)
        weights_[k] -= lr * grad_weights_[k];

    for (size_t k = 0; k < alpha_.size(); ++k)
        alpha_[k] -= lr * grad_alpha_[k];
}

// =============================================================================
//  Utilitaires (inchangés)
// =============================================================================
void NAPCAS::set_weights(const Tensor &weights)
{
    if (weights.shape() != std::vector<int>{output_size_, input_size_})
        throw std::invalid_argument("Weight shape mismatch");
    weights_ = weights;
}

void NAPCAS::save(const std::string &path)
{
    weights_.save(path + "_weights.tensor");
    alpha_.save  (path + "_alpha.tensor");
}

void NAPCAS::load(const std::string &path)
{
    weights_.load(path + "_weights.tensor");
    alpha_.load  (path + "_alpha.tensor");
}

