#ifndef NAPCAS_H
#define NAPCAS_H

#include "tensor.h"
#include <string>
#include <vector>

class NAPCAS {
public:
    NAPCAS(int input_size, int output_size);
    virtual void forward(Tensor& input, Tensor& output);
    virtual void backward(Tensor& grad_output, Tensor& grad_input);
    virtual void update(float lr);
    void set_weights(const Tensor& weights);
    void save(const std::string& path);
    void load(const std::string& path);

protected:
    int input_size_;
    int output_size_;
    Tensor weights_;
    Tensor alpha_;
    Tensor grad_weights_;
    Tensor grad_alpha_;
    Tensor masked_weights_;
    Tensor input_; // Cached input for backward pass

    void compute_mask();
};

#endif // NAPCAS_H
