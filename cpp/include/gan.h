#ifndef GAN_H
#define GAN_H

#include "module.h"
#include "tensor.h"
#include "linear.h"
#include "activation.h"
#include <vector>
#include <memory>

class GAN : public Module {
public:
    GAN(const std::vector<int>& generator_layers, const std::vector<int>& discriminator_layers);
    void forward(Tensor& input, Tensor& output) override;
    void backward(Tensor& grad_output, Tensor& grad_input) override;
    void update(float lr) override;
    Tensor& get_weights() override { return weights_; }
    Tensor& get_grad_weights() override { return grad_weights_; }
    void set_weights(const Tensor& weights) override;
    void save(const std::string& path) override;
    void load(const std::string& path) override;
    void train_step(Tensor& real_data, Tensor& noise, float lr, Tensor& gen_loss, Tensor& disc_loss);

private:
    std::vector<std::shared_ptr<Module>> generator_;
    std::vector<std::shared_ptr<Module>> discriminator_;
    Tensor weights_;
    Tensor grad_weights_;
};

#endif
