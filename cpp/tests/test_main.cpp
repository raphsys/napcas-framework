#include "napcas.h"
#include "napca_sim.h"
#include "nncell.h"
#include "linear.h"
#include "conv2d.h"
#include "pooling.h"
#include "attention.h"
#include "activation.h"
#include "loss.h"
#include "optimizer.h"
#include "data_loader.h"
#include "mlp.h"
#include "rnn.h"
#include "transformer.h"
#include "gan.h"
#include "lstm.h"
#include "gru.h"
#include <cassert>
#include <iostream>
#include <chrono>

void test_napcas() {
    NAPCAS model(10, 5);
    Tensor input({2, 10}, std::vector<float>(20, 1.0f));
    Tensor output({2, 5});
    model.forward(input, output);
    assert(output.shape() == std::vector<int>{2, 5});
    std::cout << "NAPCAS test passed\n";
}

void test_napca_sim() {
    NAPCA_Sim model(10, 5, 0.6f, 0.5f);
    Tensor input({2, 10}, std::vector<float>(20, 1.0f));
    Tensor output({2, 5});
    model.forward(input, output);
    model.prune_connections(0.01f);
    assert(output.shape() == std::vector<int>{2, 5});
    std::cout << "NAPCA_Sim test passed\n";
}

void test_nncell() {
    NNCel model(10, 5);
    Tensor input({2, 10}, std::vector<float>(20, 1.0f));
    Tensor output({2, 5});
    model.forward(input, output);
    assert(output.shape() == std::vector<int>{2, 5});
    std::cout << "NNCel test passed\n";
}

void test_mlp() {
    MLP model({10, 20, 5}, "relu");
    Tensor input({2, 10}, std::vector<float>(20, 1.0f));
    Tensor output({2, 5});
    model.forward(input, output);
    assert(output.shape() == std::vector<int>{2, 5});
    std::cout << "MLP test passed\n";
}

void test_rnn() {
    RNN model(10, 20, 2);
    Tensor input({5, 2, 10}, std::vector<float>(100, 1.0f));
    Tensor output({5, 2, 20});
    model.forward(input, output);
    assert(output.shape() == std::vector<int>{5, 2, 20});
    std::cout << "RNN test passed\n";
}

void test_transformer() {
    Transformer model(64, 8, 2, 128);
    Tensor input({512, 2, 64}, std::vector<float>(512 * 2 * 64, 1.0f)); // Large sequence
    Tensor output({512, 2, 64});
    auto start = std::chrono::high_resolution_clock::now();
    model.forward(input, output);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    assert(output.shape() == std::vector<int>{512, 2, 64});
    std::cout << "Transformer test passed, time: " << duration.count() << "ms\n";
}

void test_gan() {
    GAN model({100, 256, 784}, {784, 256, 1});
    Tensor noise({64, 100}, std::vector<float>(6400, 1.0f));
    Tensor output({64, 784});
    model.forward(noise, output);
    assert(output.shape() == std::vector<int>{64, 784});
    std::cout << "GAN test passed\n";
}

void test_maxpool2d() {
    MaxPool2d model(2, 2);
    Tensor input({2, 3, 32, 32}, std::vector<float>(6144, 1.0f));
    Tensor output({2, 3, 16, 16});
    model.forward(input, output);
    assert(output.shape() == std::vector<int>{2, 3, 16, 16});
    std::cout << "MaxPool2d test passed\n";
}

void test_lstm() {
    LSTM model(10, 20, 2);
    Tensor input({5, 2, 10}, std::vector<float>(100, 1.0f));
    Tensor output({5, 2, 20});
    model.forward(input, output);
    assert(output.shape() == std::vector<int>{5, 2, 20});
    std::cout << "LSTM test passed\n";
}

void test_gru() {
    GRU model(10, 20, 2);
    Tensor input({5, 2, 10}, std::vector<float>(100, 1.0f));
    Tensor output({5, 2, 20});
    model.forward(input, output);
    assert(output.shape() == std::vector<int>{5, 2, 20});
    std::cout << "GRU test passed\n";
}

void test_conv2d() {
    Conv2d model(3, 16, 3, 1, 1);
    Tensor input({2, 3, 64, 64}, std::vector<float>(2 * 3 * 64 * 64, 1.0f));
    Tensor output({2, 16, 64, 64});
    auto start = std::chrono::high_resolution_clock::now();
    model.forward(input, output);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    assert(output.shape() == std::vector<int>{2, 16, 64, 64});
    std::cout << "Conv2d test passed, time: " << duration.count() << "ms\n";
}

int main() {
    test_napcas();
    test_napca_sim();
    test_nncell();
    test_mlp();
    test_rnn();
    test_transformer();
    test_gan();
    test_maxpool2d();
    test_lstm();
    test_gru();
    test_conv2d();
    std::cout << "All tests passed!\n";
    return 0;
}
