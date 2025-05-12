NAPCAS Framework
Neural Adaptive Processing and Connectivity Analysis System (NAPCAS) is a C++ and Python framework for building and training neural networks with adaptive connectivity and path similarity analysis.
Features

Core modules: NAPCAS, NAPCA_Sim, NNCel
Advanced models: MLP, CNN (Conv2d with stride/padding/pooling, optimized with im2col), RNN, LSTM, GRU, Transformer (memory-efficient attention), GAN
GPU support via CUDA (placeholders)
Python bindings via Pybind11
Visualization tools with Matplotlib, TensorBoard, and Plotly (interactive)
Comprehensive testing suite
Doxygen and Sphinx documentation

Installation
Prerequisites

CMake >= 3.10
Eigen3
CUDA (optional for GPU support)
Python >= 3.8
Pybind11
Python packages: numpy, matplotlib, h5py, tensorboard, plotly

Build Instructions
mkdir build && cd build
cmake ..
make
make install

Python Package
cd python
pip install .

Usage
Python Example
from napcas import NAPCAS, DataLoader, SGD, CrossEntropyLoss

loader = DataLoader("data/synthetic.csv", batch_size=2)
model = NAPCAS(10, 5)
optimizer = SGD([model], lr=0.01)
loss_fn = CrossEntropyLoss()

input, target = loader.next()
output = Tensor([2, 5])
model.forward(input, output)
loss = loss_fn.forward(output, target)
grad_output = loss_fn.backward(output, target)
grad_input = Tensor(input.shape())
model.backward(grad_output, grad_input)
optimizer.step()

Interactive Visualization
from napcas import plot_tensor_interactive

tensor = Tensor([10], [float(i) for i in range(10)])
plot_tensor_interactive(tensor, "Interactive Tensor Plot", "tensor_plot.html")

Performance Considerations

Conv2d: Uses im2col algorithm for efficient convolution via matrix multiplication, optimized for large inputs.
Transformer: Implements memory-efficient attention by processing queries in chunks (default: 512 tokens), reducing memory usage for large sequences.
Testing: Performance tests in test_main.cpp measure execution time for Conv2d and Transformer on large inputs.

Documentation

API documentation: docs/build/html/index.html
Run doxygen in docs/ to generate C++ API docs
Run sphinx-build -b html docs/source docs/build for Python docs

Testing

C++ tests: ./build/napcas_test
Python tests: pytest python/tests

License
MIT License

