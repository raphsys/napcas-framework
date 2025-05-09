NAPCAS Framework
NAPCAS is a PyTorch-like deep learning framework implemented in C++ with Python bindings. It supports neural network components such as Linear, Conv2d, activation functions (ReLU, Sigmoid, Tanh), loss functions (MSELoss, CrossEntropyLoss), optimizers (SGD, Adam), and a data loader.
Installation
Prerequisites

CMake (>= 3.14)
Python 3.10
pybind11
C++17 compatible compiler (e.g., g++)

Build Instructions

Clone the repository:git clone <repository_url>
cd napcas-framework


Create a build directory and compile the C++ module:mkdir cpp/build
cd cpp/build
cmake ..
make


Install the Python package:cd ../../python
pip install .



Usage
import napcas
from napcas import Linear, ReLU, MSELoss, SGD, DataLoader

# Create a simple model
model = [Linear(784, 128), ReLU(), Linear(128, 10)]
criterion = MSELoss()
optimizer = SGD(model)

# Load data
dataloader = DataLoader("dataset.csv", 64)

# Train
for epoch in range(5):
    for inputs, targets in dataloader.next():
        output = inputs
        for layer in model:
            temp = napcas.Tensor(inputs.shape())
            layer.forward(output, temp)
            output = temp
        loss = criterion.forward(output, targets)
        grad = criterion.backward(output, targets)
        grad_input = grad
        for layer in reversed(model):
            temp = napcas.Tensor(inputs.shape())
            layer.backward(grad_input, temp)
            grad_input = temp
        optimizer.step()

Directory Structure

cpp/: C++ source code and headers
python/: Python bindings and scripts
tests/: Test scripts
docs/: Documentation


