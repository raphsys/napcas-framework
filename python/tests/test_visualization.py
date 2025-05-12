import pytest
from napcas import Tensor, plot_tensor, plot_training_curves
import os

def test_plot_tensor():
    tensor = Tensor([10], [float(i) for i in range(10)])
    output_path = "test_tensor_plot.png"
    plot_tensor(tensor, "Test Tensor", output_path)
    assert os.path.exists(output_path)
    os.remove(output_path)

def test_plot_training_curves():
    losses = [float(i) for i in range(10)]
    accuracies = [0.9 for _ in range(10)]
    output_path = "test_training_curves.png"
    plot_training_curves(losses, accuracies, output_path)
    assert os.path.exists(output_path)
    os.remove(output_path)
