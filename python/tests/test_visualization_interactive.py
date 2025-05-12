import pytest
from napcas import Tensor, plot_tensor_interactive, plot_training_curves_interactive
import os

def test_plot_tensor_interactive():
    tensor = Tensor([10], [float(i) for i in range(10)])
    output_path = "test_tensor_interactive.html"
    plot_tensor_interactive(tensor, "Test Tensor", output_path)
    assert os.path.exists(output_path)
    os.remove(output_path)

def test_plot_training_curves_interactive():
    losses = [float(i) for i in range(10)]
    accuracies = [0.9 for _ in range(10)]
    output_path = "test_training_curves_interactive.html"
    plot_training_curves_interactive(losses, accuracies, output_path)
    assert os.path.exists(output_path)
    os.remove(output_path)
