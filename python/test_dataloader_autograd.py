import pytest
import napcas
import os
import math

# Créer un fichier CSV temporaire pour tester DataLoader
@pytest.fixture
def temp_csv(tmp_path):
    data = "1.0,2.0,3.0\n4.0,5.0,6.0\n"
    file_path = tmp_path / "test.csv"
    with open(file_path, "w") as f:
        f.write(data)
    return str(file_path)

def test_dataloader(temp_csv):
    dataloader = napcas.DataLoader(temp_csv, 2)
    inputs, targets = dataloader.next()
    
    assert inputs.shape() == [2, 3], f"Expected input shape [2, 3], got {inputs.shape()}"
    assert targets.shape() == [2, 1], f"Expected target shape [2, 1], got {targets.shape()}"
    
    expected_inputs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    for i in range(inputs.size()):
        assert math.isclose(inputs[i], expected_inputs[i], rel_tol=1e-5), f"Input data mismatch at index {i}"

def test_autograd():
    # Créer un modèle simple
    linear = napcas.Linear(3, 2)
    input_tensor = napcas.Tensor([1, 3], [0.5, 0.5, 0.5])
    output_tensor = napcas.Tensor([1, 2])
    grad_output = napcas.Tensor([1, 2], [1.0, 1.0])
    
    # Forward
    linear.forward(input_tensor, output_tensor)
    
    # Backward avec Autograd
    napcas.Autograd.zero_grad()
    grad_input = napcas.Tensor(input_tensor.shape())
    linear.backward(grad_output, grad_input)
    
    # Vérifier que les gradients sont calculés
    grad_weights = linear.get_grad_weights()
    for i in range(grad_weights.size()):
        assert grad_weights[i] != 0.0, f"Gradient not computed at index {i}"
    
    # Vérifier zero_grad
    napcas.Autograd.zero_grad()
    grad_weights = linear.get_grad_weights()
    for i in range(grad_weights.size()):
        assert math.isclose(grad_weights[i], 0.0, rel_tol=1e-5), f"Gradient not zeroed at index {i}"
