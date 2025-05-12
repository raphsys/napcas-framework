import pytest
from napcas import MLP, Tensor

def test_mlp():
    model = MLP([10, 20, 5], "relu")
    input = Tensor([2, 10], [float(i) for i in range(20)])
    output = Tensor([2, 5])
    model.forward(input, output)
    assert output.shape() == [2, 5]
