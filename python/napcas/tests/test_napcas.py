import pytest
from napcas import NAPCAS, Tensor

def test_napcas_forward():
    model = NAPCAS(10, 5)
    input = Tensor([2, 10], [float(i) for i in range(20)])
    output = Tensor([2, 5])
    model.forward(input, output)
    assert output.shape() == [2, 5]
