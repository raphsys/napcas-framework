import pytest
from napcas import Transformer, Tensor

def test_transformer():
    model = Transformer(512, 8, 6, 2048)
    input = Tensor([10, 2, 512], [float(i) for i in range(10240)])
    output = Tensor([10, 2, 512])
    model.forward(input, output)
    assert output.shape() == [10, 2, 512]
